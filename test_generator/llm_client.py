"""
LLM client wrapping Ollama via LangChain.

Key design: LLM is called with structured analytical data.
It formats intelligence, it does NOT invent intelligence.
"""

import json
import re
import threading
from typing import Any, Dict, Generator, List, Optional
import structlog

from config import settings

log = structlog.get_logger()

# Limits concurrent LLM calls globally — prevents API timeouts when many files
# are ingested in parallel (e.g. 18 files × 2 LLM calls each = 36 requests).
_llm_semaphore = threading.Semaphore(3)

# Patterns that indicate the LLM returned a template placeholder instead of real content.
# These are leaked prompt instructions, not real test data.
_PLACEHOLDER_RE = re.compile(
    r'\[(?:exact\s+)?(?:field|screen|button|role|user|value|module|endpoint|error|message'
    r'|name|step|precondition|action|outcome|description|test)[^\]]{0,60}\]',
    re.IGNORECASE,
)
# Strings that flag a generic/boilerplate sentence with no grounding
_GENERIC_PHRASES = (
    "navigate to the form",
    "enter a value",
    "submit the operation",
    "perform the action",
    "the expected outcome",
    "the system responds correctly",
    "a valid input",
    "an invalid input",
    "the feature is working",
)


_project_ctx = threading.local()


class LLMClient:
    def __init__(self):
        self._llm = None
        self._streaming_llm = None
        self._usage_lock = threading.Lock()
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._project_usage: dict = {}
        self._project_lock = threading.Lock()
        self._model_override: Optional[str] = None
        self._api_key_override: Optional[str] = None
        self._base_url_override: Optional[str] = None

    def configure(self, model: str, api_key: str, base_url: str) -> None:
        self._model_override = model
        self._api_key_override = api_key
        self._base_url_override = base_url

    def is_api_key_set(self) -> bool:
        return bool(self._api_key_override)

    def set_project_context(self, project_id: Optional[str]) -> None:
        _project_ctx.project_id = project_id

    def clear_project_context(self) -> None:
        _project_ctx.project_id = None

    # ── Token tracking ─────────────────────────────────────────────────────────

    def _record_usage(self, result: Any, prompt: str = "") -> None:
        """Extract token counts from an invoke() response and accumulate them.

        Tries three sources in order:
          1. result.usage_metadata (LangChain standard TypedDict)
          2. result.response_metadata['token_usage'] (OpenAI-style raw dict)
          3. Character-length estimate (4 chars ≈ 1 token) as guaranteed fallback
        """
        inp = out = 0

        # Source 1 — LangChain UsageMetadata
        usage = getattr(result, "usage_metadata", None)
        if usage:
            inp = int(getattr(usage, "input_tokens", 0) or (usage.get("input_tokens", 0) if isinstance(usage, dict) else 0))
            out = int(getattr(usage, "output_tokens", 0) or (usage.get("output_tokens", 0) if isinstance(usage, dict) else 0))

        # Source 2 — OpenAI-style response_metadata
        if not inp and not out:
            meta = getattr(result, "response_metadata", {}) or {}
            tu = meta.get("token_usage") or {}
            inp = int(tu.get("prompt_tokens", 0))
            out = int(tu.get("completion_tokens", 0))

        # Source 3 — character-length estimate (always non-zero after a real call)
        if not inp:
            inp = max(1, len(prompt) // 4)
        if not out:
            content = getattr(result, "content", "") or ""
            out = max(1, len(content) // 4)

        with self._usage_lock:
            self._input_tokens += inp
            self._output_tokens += out

        pid = getattr(_project_ctx, "project_id", None)
        if pid:
            with self._project_lock:
                if pid not in self._project_usage:
                    self._project_usage[pid] = {"input_tokens": 0, "output_tokens": 0}
                self._project_usage[pid]["input_tokens"] += inp
                self._project_usage[pid]["output_tokens"] += out

    def get_usage(self) -> dict:
        """Return a snapshot of accumulated token usage."""
        with self._usage_lock:
            return {"input_tokens": self._input_tokens, "output_tokens": self._output_tokens}

    def reset_usage(self) -> None:
        with self._usage_lock:
            self._input_tokens = 0
            self._output_tokens = 0

    def get_project_usage(self, project_id: str) -> dict:
        with self._project_lock:
            return dict(self._project_usage.get(project_id, {"input_tokens": 0, "output_tokens": 0}))

    def reset_project_usage(self, project_id: str) -> None:
        with self._project_lock:
            self._project_usage[project_id] = {"input_tokens": 0, "output_tokens": 0}

    # ── Anti-hallucination helpers ────────────────────────────────────────────

    @staticmethod
    def _clean_text(text: str) -> str:
        """Strip placeholder patterns left by the LLM echoing prompt templates."""
        return _PLACEHOLDER_RE.sub("", text).strip()

    @staticmethod
    def _is_generic(text: str) -> bool:
        """Return True if the text contains boilerplate/generic filler, not real content."""
        lower = text.lower()
        return any(phrase in lower for phrase in _GENERIC_PHRASES)

    @staticmethod
    def _sanitize_scenarios(scenarios: List[Dict]) -> List[Dict]:
        """Drop or trim scenarios that are mostly template placeholders or generic filler."""
        clean = []
        for s in scenarios:
            title = LLMClient._clean_text(s.get("title", ""))
            if not title or LLMClient._is_generic(title):
                log.warning("hallucination_scenario_dropped", title=s.get("title", ""))
                continue
            s["title"] = title
            s["description"] = LLMClient._clean_text(s.get("description", s.get("title", "")))
            s["expected_result"] = LLMClient._clean_text(s.get("expected_result", ""))
            s["steps"] = [
                LLMClient._clean_text(step) for step in s.get("steps", [])
                if step and not LLMClient._is_generic(step)
            ]
            clean.append(s)
        return clean

    @staticmethod
    def _validate_risk_reasons(reasons: List[str], known_bug_ids: List[str]) -> List[str]:
        """Remove reason strings that cite specific bug IDs not present in the known list."""
        if not known_bug_ids:
            return reasons
        known_lower = {b.lower() for b in known_bug_ids}
        clean = []
        for r in reasons:
            # If the reason mentions a BUG-xxx style ID, verify it's in the known list
            cited = re.findall(r'\bBUG[-_]\w+\b', r, re.IGNORECASE)
            if cited and not any(c.lower() in known_lower for c in cited):
                log.warning("hallucinated_bug_id_stripped", reason=r, cited=cited)
                continue
            clean.append(r)
        return clean or reasons  # never return empty — keep originals if all stripped

    # ── LLM helpers ───────────────────────────────────────────────────────────

    def _get_llm(self, temperature: float = 0):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            openai_api_key=self._api_key_override or settings.DEEPINFRA_API_KEY,
            openai_api_base=self._base_url_override or settings.DEEPINFRA_BASE_URL,
            model=self._model_override or settings.LLM_MODEL,
            temperature=temperature,
            max_tokens=4096,
            seed=42,
            timeout=60,
            max_retries=0,  # no internal SDK retry; generate() handles its own retry logic
        )

    def generate(self, prompt: str, temperature: float = 0, max_retries: int = 0) -> str:
        """Generate text from a prompt. Returns raw string."""
        llm = self._get_llm(temperature)
        for attempt in range(max_retries + 1):
            try:
                with _llm_semaphore:
                    result = llm.invoke(prompt)
                self._record_usage(result, prompt)
                return result.content if hasattr(result, "content") else str(result)
            except Exception as e:
                log.warning("llm_generation_failed", attempt=attempt, error=str(e))
                if attempt == max_retries:
                    return f"[LLM unavailable: {str(e)}]"

    def generate_json(self, prompt: str, fallback: Any = None) -> Any:
        """Generate and parse JSON from LLM. Falls back gracefully."""
        raw = self.generate(prompt + "\n\nRespond with valid JSON only, no markdown fences.")
        return self._parse_json(raw, fallback)

    # ── Chat pre-filters ──────────────────────────────────────────────────────

    _GREETING_RE = re.compile(
        r"^\s*(hi+|hey+|hello+|helo+|howdy|good\s*(morning|afternoon|evening|day|night)|"
        r"how\s+(are\s+you|r\s+u|do\s+you\s+do)|what'?s\s+up|sup|"
        r"thanks?(\s+you)?|thank\s+you|ty|thx|okay|ok|got\s+it|"
        r"great|nice|cool|sure|sounds?\s+good|alright|alright then|"
        r"welcome|good\s+to\s+meet|nice\s+to\s+meet)\s*[!.?]*\s*$",
        re.IGNORECASE,
    )

    _QA_KEYWORDS_RE = re.compile(
        r"test|qa|quality|bug|defect|scenario|gherkin|bdd|regression|"
        r"coverage|risk|assertion|assert|acceptance|criteria|sprint|story|"
        r"feature|automation|manual|selenium|playwright|pytest|unittest|"
        r"api|endpoint|request|response|payload|status\s*code|mock|stub|"
        r"performance|load|stress|security|penetration|sanity|smoke|"
        r"exploratory|boundary|equivalence|partition|edge\s*case|"
        r"traceability|priority|severity|blocker|critical|module|"
        r"integration|system|unit|e2e|end.to.end|release|deploy|"
        # document / KB related
        r"document|doc(s|type|ument)?|upload|ingest|brd|srs|user.?stor|"
        r"knowledge.?base|kb|contract|schema|spec(ification)?|pdf|docx|xlsx|"
        r"accurate|accuracy|better.?result|more.?correct|improve|coverage|"
        r"what.*(accept|need|require|help)|which.*(doc|file|upload)|"
        r"validate|verify|check|pass|fail|expected|actual|steps?|precondition",
        re.IGNORECASE,
    )

    _GREETING_REPLIES = [
        "Hey there! 👋 I'm your QA assistant. Ask me anything about test strategies, test cases, bug triage, or coverage.",
        "Hello! Great to see you. I'm here to help with all things QA — test plans, Gherkin, risk analysis, you name it.",
        "Hi! I'm your QA Intelligence assistant. What QA challenge can I help you with today?",
        "Hey! Ready to talk testing. Ask me about test scenarios, edge cases, regression suites, or anything quality-related.",
        "Good to hear from you! I specialise in QA and software testing. What would you like help with?",
    ]

    _BLOCK_REPLY = (
        "🚫 I'm a QA-focused assistant — I can only help with software testing and quality topics.\n\n"
        "Try asking me about:\n"
        "- **Test cases or scenarios** for your feature\n"
        "- **Bug triage** or defect severity\n"
        "- **Regression suite** coverage\n"
        "- **Gherkin / BDD** test writing\n"
        "- **Risk-based testing** strategies"
    )

    def _classify_message(self, message: str) -> str:
        """Returns 'greeting', 'qa', or 'offtopic'."""
        stripped = message.strip()
        if self._GREETING_RE.match(stripped):
            return "greeting"
        if self._QA_KEYWORDS_RE.search(stripped):
            return "qa"
        # Short messages (≤6 words) that aren't greetings but have no QA keywords
        # are likely small-talk — treat as off-topic only if they look like statements
        # about non-QA subjects (simple word-count heuristic).
        words = stripped.split()
        if len(words) <= 3 and not self._QA_KEYWORDS_RE.search(stripped):
            return "greeting"   # probably a very short casual message — be lenient
        if len(words) > 3 and not self._QA_KEYWORDS_RE.search(stripped):
            return "offtopic"
        return "qa"

    def chat_stream(
        self,
        message: str,
        history: List[Dict[str, str]],
        context: str = "",
    ) -> Generator[str, None, None]:
        """Stream a QA-scoped chat response. History items: [{role, content}]."""
        import random

        intent = self._classify_message(message)

        # ── Instant greeting — no LLM call ───────────────────────────────────
        if intent == "greeting":
            yield random.choice(self._GREETING_REPLIES)
            return

        # ── Instant block for off-topic — no LLM call ────────────────────────
        if intent == "offtopic":
            yield self._BLOCK_REPLY
            return

        # ── QA question — call LLM ───────────────────────────────────────────
        from test_generator.prompt_templates import QA_CHAT_PROMPT
        history_text = "\n".join(
            f"{'USER' if m['role'] == 'user' else 'ASSISTANT'}: {m['content']}"
            for m in history[-10:]  # keep last 10 turns
        ) or "None"
        prompt = QA_CHAT_PROMPT.format(
            context=context[:1500] if context else "No analysis context available.",
            history=history_text,
            message=message,
        )
        yield from self.stream(prompt, temperature=0.3)

    def stream(self, prompt: str, temperature: float = 0) -> Generator[str, None, None]:
        """Stream tokens from the LLM. Estimates usage at ~4 chars per token."""
        llm = self._get_llm(temperature)
        output_chars = 0
        try:
            for chunk in llm.stream(prompt):
                text = chunk.content if hasattr(chunk, "content") else str(chunk)
                output_chars += len(text)
                yield text
        except Exception as e:
            log.warning("llm_stream_failed", error=str(e))
            yield f"[Stream error: {str(e)}]"
        finally:
            # Estimate tokens: input from prompt length, output from streamed chars
            with self._usage_lock:
                self._input_tokens += max(1, len(prompt) // 4)
                self._output_tokens += max(0, output_chars // 4)

    def generate_clarification_questions(self, user_story: str, rag_context: str) -> List[Dict[str, str]]:
        from test_generator.prompt_templates import CLARIFICATION_QUESTIONS_PROMPT
        prompt = CLARIFICATION_QUESTIONS_PROMPT.format(
            user_story=user_story[:1000],
            rag_context=rag_context[:2000] if rag_context else "No knowledge base loaded.",
        )
        result = self.generate_json(prompt, fallback=[])
        if isinstance(result, list):
            return [q for q in result if isinstance(q, dict) and "question" in q]
        return []

    def generate_feature_understanding(
        self,
        user_story: str,
        rag_context: str,
        module_name: str,
        apis: List[str],
        events: List[str],
        business_rules: List[str],
        clarifying_answers: Optional[Dict[str, str]] = None,
    ) -> str:
        from test_generator.prompt_templates import FEATURE_UNDERSTANDING_PROMPT
        enriched_story = user_story
        if clarifying_answers:
            qa_lines = "\n".join(f"Q: {q}\nA: {a}" for q, a in clarifying_answers.items() if a.strip())
            if qa_lines:
                enriched_story = user_story + "\n\nADDITIONAL CONTEXT FROM USER:\n" + qa_lines
        prompt = FEATURE_UNDERSTANDING_PROMPT.format(
            user_story=enriched_story[:1200],
            rag_context=rag_context[:4000],
            module_name=module_name or "Unknown",
            apis=", ".join(apis[:5]) or "None identified",
            events=", ".join(events[:5]) or "None identified",
            business_rules=", ".join(str(r) for r in business_rules[:5]) or "None identified",
        )
        result = self.generate(prompt, temperature=0)
        return result or "Feature understanding not available — upload BRD, SRS, or user story documents in the Ingest tab to enable full analysis."

    def generate_gherkin(
        self,
        user_story: str,
        scenarios: List[Dict[str, Any]],
        risk_context: str,
        warnings: List[Dict[str, Any]],
        gherkin_limit: int = 12,
    ) -> List[Dict[str, Any]]:
        from test_generator.prompt_templates import GHERKIN_GENERATION_PROMPT

        scenarios_text = "\n".join(
            f"{i+1}. [{s.get('type','').upper()}] {s.get('title','')}: {s.get('description','')}"
            for i, s in enumerate(scenarios[:gherkin_limit])
        )
        warnings_text = "\n".join(
            f"- {w.get('warning', '')} → {w.get('recommendation', '')}"
            for w in warnings[:5]
        )

        prompt = GHERKIN_GENERATION_PROMPT.format(
            user_story=user_story[:400],
            scenarios=scenarios_text,
            risk_context=risk_context[:400],
            warnings=warnings_text,
            gherkin_limit=gherkin_limit,
        )
        raw = self.generate(prompt, temperature=0)
        return self._parse_gherkin(raw, scenarios[:gherkin_limit])

    def extract_pairwise_parameters(
        self,
        user_story: str,
        rag_context: str,
        feature_name: str,
    ) -> Dict[str, List[Any]]:
        """Extract real testable parameter combinations from RAG document content."""
        from test_generator.prompt_templates import PAIRWISE_PARAMS_PROMPT

        prompt = PAIRWISE_PARAMS_PROMPT.format(
            user_story=user_story[:600],
            rag_context=rag_context[:4000],
            feature_name=feature_name,
        )
        result = self.generate_json(prompt, fallback={})
        if not isinstance(result, dict):
            return {}
        # Keep only params with at least 2 distinct real values
        return {
            k: list(dict.fromkeys(str(v) for v in vals))
            for k, vals in result.items()
            if isinstance(vals, list) and len(vals) >= 2
        }

    def generate_functional_scenarios(
        self,
        user_story: str,
        rag_context: str,
        feature_name: str,
        risk_areas: List[Dict[str, Any]],
        scenario_count: int = 12,
        kb_sparse: bool = False,
    ) -> List[Dict[str, Any]]:
        """Generate structured functional test scenarios grounded in RAG-retrieved document content."""
        from test_generator.prompt_templates import FUNCTIONAL_SCENARIOS_PROMPT, SPARSE_KB_ADDENDUM

        risk_summary = "; ".join(
            f"{r.get('priority','?')} — {r.get('feature', r.get('module','?'))}: "
            + ", ".join(r.get("reasons", [])[:2])
            for r in risk_areas[:3]
        ) or "No specific risk areas identified."

        # When KB is sparse, cap at 5 scenarios and append conservative addendum
        if kb_sparse:
            scenario_count = min(scenario_count, 5)
            extra = SPARSE_KB_ADDENDUM
        else:
            extra = ""

        prompt = FUNCTIONAL_SCENARIOS_PROMPT.format(
            user_story=user_story[:800],
            rag_context=rag_context[:5000],
            feature_name=feature_name,
            risk_areas=risk_summary,
            scenario_count=scenario_count,
        ) + extra

        raw = self.generate_json(prompt, fallback=[])
        if not isinstance(raw, list):
            return []
        result = []
        for i, s in enumerate(raw):
            if not isinstance(s, dict):
                continue
            result.append({
                "id": s.get("id", f"TC-FUNC-{i+1:03d}"),
                "type": s.get("type", "functional"),
                "scenario_type": s.get("scenario_type", "functional"),
                "title": self._clean_text(s.get("title", "")),
                "description": self._clean_text(s.get("description", s.get("title", ""))),
                "preconditions": [self._clean_text(p) for p in s.get("preconditions", []) if p],
                "steps": [self._clean_text(st) for st in s.get("steps", []) if st],
                "expected_result": self._clean_text(s.get("expected_result", "")),
                "risk_level": s.get("risk_level", "medium"),
                "traceability": self._clean_text(
                    s.get("traceability", "Derived from user story — upload BRD/SRS for KB-grounded traceability")
                ),
                "kb_grounded": not kb_sparse,
            })
        return self._sanitize_scenarios(result)

    def generate_edge_cases(
        self,
        feature_name: str,
        bug_context: str,
        rag_context: str,
        state_machine: Optional[Dict],
        edge_count: int = 8,
        kb_sparse: bool = False,
    ) -> List[Dict[str, Any]]:
        from test_generator.prompt_templates import EDGE_CASE_PROMPT, SPARSE_KB_ADDENDUM

        if kb_sparse:
            edge_count = min(edge_count, 3)

        prompt = EDGE_CASE_PROMPT.format(
            feature_name=feature_name,
            bug_context=bug_context[:800],
            rag_context=rag_context[:4000],
            state_machine=json.dumps(state_machine or {}, default=str),
            edge_count=edge_count,
        ) + (SPARSE_KB_ADDENDUM if kb_sparse else "")

        raw = self.generate_json(prompt, fallback=[])
        if not isinstance(raw, list):
            return []
        # Strip placeholder text from condition / expected fields
        cleaned = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            item["title"] = self._clean_text(item.get("title", ""))
            item["condition"] = self._clean_text(item.get("condition", ""))
            item["expected"] = self._clean_text(item.get("expected", ""))
            if item["title"] and not self._is_generic(item["title"]):
                cleaned.append(item)
            else:
                log.warning("hallucination_edge_case_dropped", title=item.get("title", ""))
        return cleaned

    def generate_signoff_checklist(
        self,
        feature_name: str,
        risk_level: str,
        modules: List[str],
        total_tests: int,
        regression_count: int,
        gaps: List[Dict],
        warnings_count: int,
    ) -> List[Dict[str, Any]]:
        from test_generator.prompt_templates import SIGNOFF_CHECKLIST_PROMPT

        prompt = SIGNOFF_CHECKLIST_PROMPT.format(
            feature_name=feature_name,
            risk_level=risk_level,
            modules=", ".join(modules[:5]),
            total_tests=total_tests,
            regression_count=regression_count,
            gaps=json.dumps([g.get("description", "") for g in gaps[:5]], default=str),
            warnings_count=warnings_count,
        )
        result = self.generate_json(prompt, fallback=[])
        return result if isinstance(result, list) else self._default_checklist()

    def generate_risk_assessment(
        self,
        user_story: str,
        feature_name: str,
        module_name: str,
        bugs: list,
        modules: list,
        apis: list,
        events: list,
        states: list,
        priority: str,
    ) -> list:
        """LLM-based risk reasoning — returns ranked list of risk areas.

        Each item matches the schema:
        { module, feature, risk_score, priority, reasons, past_bug_count }
        """
        bugs_summary = "\n".join(
            f"- [{b.get('severity','?').upper()}] {b.get('title', b.get('name','?'))}: "
            f"{b.get('description', b.get('root_cause', ''))[:120]}"
            for b in bugs[:15]
        ) or "None"

        modules_summary = ", ".join(
            f"{m.get('name','?')} (criticality {m.get('criticality', 3)}/5)"
            for m in modules[:8]
        ) or "None"

        apis_summary = ", ".join(
            f"{a.get('method','?')} {a.get('endpoint', a.get('name','?'))}"
            for a in apis[:8]
        ) or "None"

        events_summary = ", ".join(e.get("name", "?") for e in events[:6]) or "None"
        states_summary = ", ".join(s.get("name", "?") for s in states[:6]) or "None"

        prompt = (
            "You are a Senior QA Architect with 20+ years of risk-based testing experience.\n"
            "Assess the risk areas for the feature below using ONLY the data provided.\n"
            "Do NOT invent bugs, APIs, or modules not listed in the input.\n\n"
            "═══ INPUT DATA ═══\n\n"
            f"Feature: {feature_name}\n"
            f"Module: {module_name}\n"
            f"Detected Priority: {priority}\n\n"
            f"User Story:\n{user_story[:600]}\n\n"
            f"Historical Bugs (from knowledge base):\n{bugs_summary}\n\n"
            f"Modules involved: {modules_summary}\n"
            f"APIs involved: {apis_summary}\n"
            f"Events involved: {events_summary}\n"
            f"States involved: {states_summary}\n\n"
            "═══ RISK SCORING GUIDE ═══\n\n"
            "Score each risk area on these dimensions:\n"
            "- Past bugs: each historical bug in this area adds 0.10 to score\n"
            "- Financial/data sensitivity: payment, auth, personal data → +0.20\n"
            "- Integration complexity: 3+ APIs or events → +0.15\n"
            "- State machine complexity: 4+ states or invalid transitions → +0.10\n"
            "- Security surface: unauthenticated endpoints, user input → +0.15\n"
            "P1 ≥ 0.75 | P2 ≥ 0.50 | P3 ≥ 0.25 | P4 < 0.25\n\n"
            "═══ OUTPUT FORMAT ═══\n\n"
            "Return a JSON array — one object per distinct risk area (max 6), sorted by risk_score descending.\n"
            "Each object must have exactly these fields:\n"
            '  "feature": "the specific sub-feature or area at risk (not just the feature name)"\n'
            '  "module": "the module name from the input"\n'
            '  "risk_score": 0.0–1.0 float derived from the scoring guide above\n'
            '  "priority": "P1" | "P2" | "P3" | "P4"\n'
            '  "reasons": ["specific reason 1 citing actual data", "specific reason 2", ...] (2–4 items)\n'
            '  "past_bug_count": integer — count of bugs from the Historical Bugs list that apply\n\n'
            "Example reasons (be this specific):\n"
            '  "3 past bugs in payment validation — BUG-12, BUG-45, BUG-67"\n'
            '  "POST /api/checkout has no auth check in current spec"\n'
            '  "COUPON_APPLIED → EXPIRED transition not handled in state machine"\n\n'
            "Return valid JSON array only — no markdown fences, no explanation."
        )

        result = self.generate_json(prompt, fallback=[])
        if not isinstance(result, list):
            return self._fallback_risk(feature_name, module_name, bugs, priority)

        known_bug_ids = [b.get("id", b.get("title", "")) for b in bugs]
        valid = []
        for item in result:
            if not isinstance(item, dict):
                continue
            score = float(item.get("risk_score", 0.5))
            score = round(min(1.0, max(0.0, score)), 3)
            item["risk_score"] = score
            item.setdefault("priority", self._risk_label(score))
            item.setdefault("feature", feature_name)
            item.setdefault("module", module_name)
            # Strip reasons that cite specific bug IDs not present in the input list
            item["reasons"] = self._validate_risk_reasons(
                item.get("reasons", ["LLM-assessed risk"]), known_bug_ids
            )
            item.setdefault("past_bug_count", 0)
            valid.append(item)

        log.info("llm_risk_assessment_complete", areas=len(valid), top_priority=valid[0]["priority"] if valid else "?")
        return sorted(valid, key=lambda x: x["risk_score"], reverse=True)

    @staticmethod
    def _risk_label(score: float) -> str:
        if score >= 0.75: return "P1"
        if score >= 0.50: return "P2"
        if score >= 0.25: return "P3"
        return "P4"

    @staticmethod
    def _fallback_risk(feature_name: str, module_name: str, bugs: list, priority: str) -> list:
        """Rule-based fallback when LLM is unavailable."""
        bug_count = len(bugs)
        score = min(1.0, 0.3 + (bug_count * 0.08) + (0.2 if priority in ("high", "critical") else 0))
        return [{
            "feature": feature_name,
            "module": module_name,
            "risk_score": round(score, 3),
            "priority": "P1" if score >= 0.75 else "P2" if score >= 0.5 else "P3",
            "reasons": [f"{bug_count} historical bugs", f"Priority: {priority}"],
            "past_bug_count": bug_count,
        }]

    def infer_story_metadata(self, user_story: str) -> dict:
        """Auto-detect module, priority, and risk level from a user story.

        Returns: {"module": str, "priority": "low|medium|high|critical", "risk_level": "low|medium|high|critical"}
        """
        prompt = (
            "You are a Senior QA Analyst. Read the user story below and classify it.\n\n"
            f"User Story:\n{user_story}\n\n"
            "Answer these three classification questions:\n\n"
            "1. MODULE — which application module does this belong to?\n"
            "   Choose the single best match from: Checkout, Authentication, Payments, Cart, "
            "User Management, Search, Notifications, Reporting, Inventory, Shipping, "
            "Document Management, Learning, Certification, Dashboard, Core\n"
            "   If none match well, infer the closest one from the story context.\n\n"
            "2. PRIORITY — how urgent is this to test? Use these criteria:\n"
            "   critical = financial transactions, authentication, data loss risk\n"
            "   high     = core user-facing workflows, regulatory compliance\n"
            "   medium   = standard features with moderate business impact\n"
            "   low      = cosmetic, informational, or rarely-used features\n\n"
            "3. RISK LEVEL — how likely is this to fail or cause production issues?\n"
            "   critical = complex integrations, past bug history, security-sensitive\n"
            "   high     = multiple system dependencies, real-time processing, state machines\n"
            "   medium   = single system, standard CRUD, moderate validation\n"
            "   low      = read-only, no integrations, simple display logic\n\n"
            "Return ONLY this JSON — no markdown, no explanation:\n"
            "{\n"
            '  "module": "<module name>",\n'
            '  "priority": "low | medium | high | critical",\n'
            '  "risk_level": "low | medium | high | critical",\n'
            '  "reason": "<one sentence citing the specific factor that drove priority and risk — name the actual risk, e.g. \'Payment processing with 3 external API calls and past BUG-45 history\'>"\n'
            "}"
        )
        result = self.generate_json(prompt, fallback={
            "module": "Core", "priority": "medium", "risk_level": "medium", "reason": "fallback"
        })
        if not isinstance(result, dict):
            return {"module": "Core", "priority": "medium", "risk_level": "medium", "reason": "fallback"}

        valid_priorities = {"low", "medium", "high", "critical"}
        if result.get("priority") not in valid_priorities:
            result["priority"] = "medium"
        if result.get("risk_level") not in valid_priorities:
            result["risk_level"] = "medium"
        if not result.get("module"):
            result["module"] = "Core"

        log.info("story_metadata_inferred",
                 module=result["module"], priority=result["priority"],
                 risk=result["risk_level"], reason=result.get("reason", ""))
        return result

    def recommend_chunking_strategy(self, document: dict) -> str:
        """Send a document preview to the LLM and ask which chunking strategy fits best.

        Returns one of: structured | section | paragraph | sentence | semantic
        Falls back to a rule-based guess if the LLM is unavailable.
        """
        doc_type = document.get("doc_type", "unknown")
        fmt = document.get("format", "text")
        content = document.get("content", "")[:2000]

        prompt = (
            "You are a document processing expert building a RAG knowledge base. "
            "Analyze the document excerpt below and pick the single best chunking strategy.\n\n"
            f"Document Type: {doc_type}\n"
            f"Format: {fmt}\n"
            f"Document Excerpt (first 2000 chars):\n{content}\n\n"
            "Choose ONE strategy from this exact list:\n"
            '- "recursive_meta"   : Recursive splitting with rich metadata on every chunk. '
            "Best for most flat documents — recommended default.\n"
            '- "auto"             : Let the system inspect the document and decide automatically.\n'
            '- "recursive"        : Smart recursive split using multiple separator tiers '
            "(double newline → newline → sentence → word). Good for long mixed prose.\n"
            '- "markdown_header"  : Split strictly at # / ## / ### / #### boundaries. '
            "Best when the document is markdown with clear header hierarchy.\n"
            '- "structure"        : Field-aware split for typed JSON documents '
            "(user stories, bug reports, API specs, DB schemas, event definitions).\n"
            '- "sentence"         : Sentence-level splitting. Best for dense technical specs '
            "where every sentence carries critical information.\n"
            '- "paragraph"        : Pure paragraph splitting. Best for narrative prose, '
            "meeting notes, plain-text descriptions.\n"
            '- "fixed"            : Fixed 500-character chunks with 50-char overlap. '
            "Use when structure is absent and uniform chunk size matters.\n"
            '- "token"            : Approximate token-based splitting (~256 tokens per chunk). '
            "Best for LLM context-window alignment.\n\n"
            'Respond with JSON only: {"strategy": "<strategy>", "reason": "<one sentence why>"}'
        )

        result = self.generate_json(
            prompt, fallback={"strategy": self._fallback_strategy(doc_type, fmt)}
        )

        valid = {
            "recursive_meta", "auto", "recursive", "markdown_header",
            "structure", "sentence", "paragraph", "fixed", "token",
        }
        strategy = result.get("strategy", "") if isinstance(result, dict) else ""
        reason = result.get("reason", "") if isinstance(result, dict) else "fallback"

        if strategy not in valid:
            strategy = self._fallback_strategy(doc_type, fmt)
            reason = "fallback — LLM returned an invalid strategy"

        log.info("chunking_strategy_selected", doc_type=doc_type, strategy=strategy, reason=reason)
        return strategy

    @staticmethod
    def _fallback_strategy(doc_type: str, fmt: str) -> str:
        """Rule-based fallback when the LLM is unavailable."""
        if fmt == "json" or doc_type in {
            "user_story", "bug_report", "api_contract", "db_schema", "event_definition"
        }:
            return "structure"
        if doc_type in {"brd", "srs"}:
            return "recursive_meta"
        return "auto"

    def check_document_relevance(self, content: str, filename: str, doc_type: str) -> Dict[str, Any]:
        """Check whether a document is relevant to a software QA / project knowledge base.

        Returns:
            {
              "is_relevant": bool,
              "confidence": "high" | "medium" | "low",
              "detected_type": str,   # what we think the doc actually is
              "reason": str,          # one-sentence explanation
            }
        """
        preview = content[:1500].strip()

        prompt = (
            "You are a QA knowledge base gatekeeper. Decide if this document belongs in a software QA knowledge base.\n\n"
            f"Filename: {filename}\n"
            f"Detected type: {doc_type}\n"
            f"Document preview (first 1500 chars):\n{preview}\n\n"
            "A RELEVANT document is one of:\n"
            "  - Business Requirements Document (BRD), SRS, PRD, product spec\n"
            "  - User story, use case, acceptance criteria\n"
            "  - Bug report, defect log, issue tracker export\n"
            "  - API contract, Swagger/OpenAPI spec, REST endpoint list\n"
            "  - Database schema, data dictionary, ER diagram description\n"
            "  - Test plan, test cases, test strategy, QA checklist\n"
            "  - Event definition, message schema, Kafka topic spec\n"
            "  - Architecture document, technical design document\n"
            "  - Release notes, changelog, sprint backlog\n\n"
            "NOT RELEVANT: resumes/CVs, recipes, news articles, marketing copy, "
            "personal documents, invoices, legal contracts unrelated to software, random text.\n\n"
            "Return ONLY valid JSON — no markdown:\n"
            "{\n"
            '  "is_relevant": true | false,\n'
            '  "confidence": "high" | "medium" | "low",\n'
            '  "detected_type": "what this document actually appears to be",\n'
            '  "reason": "one sentence explaining the decision"\n'
            "}"
        )

        result = self.generate_json(prompt, fallback={
            "is_relevant": True,
            "confidence": "low",
            "detected_type": doc_type,
            "reason": "Could not assess — defaulting to relevant",
        })

        if not isinstance(result, dict):
            return {
                "is_relevant": True,
                "confidence": "low",
                "detected_type": doc_type,
                "reason": "LLM check unavailable — defaulting to relevant",
            }

        result.setdefault("is_relevant", True)
        result.setdefault("confidence", "low")
        result.setdefault("detected_type", doc_type)
        result.setdefault("reason", "")

        log.info("document_relevance_check",
                 filename=filename,
                 is_relevant=result["is_relevant"],
                 confidence=result["confidence"],
                 detected_type=result["detected_type"])
        return result

    def infer_state_machine_from_story(self, user_story: str) -> Optional[Dict[str, Any]]:
        """LLM extracts domain-accurate state machine from user story."""
        prompt = (
            "You are a QA engineer specializing in state-based testing.\n"
            "Extract the complete state machine from this user story.\n\n"
            f"User Story:\n{user_story[:800]}\n\n"
            "Identify ALL states the entity can be in, transitions, and triggering events.\n"
            "Use UPPERCASE_SNAKE_CASE for state names (e.g. COUPON_APPLIED, ORDER_CONFIRMED).\n"
            "Return ONLY valid JSON — no markdown:\n"
            "{\n"
            '  "entity": "the main entity name (e.g. Order, Coupon, Cart)",\n'
            '  "states": ["STATE_1", "STATE_2", ...],\n'
            '  "initial_state": "STARTING_STATE",\n'
            '  "final_states": ["TERMINAL_STATE", ...],\n'
            '  "transitions": [\n'
            '    {"from": "STATE_1", "event": "event_name", "to": "STATE_2", "guard": "optional condition"}\n'
            "  ]\n"
            "}\n"
            "If no meaningful states can be identified, return null."
        )
        result = self.generate_json(prompt, fallback=None)
        if not isinstance(result, dict):
            return None
        states = result.get("states", [])
        transitions = result.get("transitions", [])
        if len(states) < 2 or not transitions:
            return None
        log.info("llm_state_machine_inferred", entity=result.get("entity"), states=len(states), transitions=len(transitions))
        return result

    def infer_apis_from_story(self, user_story: str) -> List[Dict[str, Any]]:
        """LLM infers likely API endpoints from user story when KB has no APIs."""
        prompt = (
            "You are a backend architect. Based on this user story, infer the REST API endpoints needed.\n\n"
            f"User Story:\n{user_story[:800]}\n\n"
            "Return a JSON array of API endpoints. Each must have endpoint path, HTTP method, and short description.\n"
            "Use realistic REST paths (e.g. /api/cart/apply-coupon).\n"
            "Return ONLY valid JSON array — no markdown:\n"
            '[{"endpoint": "/api/...", "method": "POST|GET|PUT|DELETE|PATCH", "description": "..."}]\n'
            "Return empty array [] if nothing can be inferred."
        )
        result = self.generate_json(prompt, fallback=[])
        if not isinstance(result, list):
            return []
        apis = []
        for item in result:
            if isinstance(item, dict) and item.get("endpoint"):
                apis.append({
                    "endpoint": item["endpoint"],
                    "method": item.get("method", "POST"),
                    "name": item.get("description", item["endpoint"]),
                    "description": item.get("description", ""),
                    "source": "inferred_from_story",
                })
        log.info("apis_inferred_from_story", count=len(apis))
        return apis

    def generate_regression_gherkin(
        self,
        feature_name: str,
        module_name: str,
        user_story: str,
        impacted_modules: List[Dict[str, Any]],
        regression_suite: List[Dict[str, Any]],
        scenario_limit: int = 5,
    ) -> List[Dict[str, Any]]:
        from test_generator.prompt_templates import REGRESSION_GHERKIN_PROMPT

        modules_text = "\n".join(
            f"- {m.get('name', m.get('id', ''))}: impact_type={m.get('impact_type', 'TRANSITIVE')}"
            for m in impacted_modules[:8]
        ) or "None identified"

        entries_text = "\n".join(
            f"- [{r.get('priority', '')}] {r.get('test_case_name', '')}: {r.get('reason', '')}"
            for r in regression_suite[:10]
        ) or "None"

        prompt = REGRESSION_GHERKIN_PROMPT.format(
            feature_name=feature_name,
            module_name=module_name,
            user_story=user_story[:400],
            impacted_modules=modules_text,
            regression_entries=entries_text,
            scenario_limit=scenario_limit,
        )
        raw = self.generate(prompt, temperature=0)
        return self._parse_gherkin(raw, [])

    def generate_api_validations(
        self, apis: List[Dict[str, Any]], events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        from test_generator.prompt_templates import API_VALIDATION_PROMPT

        apis_text = json.dumps([
            {"endpoint": a.get("endpoint", a.get("name", "")), "method": a.get("method", "GET")}
            for a in apis[:8]
        ], default=str)
        events_text = json.dumps([
            {"name": e.get("name", ""), "topic": e.get("topic", "")}
            for e in events[:5]
        ], default=str)

        prompt = API_VALIDATION_PROMPT.format(apis=apis_text, events=events_text)
        result = self.generate_json(prompt, fallback=[])
        return result if isinstance(result, list) else self._default_api_validations(apis)

    # ── Parsers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json(raw: str, fallback: Any) -> Any:
        if not raw:
            return fallback
        raw = raw.strip()
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r'(\[.*\]|\{.*\})', raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    pass
        return fallback

    @staticmethod
    def _parse_gherkin(raw: str, scenarios: List[Dict]) -> List[Dict[str, Any]]:
        """Parse LLM-generated Gherkin into structured objects."""
        if not raw:
            return []

        gherkin_tests = []
        current = None
        feature_name = ""

        for line in raw.split("\n"):
            line_stripped = line.strip()
            if line_stripped.startswith("Feature:"):
                feature_name = line_stripped[len("Feature:"):].strip()
            elif line_stripped.startswith("Scenario:") or line_stripped.startswith("Scenario Outline:"):
                if current:
                    gherkin_tests.append(current)
                current = {
                    "feature": feature_name,
                    "scenario_title": line_stripped.split(":", 1)[1].strip(),
                    "given": [],
                    "when": [],
                    "then": [],
                    "tags": [],
                }
            elif line_stripped.startswith("@") and current is None:
                pass  # top-level tags
            elif line_stripped.startswith("@") and current is not None:
                current["tags"].extend(t.strip() for t in line_stripped.split() if t.startswith("@"))
            elif line_stripped.startswith("Given") and current:
                current["given"].append(line_stripped)
            elif line_stripped.startswith("And") or line_stripped.startswith("But"):
                if current:
                    for section in ("then", "when", "given"):
                        if current[section]:
                            current[section].append(line_stripped)
                            break
            elif line_stripped.startswith("When") and current:
                current["when"].append(line_stripped)
            elif line_stripped.startswith("Then") and current:
                current["then"].append(line_stripped)

        if current:
            gherkin_tests.append(current)

        # If parsing failed, generate minimal Gherkin from scenarios
        if not gherkin_tests:
            for s in scenarios[:8]:
                gherkin_tests.append({
                    "feature": feature_name or "Feature",
                    "scenario_title": s.get("title", "Test scenario"),
                    "given": ["Given the system is in a known state"],
                    "when": [f"When {s.get('description', 'the action is performed')}"],
                    "then": [f"Then {s.get('expected_result', 'the system responds correctly')}"],
                    "tags": ["@" + s.get("scenario_type", "functional")],
                })

        return gherkin_tests

    @staticmethod
    def _default_checklist() -> List[Dict[str, Any]]:
        return [
            {"category": "Functional", "item": "All happy-path scenarios pass", "status": "MUST_VERIFY", "owner": "QA"},
            {"category": "Functional", "item": "All negative scenarios return correct errors", "status": "MUST_VERIFY", "owner": "QA"},
            {"category": "Regression", "item": "Full regression suite executed", "status": "MUST_VERIFY", "owner": "QA"},
            {"category": "Security", "item": "Auth/authorization tested", "status": "MUST_VERIFY", "owner": "QA"},
            {"category": "Performance", "item": "API response time ≤ SLA under normal load", "status": "PENDING", "owner": "QA"},
            {"category": "Data", "item": "DB state correct after all operations", "status": "MUST_VERIFY", "owner": "Dev"},
            {"category": "Documentation", "item": "Test evidence attached to story", "status": "PENDING", "owner": "QA"},
        ]

    @staticmethod
    def _default_api_validations(apis: List[Dict]) -> List[Dict[str, Any]]:
        result = []
        for api in apis:
            endpoint = api.get("endpoint", api.get("name", "endpoint"))
            method = api.get("method", "GET")
            result.append({
                "endpoint": endpoint,
                "method": method,
                "validations": [
                    f"200: {method} {endpoint} returns success with valid input",
                    f"400: {method} {endpoint} returns validation error with invalid input",
                    f"401: {method} {endpoint} returns 401 with missing/invalid auth",
                    f"403: {method} {endpoint} returns 403 with insufficient permissions",
                    f"404: {method} {endpoint} returns 404 for non-existent resource",
                    f"500: {method} {endpoint} handles server errors gracefully",
                ],
                "event_triggers": [],
                "db_impacts": [],
            })
        return result


llm_client = LLMClient()
