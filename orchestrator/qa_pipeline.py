"""
QA Intelligence Orchestrator – The Three-Brain Pipeline.

Flow:
  User Story
    → BRAIN 1: Knowledge Graph query (entities, bugs, dependencies)
    → BRAIN 2: Analytical Engine (BVA, EP, State, Pairwise, Risk, Regression)
    → BRAIN 3: RAG retrieve + LLM generate
    → Combine → Structured 12-section QA output
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
import structlog

from graph_builder.neo4j_client import get_graph
from graph_builder.graph_queries import GraphQueryEngine
from rag_engine.vector_store import get_vector_store
from analytics_engine.state_transition import state_engine
from analytics_engine.pairwise_engine import pairwise_engine
from analytics_engine.decision_table import decision_table_builder
from analytics_engine.bug_intelligence import bug_intelligence
from analytics_engine.event_flow_tracer import event_flow_tracer
from analytics_engine.coverage_analyzer import coverage_analyzer
from analytics_engine.regression_analyzer import regression_analyzer
from rag_engine.retriever import retriever
from test_generator.llm_client import llm_client

log = structlog.get_logger()


class QAPipeline:
    def __init__(self):
        self._graph_engine: Optional[GraphQueryEngine] = None

    def _get_graph_engine(self) -> GraphQueryEngine:
        if self._graph_engine is None:
            self._graph_engine = GraphQueryEngine(get_graph())
        return self._graph_engine

    # ════════════════════════════════════════════════════════════════════════
    # MAIN ENTRY POINT
    # ════════════════════════════════════════════════════════════════════════

    # ════════════════════════════════════════════════════════════════════════
    # COMPLEXITY ASSESSMENT
    # ════════════════════════════════════════════════════════════════════════

    def _assess_complexity(
        self,
        feature_understanding: str,
        user_story: str,
        modules: List[Dict],
        apis: List[Dict],
        events: List[Dict],
        states: List[Dict],
        risk_priority: str,
    ) -> tuple:
        """Score feature complexity → returns (level, limits).

        level: 'simple' | 'moderate' | 'complex'
        limits: per-technique scenario caps.
        """
        score = 0

        word_count = len(user_story.split())
        if word_count > 100:
            score += 2
        elif word_count > 50:
            score += 1

        score += min(len(apis), 3)
        score += min(len(events), 2)
        score += min(len(states), 2)
        score += min(len(modules), 3)

        risk_scores = {"P1": 4, "P2": 3, "P3": 2, "P4": 0,
                       "critical": 4, "high": 3, "medium": 2, "low": 0}
        score += risk_scores.get(risk_priority, 1)

        if len(feature_understanding) > 500:
            score += 2
        elif len(feature_understanding) > 200:
            score += 1

        if score <= 5:
            level = "simple"
            limits = {"functional": 8,  "pairwise": 2, "dt": 2,
                      "state": 1, "flow": 2, "edge_count": 3, "gherkin": 3}
        elif score <= 10:
            level = "moderate"
            limits = {"functional": 12, "pairwise": 5, "dt": 5,
                      "state": 3, "flow": 3, "edge_count": 5, "gherkin": 5}
        else:
            level = "complex"
            limits = {"functional": 18, "pairwise": 8, "dt": 8,
                      "state": 5, "flow": 5, "edge_count": 8, "gherkin": 8}

        log.info("complexity_assessed", level=level, score=score)
        return level, limits

    def run(self, user_story: str, project_id: str = "default", clarifying_answers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        log.info("pipeline_start", story_preview=user_story[:80], has_answers=bool(clarifying_answers))

        # Enrich the user story with clarifying answers for all downstream prompts
        enriched_story = user_story
        if clarifying_answers:
            qa_lines = "\n".join(f"Q: {q}\nA: {a}" for q, a in clarifying_answers.items() if a.strip())
            if qa_lines:
                enriched_story = user_story + "\n\nADDITIONAL CONTEXT FROM USER:\n" + qa_lines

        project_store = get_vector_store(project_id)
        project_graph = get_graph(project_id)
        gq = GraphQueryEngine(project_graph)
        feature_name = self._extract_feature_name(user_story)

        # LLM auto-detects module, priority and risk from the story
        story_meta = llm_client.infer_story_metadata(user_story)
        module_name = story_meta["module"]
        priority = story_meta["priority"]
        log.info("story_metadata", module=module_name, priority=priority, risk=story_meta["risk_level"])

        # ──────────────────────────────────────────────────────────────────
        # BRAIN 1: Knowledge Graph
        # ──────────────────────────────────────────────────────────────────
        log.info("brain1_graph_start")
        graph_data = gq.full_impact_analysis(feature_name, module_name=module_name)

        modules = graph_data.get("modules", [])
        impacted_modules = graph_data.get("dependent_modules", [])
        all_bugs = gq.get_all_bugs()
        graph_bugs = graph_data.get("bugs", [])
        test_cases_in_graph = graph_data.get("test_cases", [])
        events = graph_data.get("events", []) or gq.get_all_events()
        apis = graph_data.get("apis", []) or gq.get_all_apis()
        states = graph_data.get("states", [])
        journeys = graph_data.get("journeys", [])

        # If graph has no APIs, infer from user story via LLM (flagged so UI can warn)
        apis_inferred = False
        if not apis:
            apis = llm_client.infer_apis_from_story(user_story)
            apis_inferred = bool(apis)
            log.info("graph_empty_apis_inferred_from_story", count=len(apis))

        primary_module = modules[0] if modules else {"id": module_name, "name": module_name, "criticality": 3}

        # ──────────────────────────────────────────────────────────────────
        # BRAIN 2: Analytical Engine
        # ──────────────────────────────────────────────────────────────────
        log.info("brain2_analytics_start")

        # 2a. LLM-based risk reasoning
        all_risk_areas = llm_client.generate_risk_assessment(
            user_story=user_story,
            feature_name=feature_name,
            module_name=module_name,
            bugs=all_bugs,
            modules=modules + impacted_modules,
            apis=apis,
            events=events,
            states=states,
            priority=priority,
        )
        feature_risk = all_risk_areas[0] if all_risk_areas else {
            "priority": "P2", "risk_score": 0.5, "reasons": []
        }

        # 2c. Pairwise — parameters extracted from RAG context later (after retrieval)
        pairwise_results: List[Dict] = []

        # 2d. Decision table
        dt_conditions, dt_actions = decision_table_builder.infer_conditions_from_story(user_story)
        dt_result = decision_table_builder.build(dt_conditions, dt_actions)

        # 2e. State transition — LLM first for domain-accurate states, heuristic as last resort
        _generic_states = {"active", "inactive", "pending", "processing", "completed", "failed",
                           "cancelled", "rejected", "approved", "submitted", "draft", "published",
                           "archived", "locked", "open", "closed", "resolved", "in_progress", "paused"}
        state_machine_spec = state_engine.infer_from_description(user_story)
        # If heuristic only found generic states (no domain-specific states), prefer LLM
        heuristic_states = set(state_machine_spec.get("states", [])) if state_machine_spec else set()
        if not state_machine_spec or heuristic_states.issubset(_generic_states):
            llm_spec = llm_client.infer_state_machine_from_story(user_story)
            if llm_spec:
                state_machine_spec = llm_spec
                log.info("state_machine_from_llm", entity=llm_spec.get("entity"), states=len(llm_spec.get("states", [])))
        state_tests: Dict = {}
        if state_machine_spec:
            machine = state_engine.build_machine(state_machine_spec)
            state_tests = state_engine.generate_tests(machine)

        # 2f. Bug intelligence
        similar_bugs = bug_intelligence.find_similar_bugs(
            feature_name, primary_module.get("name", ""), user_story, all_bugs
        )
        warnings = bug_intelligence.generate_warnings(similar_bugs, user_story)

        # 2g. Event flow trace
        flow_steps = event_flow_tracer.trace(feature_name, apis, events, [], modules)

        # 2h. Coverage analysis
        all_features = gq.find_nodes("Feature") if hasattr(gq, 'find_nodes') else []
        coverage_gaps = coverage_analyzer.analyze(
            features=all_features,
            test_cases=test_cases_in_graph,
            apis=apis,
            events=events,
            states=states,
            bugs=all_bugs,
        )

        # 2i. Regression analysis
        regression = regression_analyzer.synthesize_from_graph_data(
            feature_name, primary_module.get("name", ""),
            impacted_modules, test_cases_in_graph,
        )
        # If no existing test cases, synthesise regression suite from analytical scenarios
        if not test_cases_in_graph and not regression.get("must_run"):
            regression = self._synthesise_regression_from_scenarios(
                feature_name, primary_module.get("name", module_name), all_bugs
            )

        # ──────────────────────────────────────────────────────────────────
        # BRAIN 3: RAG + LLM Generation
        # ──────────────────────────────────────────────────────────────────
        log.info("brain3_rag_llm_start")

        # Primary retrieval — broad feature context
        rag_result = retriever.retrieve(user_story + " " + feature_name, store=project_store)
        rag_context_str = retriever.build_context_string(rag_result)

        # Targeted retrieval — pulls field names, validation rules, and acceptance criteria
        # from the KB more directly; merged into the context if it adds new content
        _ac_query = f"{feature_name} acceptance criteria validation rules required fields error messages"
        _ac_result = retriever.retrieve(_ac_query, top_k=10, store=project_store)
        _ac_context = retriever.build_context_string(_ac_result, max_chars=3000)
        if _ac_context and _ac_context.strip() not in rag_context_str:
            rag_context_str = rag_context_str + "\n\n" + _ac_context

        # ── Feature status: existing vs new ───────────────────────────────
        # "existing" = KB has relevant prior knowledge (docs + graph entities)
        # "new"      = no meaningful KB match; analysing from scratch
        rag_hits = rag_result.get("raw_hits", [])
        top_score = max((h.get("relevance_score", 0) for h in rag_hits), default=0)
        # Only count entities that came from the graph, not LLM-inferred ones
        has_graph_context = bool(modules or graph_bugs or graph_data.get("events", []))
        if top_score >= 0.70 and rag_result["total_retrieved"] >= 5:
            # Strong RAG match alone is enough — the KB clearly knows this feature
            feature_status = "existing"
            feature_status_reason = (
                f"{rag_result['total_retrieved']} KB chunks matched "
                f"(top similarity {top_score:.0%})"
                + (f"; {len(modules)} module(s) and {len(graph_bugs)} bug(s) in graph." if has_graph_context else ".")
            )
        elif top_score >= 0.60 or (rag_result["total_retrieved"] >= 3 and top_score >= 0.55) or has_graph_context:
            feature_status = "partial"
            feature_status_reason = (
                f"Some prior knowledge found — "
                f"{rag_result['total_retrieved']} KB chunk(s), top similarity {top_score:.0%}."
            )
        else:
            feature_status = "new"
            feature_status_reason = "No matching content in knowledge base — treating as a new feature."

        log.info("feature_status_detected",
                 status=feature_status, top_score=round(top_score, 2),
                 kb_chunks=rag_result["total_retrieved"])

        # Sparse = KB gave us little relevant content; tighten all downstream generation
        kb_sparse = len(rag_context_str.strip()) < 400 or (top_score < 0.50 and rag_result["total_retrieved"] < 3)
        if kb_sparse:
            log.warning("kb_sparse_detected", chars=len(rag_context_str.strip()),
                        top_score=round(top_score, 2), chunks=rag_result["total_retrieved"])

        # LLM: Feature Understanding
        feature_understanding = llm_client.generate_feature_understanding(
            user_story=enriched_story,
            rag_context=rag_context_str,
            module_name=primary_module.get("name", module_name),
            apis=[f"{a.get('method','')} {a.get('endpoint', a.get('name',''))}" for a in apis[:5]],
            events=[e.get("name", "") for e in events[:5]],
            business_rules=[],
            clarifying_answers=clarifying_answers,
        )

        # Agentic complexity assessment — scales all downstream generation
        complexity_level, complexity_limits = self._assess_complexity(
            feature_understanding=feature_understanding,
            user_story=user_story,
            modules=modules + impacted_modules,
            apis=apis,
            events=events,
            states=states,
            risk_priority=feature_risk["priority"],
        )

        # Pairwise parameters extracted from RAG context — uses real values from the documents
        pw_params = llm_client.extract_pairwise_parameters(
            user_story=user_story,
            rag_context=rag_context_str,
            feature_name=feature_name,
        )
        if pw_params:
            pairwise_results = pairwise_engine.generate(pw_params)
            log.info("pairwise_from_rag", params=list(pw_params.keys()), combos=len(pairwise_results))
        else:
            pairwise_results = []
            log.info("pairwise_skipped", reason="no relevant parameters found in RAG context")

        # RAG-based functional scenario generation — scenarios come from the actual ingested documents
        functional_scenarios = llm_client.generate_functional_scenarios(
            user_story=enriched_story,
            rag_context=rag_context_str,
            feature_name=feature_name,
            risk_areas=all_risk_areas[:3],
            scenario_count=complexity_limits.get("functional", 12),
            kb_sparse=kb_sparse,
        )

        analytical_scenarios = self._build_scenarios(
            functional_scenarios=functional_scenarios,
            pairwise_results=pairwise_results,
            dt_result=dt_result,
            state_tests=state_tests,
            flow_steps=flow_steps,
            limits=complexity_limits,
            feature_name=feature_name,
        )

        # LLM: Gherkin, edge cases, API validations — run in parallel (independent of each other)
        risk_context = f"Risk Level: {feature_risk['priority']}. Reasons: {'; '.join(feature_risk['reasons'])}"
        bug_context_str = (
            rag_result.get("context_by_type", {}).get("bug_report", [""])[0][:800]
            if rag_result.get("context_by_type", {}).get("bug_report")
            else ""
        )

        all_regression_entries = regression.get("must_run", []) + regression.get("should_run", [])
        reg_scenario_limit = complexity_limits.get("gherkin", 5)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as _executor:
            _f_gherkin = _executor.submit(
                llm_client.generate_gherkin,
                user_story=user_story,
                scenarios=analytical_scenarios[:complexity_limits["gherkin"]],
                risk_context=risk_context,
                warnings=warnings[:5],
                gherkin_limit=complexity_limits["gherkin"],
            )
            _f_edge = _executor.submit(
                llm_client.generate_edge_cases,
                feature_name=feature_name,
                bug_context=bug_context_str,
                rag_context=rag_context_str,
                state_machine=state_machine_spec,
                edge_count=complexity_limits["edge_count"],
                kb_sparse=kb_sparse,
            )
            _f_api = _executor.submit(
                llm_client.generate_api_validations,
                apis[:6],
                events[:4],
            )
            _f_reg_gherkin = _executor.submit(
                llm_client.generate_regression_gherkin,
                feature_name=feature_name,
                module_name=primary_module.get("name", module_name),
                user_story=user_story,
                impacted_modules=self._format_modules(modules, impacted_modules),
                regression_suite=all_regression_entries,
                scenario_limit=reg_scenario_limit,
            )
            gherkin_tests = _f_gherkin.result()
            edge_cases = _f_edge.result()
            api_validations = _f_api.result()
            regression_gherkin = _f_reg_gherkin.result()

        # ──────────────────────────────────────────────────────────────────
        # Assemble Final Output
        # ──────────────────────────────────────────────────────────────────
        overall_risk = feature_risk["priority"]
        total_scenarios = len(analytical_scenarios) + len(gherkin_tests) + len(edge_cases)

        result = {
            "feature_understanding": feature_understanding,
            "impacted_modules": self._format_modules(modules, impacted_modules),
            "event_flow": flow_steps,
            "risk_areas": all_risk_areas,
            "heads_up_warnings": warnings,
            "test_scenarios": self._format_test_scenarios(analytical_scenarios, edge_cases),
            "gherkin_test_cases": gherkin_tests,
            "regression_suite": self._enrich_regression_entries(regression.get("must_run", []), regression.get("should_run", [])),
            "regression_gherkin": regression_gherkin,
            "test_cases_to_update": regression.get("to_update", []),
            "missing_coverage": coverage_gaps,
            "api_event_validation": api_validations,
            # Metadata
            "user_story": user_story,
            "feature_name": feature_name,
            "detected_module": module_name,
            "detected_priority": priority,
            "total_scenarios": total_scenarios,
            "overall_risk": overall_risk,
            "complexity_level": complexity_level,
            "feature_status": feature_status,
            "feature_status_reason": feature_status_reason,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "graph_stats": gq.get_graph_stats(),
            # Grounding metadata — used by UI to surface data-quality warnings
            "kb_sparse": kb_sparse,
            "apis_inferred": apis_inferred,
        }

        log.info("pipeline_complete",
                 feature=feature_name,
                 risk=overall_risk,
                 scenarios=total_scenarios,
                 warnings=len(warnings))
        return result

    # ════════════════════════════════════════════════════════════════════════
    # Helpers
    # ════════════════════════════════════════════════════════════════════════

    def _extract_feature_name(self, story: str) -> str:
        """Extract a short feature name from a user story."""
        match = re.search(r'i want (?:to )?(.+?)(?:so that|,|\.|$)', story.lower())
        if match:
            raw = match.group(1).strip()
            return raw[:60]
        words = story.split()[:6]
        return " ".join(words)

    def _infer_module(self, story: str) -> str:
        """Guess module from story keywords."""
        story_lower = story.lower()
        module_keywords = {
            "payment": "Payment",
            "checkout": "Checkout",
            "order": "Orders",
            "cart": "Cart",
            "auth": "Authentication",
            "login": "Authentication",
            "user": "User Management",
            "profile": "User Management",
            "notification": "Notifications",
            "email": "Notifications",
            "report": "Reporting",
            "dashboard": "Dashboard",
            "search": "Search",
            "product": "Product Catalogue",
            "inventory": "Inventory",
            "shipping": "Shipping",
        }
        for kw, mod in module_keywords.items():
            if kw in story_lower:
                return mod
        return "Core"

    def _extract_field_specs(self, story: str) -> List[Dict[str, Any]]:
        """Extract testable field specs from a user story using keyword + regex heuristics."""
        import re
        fields: List[Dict[str, Any]] = []
        story_lower = story.lower()
        seen: set = set()

        def add(spec: Dict[str, Any]) -> None:
            if spec["name"] not in seen:
                seen.add(spec["name"])
                fields.append(spec)

        # ── Explicitly quoted / labelled fields in the story ─────────────────
        for m in re.finditer(r'"([^"]{2,40})"', story):
            candidate = m.group(1).lower().replace(" ", "_").replace("-", "_")
            if re.match(r'^[a-z][a-z0-9_]+$', candidate):
                add({"name": candidate, "type": "string", "min": 1, "max": 200, "required": True})

        for m in re.finditer(
            r'\b(?:enter|input|fill in?|provide|type|the)\s+(?:the\s+)?(?:their\s+)?([a-z][a-z _\-]{1,30}?)(?:\s+field|\s+value|\s+input|\b)',
            story_lower,
        ):
            candidate = m.group(1).strip().replace(" ", "_").replace("-", "_")
            if len(candidate) >= 3 and candidate not in seen:
                add({"name": candidate, "type": "string", "min": 1, "max": 200, "required": True})

        # ── Auth / identity ───────────────────────────────────────────────────
        if any(w in story_lower for w in ["username", "user name", "login name", "screen name"]):
            add({"name": "username", "type": "string", "min": 3, "max": 50, "required": True})
        if any(w in story_lower for w in ["password", "passphrase", "pass phrase"]):
            add({"name": "password", "type": "string", "min": 8, "max": 128, "required": True})
        if any(w in story_lower for w in ["otp", "one-time password", "verification code", "pin"]):
            add({"name": "otp", "type": "integer", "min": 100000, "max": 999999, "required": True})
        if "email" in story_lower:
            add({"name": "email", "type": "email", "required": True})
        if any(w in story_lower for w in ["phone", "mobile", "cell", "phone number", "mobile number"]):
            add({"name": "phone_number", "type": "phone", "required": False})

        # ── Names / personal info ─────────────────────────────────────────────
        if any(w in story_lower for w in ["first name", "given name"]):
            add({"name": "first_name", "type": "string", "min": 1, "max": 100, "required": True})
        if any(w in story_lower for w in ["last name", "surname", "family name"]):
            add({"name": "last_name", "type": "string", "min": 1, "max": 100, "required": True})
        if any(w in story_lower for w in ["full name", "display name"]) and "first name" not in story_lower:
            add({"name": "full_name", "type": "string", "min": 2, "max": 200, "required": True})
        if any(w in story_lower for w in [" name", "title", "label"]) and "first name" not in story_lower and "last name" not in story_lower:
            add({"name": "name", "type": "string", "min": 1, "max": 255, "required": True})
        if "age" in story_lower:
            add({"name": "age", "type": "integer", "min": 0, "max": 150, "required": True})
        if any(w in story_lower for w in ["date of birth", "dob", "birth date"]):
            add({"name": "date_of_birth", "type": "date", "required": True})

        # ── Address / location ────────────────────────────────────────────────
        if any(w in story_lower for w in ["address", "street address", "billing address", "shipping address"]):
            add({"name": "address", "type": "string", "min": 5, "max": 500, "required": True})
        if any(w in story_lower for w in ["zip", "postal code", "postcode", "zip code"]):
            add({"name": "zip_code", "type": "string", "min": 4, "max": 10, "required": True})
        if any(w in story_lower for w in ["city", "town"]):
            add({"name": "city", "type": "string", "min": 1, "max": 100, "required": True})
        if "country" in story_lower:
            add({"name": "country", "type": "string", "min": 2, "max": 100, "required": True})

        # ── Financial ─────────────────────────────────────────────────────────
        if any(w in story_lower for w in ["coupon", "promo", "voucher", "coupon code", "promo code"]):
            add({"name": "coupon_code", "type": "string", "min": 3, "max": 20, "required": True})
        if any(w in story_lower for w in ["order total", "cart total", "total amount", "order amount"]):
            add({"name": "order_total", "type": "float", "min": 0.01, "max": 99999.99, "required": True})
        if any(w in story_lower for w in ["amount", "price", "cost", "fee", "charge"]) and "order total" not in story_lower:
            add({"name": "amount", "type": "float", "min": 0.01, "max": 999999.99, "required": True})
        if any(w in story_lower for w in ["discount", "percentage", "percent", "%"]):
            add({"name": "discount_percentage", "type": "float", "min": 0.0, "max": 100.0, "required": False})
        if any(w in story_lower for w in ["credit card", "card number", "card no"]):
            add({"name": "card_number", "type": "string", "min": 13, "max": 19, "required": True})
        if any(w in story_lower for w in ["tax", "vat", "gst"]):
            add({"name": "tax_amount", "type": "float", "min": 0.0, "max": 99999.99, "required": False})

        # ── Inventory / catalogue ─────────────────────────────────────────────
        if any(w in story_lower for w in ["quantity", "qty", "number of items", "number of units"]):
            add({"name": "quantity", "type": "integer", "min": 1, "max": 10000, "required": True})
        if any(w in story_lower for w in ["stock", "inventory", "stock count"]):
            add({"name": "stock_count", "type": "integer", "min": 0, "max": 999999, "required": True})
        if any(w in story_lower for w in ["rating", "score", "star"]):
            add({"name": "rating", "type": "integer", "min": 1, "max": 5, "required": True})
        if any(w in story_lower for w in ["weight", "mass"]):
            add({"name": "weight", "type": "float", "min": 0.001, "max": 9999.999, "required": True})

        # ── Text / content fields ─────────────────────────────────────────────
        if any(w in story_lower for w in ["message", "comment", "note", "bio", "about", "description", "summary", "body"]):
            add({"name": "message", "type": "string", "min": 1, "max": 2000, "required": False})
        if any(w in story_lower for w in ["subject", "headline", "title of"]):
            add({"name": "subject", "type": "string", "min": 1, "max": 500, "required": True})
        if any(w in story_lower for w in ["search", "query", "keyword", "search term"]):
            add({"name": "search_query", "type": "string", "min": 0, "max": 500, "required": False})
        if any(w in story_lower for w in ["url", "link", "website", "web address"]):
            add({"name": "url", "type": "string", "min": 7, "max": 2048, "required": False})
        if any(w in story_lower for w in ["file", "upload", "attachment", "document"]):
            add({"name": "file_name", "type": "string", "min": 1, "max": 255, "required": True})

        # ── System / config fields ────────────────────────────────────────────
        if any(w in story_lower for w in ["retry", "retries", "attempt", "re-try"]):
            add({"name": "retry_count", "type": "integer", "min": 0, "max": 10, "required": False})
        if any(w in story_lower for w in ["limit", "max", "maximum", "threshold", "cap"]):
            add({"name": "limit", "type": "integer", "min": 0, "max": 10000, "required": False})
        if "status" in story_lower:
            add({"name": "status", "type": "enum",
                 "values": ["active", "inactive", "pending", "cancelled"], "required": True})

        # ── Date fields ───────────────────────────────────────────────────────
        if any(w in story_lower for w in ["expiry", "expiration", "expired", "valid until", "deadline"]):
            add({"name": "expiry_date", "type": "date", "required": True})
        elif "date" in story_lower and "date_of_birth" not in seen:
            add({"name": "date", "type": "date", "required": True})

        # ── Fallback: extract generic field from "X field" / "X input" patterns
        if not fields:
            for m in re.finditer(r'\b([a-z][a-z_\- ]{1,25}?)\s+(?:field|input|box|entry)\b', story_lower):
                candidate = m.group(1).strip().replace(" ", "_").replace("-", "_")
                if len(candidate) >= 3:
                    add({"name": candidate, "type": "string", "min": 1, "max": 200, "required": True})
                    break

        # ── Last resort: pick the most noun-like word from the story ─────────
        if not fields:
            _VERBS = {
                "create", "update", "delete", "manage", "submit", "send", "save",
                "load", "fetch", "process", "handle", "allow", "enable", "disable",
                "complete", "start", "stop", "check", "validate", "enter", "provide",
                "edit", "view", "show", "make", "give", "take", "need", "want",
                "have", "will", "should", "able", "that", "with", "this", "from",
                "when", "which", "their", "into", "after", "before", "order",
            }
            # Prefer nouns appearing after "a", "an", or "the"
            noun_m = re.search(r'\b(?:a|an|the)\s+([a-z][a-z\-]{3,25})\b', story_lower)
            if noun_m and noun_m.group(1).replace("-", "") not in _VERBS:
                field_name = noun_m.group(1).replace("-", "_")
            else:
                candidates = [w for w in re.findall(r'\b[a-z]{4,}\b', story_lower)
                              if w not in _VERBS]
                field_name = candidates[0] if candidates else "input"
            fields.append({"name": field_name, "type": "string", "min": 1, "max": 200, "required": True})

        return fields

    @staticmethod
    def _humanize_field(field_name: str) -> str:
        """Convert a programmatic field name to a readable UI label."""
        name = field_name
        for suffix in ("_value", "_field", "_input", "_data", "_text", "_str"):
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break
        return name.replace("_", " ").strip().title()

    @staticmethod
    def _step_val(value: Any) -> str:
        """Return a plain-English instruction for what to type into a field."""
        if value is None:
            return "leave this field completely empty — do not type anything into it"
        if value == "":
            return "clear any existing text so the field is completely blank"
        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "" and len(value) > 0:
                cnt = len(value)
                return f"type just {'a single space' if cnt == 1 else f'{cnt} space characters'} (press the spacebar {'' if cnt == 1 else str(cnt) + ' times'})"
            if len(value) > 20 and len(set(value)) <= 2:
                return f"type a very long string of {len(value)} repeated characters (e.g. keep typing '{value[0]}' until the field is full)"
            if len(value) > 40:
                return f"type a {len(value)}-character string — for example: \"{value[:20]}...\""
            if "<script>" in value.lower():
                return f"type this exact text (XSS test): {value}"
            if "drop table" in value.lower() or (value.count("-") > 1 and "--" in value):
                return f"type this exact text (SQL injection test): {value}"
        return f"type: {value}"

    @staticmethod
    def _fmt_val(value: Any) -> str:
        """Format a raw test value into human-readable display text."""
        if value is None:
            return "no value (leave the field empty)"
        if value == "":
            return "an empty string (submit the field completely blank)"
        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "" and len(value) > 0:
                return f"whitespace only ({len(value)} space character{'s' if len(value) > 1 else ''})"
            if len(value) > 20 and len(set(value)) <= 2:
                return f"a string of {len(value)} repeated characters (e.g. \"{value[:6]}...\")"
            if len(value) > 40:
                return f"a {len(value)}-character string (e.g. \"{value[:15]}...\")"
            if "<script>" in value.lower():
                return f"an XSS payload: {value}"
            if "drop table" in value.lower() or "--" in value:
                return f"an SQL injection string: {value}"
        return str(value)

    @staticmethod
    def _fmt_expected(expected: str, field: str) -> str:
        """Expand a terse BVA/EP expected-result label into a readable test assertion."""
        if not expected:
            return "Verify the system responds correctly."
        e  = expected.strip()
        el = e.lower()

        _MAP = [
            ("reject if required",
             f"The system must reject this submission — '{field}' is a required field. "
             f"A clear validation error message should appear next to the field."),
            ("should be rejected – validation error",
             f"The system must reject this and display a validation error for the required field '{field}'."),
            ("should be accepted – field is optional",
             f"The system should accept no value for '{field}' because it is optional — no error should appear."),
            ("reject – sanitise",
             f"The system must block this input for '{field}'. "
             f"The value contains a potentially malicious payload (XSS / injection) and must be rejected with an error."),
            ("reject – trim & validate",
             f"The system should trim any surrounding whitespace from '{field}' before validating. "
             f"If after trimming the value is still invalid, a validation error must be shown."),
            ("reject if unsigned",
             f"The system must reject this for '{field}' — negative numbers are not accepted. "
             f"A validation error indicating the minimum allowed value should be shown."),
            ("accept – internationalisation",
             f"The system should accept this unicode input for '{field}', "
             f"confirming that international characters are handled correctly."),
            ("accept/reject per spec",
             f"Verify the business rules for '{field}' and confirm the system behaves as specified. "
             f"Acceptance or rejection depends on project configuration."),
            ("accept or reject",
             f"Check the specification for '{field}' to determine the expected outcome and verify accordingly."),
            ("accept if leap year",
             f"The system should accept this date only when the year is a valid leap year. "
             f"For non-leap years, a validation error must be shown."),
            ("accept if min",
             f"The system should accept this value — it meets the minimum length requirement for '{field}'."),
        ]
        for key, msg in _MAP:
            if key in el:
                return msg

        if el == "reject":
            return (f"The system must reject this value for '{field}' and display a clear validation error. "
                    f"No data should be saved.")
        if el == "accept":
            return (f"The system should accept this value for '{field}' and continue processing normally "
                    f"without any errors.")
        if el.startswith("reject"):
            return f"The system must reject this input for '{field}'. {e}"
        if el.startswith("accept"):
            return f"The system should accept this input for '{field}'. {e}"
        return e

    def _build_scenarios(
        self,
        functional_scenarios: List[Dict],
        pairwise_results: List[Dict],
        dt_result: Dict,
        state_tests: Dict,
        flow_steps: List[Dict],
        limits: Optional[Dict] = None,
        feature_name: str = "",
    ) -> List[Dict[str, Any]]:
        """Combine all analytical scenarios into a unified list."""
        if limits is None:
            limits = {"functional": 12, "pairwise": 5, "dt": 5, "state": 3, "flow": 3}
        screen = feature_name.strip() or "the relevant screen"
        # RAG-generated functional scenarios come first — they are the primary test cases
        scenarios: List[Dict[str, Any]] = list(functional_scenarios)
        counter = len(scenarios) + 1

        # Keys that describe system/test setup state — these become preconditions, not typed steps
        _SETUP_KEYS = {
            "auth_state", "user_role", "user_state", "device", "network_condition",
            "input_type", "flag", "status", "platform", "environment", "locale",
        }

        for pw in pairwise_results[:limits["pairwise"]]:
            params = pw.get("parameters", {})
            setup  = {k: v for k, v in params.items() if k in _SETUP_KEYS}
            inputs = {k: v for k, v in params.items() if k not in _SETUP_KEYS}

            preconditions = ["The application is open and the relevant screen is loaded"]
            for k, v in setup.items():
                label = k.replace("_", " ").title()
                preconditions.append(f"{label} is set to: {self._fmt_val(v)}")

            input_steps = []
            for k, v in inputs.items():
                label = k.replace("_", " ")
                val   = self._fmt_val(v)
                if isinstance(v, bool) or str(v).lower() in ("true", "false"):
                    input_steps.append(f"Toggle the '{label}' option to {val}")
                elif isinstance(v, (int, float)):
                    input_steps.append(f"Enter {val} in the '{label}' field")
                else:
                    input_steps.append(f"Select or enter '{self._fmt_val(v)}' for the '{label}' field")

            if not input_steps:
                input_steps.append("Confirm all conditions match the preconditions listed above")

            input_steps += [
                "Submit the form with the settings above",
                "Observe the result and verify no errors or unexpected behaviour appear",
            ]

            readable_combo = ", ".join(
                f"{k.replace('_', ' ')} = {self._fmt_val(v)}" for k, v in params.items()
            )
            scenarios.append({
                "id": pw.get("id", f"TC-PW-{counter:03d}"),
                "type": "pairwise",
                "scenario_type": "functional",
                "title": f"Combination test — {readable_combo}",
                "description": pw.get("description", ""),
                "preconditions": preconditions,
                "steps": input_steps,
                "expected_result": (
                    pw.get("expected_result")
                    or "The system handles this specific combination correctly — "
                       "no errors, no data loss, and the UI reflects the expected outcome."
                ),
                "risk_level": "medium",
                "traceability": "Pairwise testing — validates that this specific parameter combination works correctly",
            })
            counter += 1

        for tc in dt_result.get("test_cases", [])[:limits["dt"]]:
            scenarios.append({
                "id": tc.get("id", f"TC-DT-{counter:03d}"),
                "type": "decision_table",
                "scenario_type": tc.get("scenario_type", "functional"),
                "title": tc.get("title", "Decision table test"),
                "description": tc.get("description", ""),
                "preconditions": tc.get("preconditions", ["System is in a valid ready state"]),
                "steps": tc.get("steps", []),
                "expected_result": tc.get("expected_result", ""),
                "risk_level": tc.get("risk_level", "medium"),
                "traceability": "Decision table — verifies the correct outcome for this specific combination of conditions",
            })
            counter += 1

        for st_list in state_tests.values():
            for tc in st_list[:limits["state"]]:
                scenarios.append({
                    "id": tc.get("id", f"TC-ST-{counter:03d}"),
                    "type": "state_transition",
                    "scenario_type": tc.get("scenario_type", "functional"),
                    "title": tc.get("title", "State transition test"),
                    "description": tc.get("description", ""),
                    "preconditions": tc.get("preconditions", ["System is in the correct starting state for this transition"]),
                    "steps": tc.get("steps", []),
                    "expected_result": tc.get("expected_result", ""),
                    "risk_level": tc.get("risk_level", "medium"),
                    "traceability": "State machine — confirms this transition is handled correctly",
                })
                counter += 1

        # UI-perspective steps for each backend layer — written as what a tester does in the browser
        _LAYER_UI = {
            "UI": {
                "title_prefix": "Submit the action through the UI",
                "preconditions": ["The application is open and the screen is fully loaded"],
                "steps": [
                    "Navigate to the relevant screen or form",
                    "Fill in all required fields with valid input",
                    "Click the submit or confirm button",
                    "Observe the on-screen feedback (loading indicator, success message, or error)",
                ],
                "verify": "The form submits without errors, a success message or confirmation is shown, "
                          "and the user is redirected or the page updates as expected.",
            },
            "API": {
                "title_prefix": "UI submits successfully and receives the correct response",
                "preconditions": ["The user is logged in and has the required permissions"],
                "steps": [
                    "Perform the action on the screen that triggers the request",
                    "Wait for the response — watch for any loading spinner or progress indicator",
                    "Observe whether a success message, error, or redirect appears",
                    "Check that the UI reflects the correct outcome (no stale data, no generic error page)",
                ],
                "verify": "The UI displays the correct response — success state for valid input, "
                          "a clear and specific error message for invalid input. No raw error codes are shown.",
            },
            "DB": {
                "title_prefix": "Data is correctly saved and visible after the action",
                "preconditions": ["The user has completed the action that writes data"],
                "steps": [
                    "Complete the user action that saves or updates data",
                    "Navigate to the view that displays the saved record (list page, detail page, or dashboard)",
                    "Verify all fields show the values that were entered",
                    "Refresh the page and confirm the data is still present (not lost on reload)",
                ],
                "verify": "The saved data appears correctly in the UI with all expected fields populated. "
                          "Data persists after a page refresh. No fields are missing or showing default values.",
            },
            "Event": {
                "title_prefix": "Event-driven UI updates appear after the action",
                "preconditions": ["The user has completed the action that triggers a background event"],
                "steps": [
                    "Complete the action that publishes the event (e.g. submit, approve, confirm)",
                    "Wait a moment for event-driven processing to propagate",
                    "Check the UI for any real-time updates — status changes, counters, banners, or badges",
                    "Refresh the page and verify the updates are still visible (not just in-memory)",
                ],
                "verify": "Any event-driven changes are visible in the UI within the expected time window. "
                          "Status labels, counts, or notifications reflect the new state. "
                          "No stale or outdated data remains displayed.",
            },
            "Consumer": {
                "title_prefix": "Background processing completes and result appears in the UI",
                "preconditions": ["The triggering action has been completed"],
                "steps": [
                    "Trigger the action that kicks off background processing",
                    "Wait for the expected processing time",
                    "Navigate to the page or section that shows the result of background processing",
                    "Verify the outcome is correctly reflected (e.g. status updated, record enriched, job completed)",
                ],
                "verify": "The result of the background job is visible in the UI. "
                          "The status correctly reflects completion. "
                          "No pending or stuck states are shown.",
            },
            "Notification": {
                "title_prefix": "User receives the correct notification after the action",
                "preconditions": ["The action that triggers a notification has been completed"],
                "steps": [
                    "Complete the action that should send a notification",
                    "Open the notification channel — check email inbox, push notifications, or SMS",
                    "Verify the notification arrived for the correct recipient",
                    "Confirm the notification content — subject, body, and any links — are accurate",
                    "Verify no duplicate notifications were sent",
                ],
                "verify": "The notification is received by the correct recipient with accurate content. "
                          "The subject and body match the expected template. "
                          "No duplicate or phantom notifications are sent.",
            },
        }

        for step in flow_steps[:limits["flow"]]:
            layer      = step.get("layer", "")
            validation = step.get("validation_point", "")
            ui         = _LAYER_UI.get(layer, {
                "title_prefix": step.get("action", "Complete the action"),
                "preconditions": ["The application is open and the user is logged in"],
                "steps": [
                    "Perform the relevant action on the screen",
                    "Observe the result in the UI",
                ],
                "verify": validation or "The UI reflects the expected outcome.",
            })

            expected = validation if validation else ui["verify"]
            scenarios.append({
                "id": f"TC-FLOW-{counter:03d}",
                "type": "event_flow",
                "scenario_type": "functional",
                "title": f"{layer} — {ui['title_prefix']}" if layer else ui["title_prefix"],
                "description": expected,
                "preconditions": ui["preconditions"],
                "steps": ui["steps"],
                "expected_result": expected,
                "risk_level": "high" if layer in ("API", "Event", "DB") else "medium",
                "traceability": f"End-to-end event flow — {layer} layer verified through the UI",
            })
            counter += 1

        return scenarios

    def _format_modules(
        self, modules: List[Dict], impacted: List[Dict]
    ) -> List[Dict[str, Any]]:
        result = []
        for m in modules:
            result.append({
                "id": m.get("id", ""),
                "name": m.get("name", ""),
                "criticality": m.get("criticality", 3),
                "impact_type": "DIRECT",
                "description": m.get("description", ""),
            })
        for m in impacted:
            result.append({
                "id": m.get("id", ""),
                "name": m.get("name", ""),
                "criticality": m.get("criticality", 3),
                "impact_type": "TRANSITIVE",
                "description": m.get("description", ""),
            })
        return result

    def _format_test_scenarios(
        self, analytical: List[Dict], edge_cases: List[Any]
    ) -> List[Dict[str, Any]]:
        scenarios = list(analytical)
        for i, ec in enumerate(edge_cases):
            if isinstance(ec, dict):
                scenarios.append({
                    "id": f"TC-EDGE-{i+1:03d}",
                    "type": "edge_case",
                    "scenario_type": "edge",
                    "title": ec.get("title", f"Edge case {i+1}"),
                    "description": ec.get("condition", ""),
                    "preconditions": ["The system is running and accessible"],
                    "steps": [
                        ec.get("condition", "Set up the edge case condition as described"),
                        "Submit or trigger the operation",
                        "Observe how the system responds",
                    ],
                    "expected_result": ec.get("expected", "The system handles this gracefully without crashing or corrupting data"),
                    "risk_level": "high",
                    "traceability": f"Edge case — {ec.get('risk_reason', 'identified as a high-risk historical pattern')}",
                })
        return scenarios

    def _synthesise_regression_from_scenarios(
        self, feature_name: str, module_name: str, bugs: List[Dict]
    ) -> Dict[str, Any]:
        """Generate a meaningful regression suite even when the KB has no test cases.
        Based on past bugs and known risk areas for the feature."""
        must_run = []
        counter = 1

        # Bug-based regression tests
        for bug in bugs[:10]:
            sev = bug.get("severity", "medium").lower()
            priority = "MUST-RUN" if sev in ("critical", "high", "blocker", "p1") else "SHOULD-RUN"
            must_run.append({
                "test_case_id": f"REG-BUG-{counter:03d}",
                "test_case_name": f"Regression: {bug.get('title', bug.get('name', 'Bug regression'))}",
                "module": bug.get("module", bug.get("module_id", module_name)),
                "reason": f"Past bug [{bug.get('id', '')}] severity={sev.upper()}. Root cause: {bug.get('root_cause', bug.get('description', ''))[:100]}",
                "priority": priority,
                "needs_update": False,
                "update_reason": None,
            })
            counter += 1

        # Standard regression tests for the feature
        standard = [
            (f"REG-{counter:03d}", f"Happy path: {feature_name}", "Core functionality must work end-to-end", "MUST-RUN"),
            (f"REG-{counter+1:03d}", f"Negative: invalid input rejected for {feature_name}", "All invalid inputs return correct error codes", "MUST-RUN"),
            (f"REG-{counter+2:03d}", f"Auth: unauthorised access blocked for {module_name}", "Unauthenticated/unauthorised requests return 401/403", "MUST-RUN"),
            (f"REG-{counter+3:03d}", f"Concurrent requests: {feature_name} under parallel load", "No race conditions or data corruption under concurrency", "SHOULD-RUN"),
            (f"REG-{counter+4:03d}", f"Idempotency: duplicate {feature_name} request handled", "Duplicate requests do not cause duplicate side-effects", "SHOULD-RUN"),
        ]
        for tc_id, name, reason, priority in standard:
            must_run.append({
                "test_case_id": tc_id,
                "test_case_name": name,
                "module": module_name,
                "reason": reason,
                "priority": priority,
                "needs_update": False,
                "update_reason": None,
            })

        return {"must_run": must_run, "should_run": [], "to_update": []}

    def _enrich_regression_entries(
        self,
        must_run: List[Dict[str, Any]],
        should_run: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Attach structured trace to each regression entry for display."""
        _trigger_labels = {
            "Validates changed module":        ("MODULE MATCH",   "Directly tests the changed module — re-run to catch direct breakage"),
            "Directly validates changed feature": ("FEATURE MATCH", "Tests the exact feature that changed — must pass"),
            "transitively impacted":           ("TRANSITIVE",     "Depends on the changed module indirectly — verify no side-effects"),
            "Tests API endpoint":              ("API DEPENDENCY", "Hits a changed API — contract and response schema may have shifted"),
            "Tests event":                     ("EVENT DEPENDENCY", "Consumes an event that may have changed payload or timing"),
            "Past bug":                        ("BUG REGRESSION", "Historical defect in this area — ensure the fix did not regress"),
            "Core functionality":              ("SMOKE",          "Happy-path smoke — must pass before any other testing"),
            "invalid input":                   ("NEGATIVE",       "Negative path — invalid inputs must still be rejected correctly"),
            "Unauthenticated":                 ("SECURITY",       "Auth boundary — unauthorised access must still be blocked"),
            "Race condition":                  ("CONCURRENCY",    "Concurrency risk — re-verify under parallel load"),
            "Idempotency":                     ("IDEMPOTENCY",    "Duplicate request protection — must not produce side-effects twice"),
        }

        def _make_trace(entry: Dict[str, Any], priority: str) -> Dict[str, Any]:
            reason = entry.get("reason", "")
            trigger_tag, what_to_verify = "IN SCOPE", "Re-run and confirm all assertions pass"
            for keyword, (tag, desc) in _trigger_labels.items():
                if keyword.lower() in reason.lower():
                    trigger_tag, what_to_verify = tag, desc
                    break
            return {
                **entry,
                "priority": priority,
                "trace": {
                    "trigger": trigger_tag,
                    "why": reason,
                    "what_to_verify": what_to_verify,
                },
            }

        result = []
        for e in must_run:
            result.append(_make_trace(e, "MUST-RUN"))
        for e in should_run:
            result.append(_make_trace(e, "SHOULD-RUN"))
        return result

    def _find_nodes(self, label: str) -> List[Dict]:
        return get_graph().find_nodes(label)


# Singleton
qa_pipeline = QAPipeline()
