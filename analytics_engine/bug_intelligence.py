"""
Bug Intelligence Engine – Brain 2, Component 7

Matches new features to historically similar bugs via:
- Module/Feature name similarity
- Root cause pattern matching
- Keyword clustering

Outputs HEADS-UP warnings – not LLM guesses, but pattern-matched evidence.
"""

import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple


SEVERITY_ORDER = {"blocker": 0, "critical": 1, "p1": 1, "high": 2, "p2": 2, "medium": 3, "p3": 3, "low": 4, "p4": 4, "trivial": 5}

# Root cause keywords → actionable recommendation
ROOT_CAUSE_PATTERNS = {
    "race condition": "Race condition risk – ensure thread-safety and idempotency for concurrent requests",
    "null pointer": "Null/undefined reference risk – add null checks at all entry points",
    "null reference": "Null/undefined reference risk – add null checks at all entry points",
    "timeout": "Timeout risk – test with slow network and high load; add circuit breakers",
    "deadlock": "Deadlock risk – review lock ordering; use timeouts on all lock acquisitions",
    "encoding": "Character encoding risk – test with Unicode, special chars, multilingual input",
    "validation": "Input validation gap – ensure server-side validation mirrors client-side rules",
    "auth": "Authentication/authorisation risk – test with expired tokens, missing roles, privilege escalation",
    "cache": "Cache invalidation risk – test after data updates; verify stale data is flushed",
    "pagination": "Pagination edge-case risk – test with 0, 1, max, and overflow page indices",
    "concurrency": "Concurrency risk – test simultaneous operations on the same resource",
    "sql injection": "SQL injection risk – parameterise all queries; fuzz input fields",
    "migration": "Data migration risk – verify existing records are correctly transformed",
    "decimal": "Floating-point/decimal precision risk – test monetary calculations with edge amounts",
    "rounding": "Rounding risk – verify rounding rules for financial calculations",
    "date": "Date/timezone risk – test across DST transitions and timezone boundaries",
    "timezone": "Timezone risk – test across timezone boundaries and DST transitions",
    "limit": "Limit/threshold risk – test at-limit, above-limit, and unlimited scenarios",
    "dependency": "External dependency risk – test with dependency unavailable/slow/returning errors",
    "event": "Event ordering risk – test out-of-order, duplicate, and missing events",
    "rollback": "Transaction rollback risk – test partial failure scenarios",
}

# Minimum score for a bug to be considered relevant (prevents noise from keyword coincidence)
_MIN_RELEVANCE = 0.20


class BugIntelligenceEngine:
    def find_similar_bugs(
        self,
        feature_name: str,
        module_name: str,
        story_text: str,
        all_bugs: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Score all known bugs by relevance to the current feature.
        Only returns bugs above the minimum relevance threshold.
        """
        if not all_bugs:
            return []

        story_lower = story_text.lower()
        feat_lower = feature_name.lower()
        mod_lower = module_name.lower() if module_name else ""

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for bug in all_bugs:
            score = self._relevance_score(bug, feat_lower, mod_lower, story_lower)
            if score >= _MIN_RELEVANCE:
                scored.append((score, bug))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [b for _, b in scored[:top_k]]

    def generate_warnings(
        self,
        similar_bugs: List[Dict[str, Any]],
        story_text: str,
    ) -> List[Dict[str, Any]]:
        """
        Convert similar bugs into actionable HEADS-UP warnings.

        Rules:
        - One warning per bug (includes root-cause pattern in recommendation when found).
        - Root-cause pattern warnings only from bugs — no generic story text scanning.
        - Max 5 warnings total, sorted critical-first.
        """
        warnings: List[Dict[str, Any]] = []
        used_patterns: set = set()

        for bug in similar_bugs:
            sev = bug.get("severity", "medium").lower()
            bug_root = (bug.get("root_cause", "") + " " + bug.get("description", "")).lower()

            # Find the single most relevant root-cause pattern for this bug
            matched_pattern = None
            for keyword, template in ROOT_CAUSE_PATTERNS.items():
                if keyword in bug_root and keyword not in used_patterns:
                    matched_pattern = (keyword, template)
                    used_patterns.add(keyword)
                    break

            bug_display  = bug.get("name", bug.get("title", bug.get("id", "UNKNOWN")))
            root_cause   = bug.get("root_cause", "") or bug.get("description", "")
            root_cause   = root_cause.strip() or None

            # Split "Risk label – actionable advice" → just the advice sentence
            pattern_label  = matched_pattern[0].title().replace("_", " ") if matched_pattern else None
            pattern_advice = None
            if matched_pattern:
                parts = matched_pattern[1].split(" – ", 1)
                pattern_advice = parts[1] if len(parts) == 2 else matched_pattern[1]

            module = (bug.get("module_id") or bug.get("module") or "").strip() or None

            warnings.append({
                "warning": (
                    f"A past {pattern_label.lower()} issue in a similar area may resurface with this change"
                    if pattern_label else
                    "A historically similar bug may affect this feature — review before shipping"
                ),
                "similar_bug_id": bug.get("id", ""),
                "bug_title":    bug_display,
                "severity":     sev,
                "module":       module,
                "pattern":      pattern_label,
                "pattern_advice": pattern_advice,
                "root_cause":   root_cause,
                "recommendation": (
                    "Run focused tests around this area before releasing. "
                    "Confirm the original fix is still in place and covers the scenario described above."
                ),
            })

        # Sort critical/blocker first, cap at 5
        warnings.sort(key=lambda w: SEVERITY_ORDER.get(w["severity"].lower(), 5))
        return warnings[:5]

    def get_bug_patterns(self, all_bugs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate bug statistics for risk calculation."""
        severity_dist: Dict[str, int] = defaultdict(int)
        module_counts: Dict[str, int] = defaultdict(int)
        root_cause_freq: Dict[str, int] = defaultdict(int)

        for bug in all_bugs:
            sev = bug.get("severity", "unknown").lower()
            severity_dist[sev] += 1
            mod = bug.get("module_id", bug.get("module", "unknown"))
            module_counts[mod] += 1
            root = bug.get("root_cause", "").lower()
            for kw in ROOT_CAUSE_PATTERNS:
                if kw in root:
                    root_cause_freq[kw] += 1

        return {
            "total_bugs": len(all_bugs),
            "severity_distribution": dict(severity_dist),
            "bugs_per_module": dict(module_counts),
            "root_cause_frequency": dict(root_cause_freq),
            "hotspot_modules": sorted(module_counts, key=module_counts.get, reverse=True)[:5],
        }

    # ── Scoring ────────────────────────────────────────────────────────────────

    _STOP_WORDS = {
        "that", "this", "with", "from", "have", "will", "been", "when",
        "they", "them", "then", "than", "into", "your", "also", "some",
        "user", "users", "system", "should", "would", "could", "their",
        "want", "able", "make", "when", "where", "which", "while",
    }

    def _relevance_score(
        self, bug: Dict[str, Any], feat_lower: str, mod_lower: str, story_lower: str
    ) -> float:
        score = 0.0

        # Build full bug text — graph nodes use "name" not "title"; root_cause is top-level
        bug_text = " ".join(filter(None, [
            bug.get("name", bug.get("title", "")),        # primary identifier
            bug.get("root_cause", ""),                    # most diagnostic field
            bug.get("description", ""),
            bug.get("feature", bug.get("feature_id", "")),
            bug.get("module_id", bug.get("module", "")),
            bug.get("source_id", ""),                     # e.g. "checkout_bugs" → module signal
        ])).lower()

        # Module match — source_id like "checkout_bugs" carries module signal
        if mod_lower and mod_lower in bug_text:
            score += 0.40

        # Feature name keywords match (individual words, not full phrase)
        feat_words = set(re.findall(r'\b\w{4,}\b', feat_lower)) - self._STOP_WORDS
        if feat_words and feat_words & set(re.findall(r'\b\w{4,}\b', bug_text)):
            score += 0.20

        # Meaningful keyword overlap with the story
        story_words = set(re.findall(r'\b\w{4,}\b', story_lower)) - self._STOP_WORDS
        bug_words = set(re.findall(r'\b\w{4,}\b', bug_text)) - self._STOP_WORDS
        overlap = len(story_words & bug_words)
        score += min(0.30, overlap * 0.06)

        # Severity boost
        sev = bug.get("severity", "medium").lower()
        sev_boost = {0: 0.10, 1: 0.10, 2: 0.07, 3: 0.04, 4: 0.01, 5: 0.0}
        score += sev_boost.get(SEVERITY_ORDER.get(sev, 3), 0.0)

        return score


bug_intelligence = BugIntelligenceEngine()
