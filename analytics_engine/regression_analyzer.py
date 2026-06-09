"""
Regression Analyzer – Brain 2, Component 8

Given a changed feature/module, identifies:
1. Which existing test cases MUST re-run
2. Which test cases need to be UPDATED (not just re-run)
3. Priority ordering
"""

from typing import Any, Dict, List, Set
import structlog

log = structlog.get_logger()


MUST_RUN_TRIGGERS = {
    "same_module",        # Test validates the changed module
    "same_feature",       # Test validates the changed feature
    "api_dependency",     # Test hits an API that changed
    "event_dependency",   # Test consumes an event that changed
    "critical_test",      # Test has critical/high priority
}


class RegressionAnalyzer:
    def analyze(
        self,
        changed_feature: str,
        changed_module: str,
        impacted_modules: List[Dict[str, Any]],
        all_test_cases: List[Dict[str, Any]],
        changed_apis: List[str] = None,
        changed_events: List[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Returns must_run and should_run regression suites.
        """
        changed_apis = set(changed_apis or [])
        changed_events = set(changed_events or [])
        impacted_module_ids = {m.get("id", "") for m in impacted_modules}
        impacted_module_ids.add(changed_module)

        must_run: List[Dict[str, Any]] = []
        should_run: List[Dict[str, Any]] = []
        to_update: List[Dict[str, Any]] = []

        for tc in all_test_cases:
            reasons: List[str] = []
            needs_update = False
            update_reason = ""

            tc_module = tc.get("module_id", tc.get("module", ""))
            tc_feature = tc.get("feature_id", tc.get("feature", ""))
            tc_name_lower = tc.get("name", "").lower()
            tc_type = tc.get("type", "").lower()

            # ── MUST-RUN triggers ─────────────────────────────────────────────
            if tc_module == changed_module or changed_module.lower() in tc_name_lower:
                reasons.append(f"Validates changed module: {changed_module}")
                needs_update = True
                update_reason = "Test module coverage changed – update assertions"

            if tc_feature == changed_feature or changed_feature.lower() in tc_name_lower:
                reasons.append(f"Directly validates changed feature: {changed_feature}")
                needs_update = True
                update_reason = "Feature logic changed – update expected results"

            if tc_module in impacted_module_ids:
                reasons.append(f"Module '{tc_module}' is transitively impacted")

            for api in changed_apis:
                if api.lower() in tc_name_lower or api.lower() in (tc.get("description", "")).lower():
                    reasons.append(f"Tests API endpoint: {api}")

            for event in changed_events:
                if event.lower() in tc_name_lower:
                    reasons.append(f"Tests event: {event}")

            # ── Priority classification ───────────────────────────────────────
            if reasons:
                priority_val = tc.get("priority", "medium").lower()
                is_must = (
                    len(reasons) >= 2
                    or tc_module == changed_module
                    or tc_feature == changed_feature
                    or priority_val in ("critical", "high", "p1", "p2")
                )

                entry = {
                    "test_case_id": tc.get("id", ""),
                    "test_case_name": tc.get("name", tc.get("id", "unknown")),
                    "module": tc_module,
                    "reason": "; ".join(reasons),
                    "priority": "MUST-RUN" if is_must else "SHOULD-RUN",
                    "needs_update": needs_update,
                    "update_reason": update_reason if needs_update else None,
                }

                if is_must:
                    must_run.append(entry)
                else:
                    should_run.append(entry)

                if needs_update:
                    to_update.append(entry)

        # Sort by priority within each bucket
        must_run.sort(key=lambda x: x["test_case_id"])
        should_run.sort(key=lambda x: x["test_case_id"])

        log.info("regression_analysis_complete",
                 must_run=len(must_run),
                 should_run=len(should_run),
                 to_update=len(to_update))

        return {
            "must_run": must_run,
            "should_run": should_run,
            "to_update": to_update,
        }

    def synthesize_from_graph_data(
        self,
        feature_name: str,
        module_name: str,
        impacted_modules: List[Dict[str, Any]],
        existing_test_cases: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Entry point when test cases come from the knowledge graph.
        Synthesises a mock regression list if the graph has no TCs yet.
        """
        if existing_test_cases:
            return self.analyze(feature_name, module_name, impacted_modules, existing_test_cases)

        # Synthesise based on impacted modules (no recorded TCs yet)
        must_run = []
        for mod in impacted_modules:
            must_run.append({
                "test_case_id": f"SYNTH-{mod.get('id', 'unknown')}",
                "test_case_name": f"[Synthesised] Smoke test for module: {mod.get('name', '')}",
                "module": mod.get("name", ""),
                "reason": f"Module is transitively impacted by changes to {feature_name}",
                "priority": "MUST-RUN",
                "needs_update": False,
                "update_reason": None,
            })

        return {"must_run": must_run, "should_run": [], "to_update": []}


regression_analyzer = RegressionAnalyzer()
