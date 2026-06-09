"""
High-level graph query helpers used by the Analytical Engine and Orchestrator.

All queries work through the GraphAdapter interface – Neo4j or NetworkX.
"""

from typing import Any, Dict, List, Optional, Set
import structlog
import networkx as nx

log = structlog.get_logger()


class GraphQueryEngine:
    def __init__(self, adapter):
        self._g = adapter

    # ─── Module queries ──────────────────────────────────────────────────────

    def get_module(self, name: str) -> Optional[Dict[str, Any]]:
        modules = self._g.find_nodes("Module")
        name_lower = name.lower()
        # Exact match first
        for m in modules:
            if m.get("name", "").lower() == name_lower or m.get("id", "").lower() == name_lower:
                return m
        # Substring match (e.g. "Checkout" matches "Checkout Module")
        for m in modules:
            m_name = m.get("name", "").lower()
            if name_lower in m_name or m_name in name_lower:
                return m
        return None

    def get_all_modules(self) -> List[Dict[str, Any]]:
        return self._g.find_nodes("Module")

    def get_module_dependencies(self, module_id: str) -> List[Dict[str, Any]]:
        return self._g.get_neighbors(module_id, rel_type="DEPENDS_ON", direction="out")

    def get_dependent_modules(self, module_id: str) -> List[Dict[str, Any]]:
        """Modules that depend ON this module (i.e. will be impacted if this changes)."""
        return self._g.get_neighbors(module_id, rel_type="DEPENDS_ON", direction="in")

    def get_impacted_modules(self, module_id: str, depth: int = 3) -> List[Dict[str, Any]]:
        """BFS through DEPENDS_ON to find all transitively impacted modules."""
        visited: Set[str] = set()
        queue = [module_id]
        impacted = []
        for _ in range(depth):
            next_queue = []
            for mid in queue:
                for nbr in self._g.get_neighbors(mid, rel_type="DEPENDS_ON", direction="in"):
                    nid = nbr.get("id", "")
                    if nid and nid not in visited and nid != module_id:
                        visited.add(nid)
                        impacted.append(nbr)
                        next_queue.append(nid)
            queue = next_queue
        return impacted

    # ─── Feature queries ─────────────────────────────────────────────────────

    def get_features_for_module(self, module_id: str) -> List[Dict[str, Any]]:
        return self._g.get_neighbors(module_id, rel_type="BELONGS_TO", direction="in")

    def find_feature(self, name: str) -> Optional[Dict[str, Any]]:
        features = self._g.find_nodes("Feature")
        name_lower = name.lower()
        for f in features:
            if name_lower in f.get("name", "").lower():
                return f
        return None

    # ─── Bug queries ─────────────────────────────────────────────────────────

    def get_bugs_for_module(self, module_id: str) -> List[Dict[str, Any]]:
        return self._g.get_neighbors(module_id, rel_type="FOUND_IN", direction="in")

    def get_bugs_for_feature(self, feature_id: str) -> List[Dict[str, Any]]:
        return self._g.get_neighbors(feature_id, rel_type="RELATED_TO", direction="in")

    def get_all_bugs(self) -> List[Dict[str, Any]]:
        return self._g.find_nodes("Bug")

    def get_critical_bugs(self) -> List[Dict[str, Any]]:
        bugs = self._g.find_nodes("Bug")
        return [b for b in bugs if b.get("severity", "").lower() in ("critical", "blocker", "p1", "high")]

    def get_bug_patterns(self, module_ids: List[str]) -> Dict[str, int]:
        """Return bug count per module."""
        pattern: Dict[str, int] = {}
        for mid in module_ids:
            bugs = self.get_bugs_for_module(mid)
            pattern[mid] = len(bugs)
        return pattern

    # ─── Test case queries ────────────────────────────────────────────────────

    def get_test_cases_for_feature(self, feature_id: str) -> List[Dict[str, Any]]:
        return self._g.get_neighbors(feature_id, rel_type="VALIDATES", direction="in")

    def get_all_test_cases(self) -> List[Dict[str, Any]]:
        return self._g.find_nodes("TestCase")

    def get_automated_tests(self) -> List[Dict[str, Any]]:
        tcs = self._g.find_nodes("TestCase")
        return [t for t in tcs if t.get("automated") is True or str(t.get("automated", "")).lower() == "true"]

    # ─── API + Event queries ──────────────────────────────────────────────────

    def get_apis_for_module(self, module_id: str) -> List[Dict[str, Any]]:
        return self._g.get_neighbors(module_id, rel_type="BELONGS_TO", direction="in")

    def get_events_triggered_by_feature(self, feature_id: str) -> List[Dict[str, Any]]:
        return self._g.get_neighbors(feature_id, rel_type="TRIGGERS", direction="out")

    def get_db_tables_updated_by_api(self, api_id: str) -> List[Dict[str, Any]]:
        return self._g.get_neighbors(api_id, rel_type="UPDATES", direction="out")

    def get_all_events(self) -> List[Dict[str, Any]]:
        return self._g.find_nodes("Event")

    def get_all_apis(self) -> List[Dict[str, Any]]:
        return self._g.find_nodes("API")

    # ─── State queries ────────────────────────────────────────────────────────

    def get_states_for_feature(self, feature_id: str) -> List[Dict[str, Any]]:
        """Find states related to a feature via the entity property."""
        states = self._g.find_nodes("State")
        return [s for s in states if feature_id in s.get("entity", "")]

    def get_state_transitions(self, state_id: str) -> List[Dict[str, Any]]:
        return self._g.get_neighbors(state_id, rel_type="TRANSITIONS_TO", direction="out")

    # ─── Journey queries ──────────────────────────────────────────────────────

    def get_journeys_spanning_module(self, module_id: str) -> List[Dict[str, Any]]:
        return self._g.get_neighbors(module_id, rel_type="SPANS", direction="in")

    # ─── Impact analysis ─────────────────────────────────────────────────────

    def full_impact_analysis(self, feature_name: str,
                              module_name: str = "") -> Dict[str, Any]:
        """Entry point for the orchestrator – returns everything connected to a feature."""
        feature = self.find_feature(feature_name)
        result: Dict[str, Any] = {
            "feature": feature,
            "modules": [],
            "dependent_modules": [],
            "bugs": [],
            "test_cases": [],
            "events": [],
            "apis": [],
            "states": [],
            "journeys": [],
        }

        if not feature:
            # Fall back to the LLM-detected module if provided
            module_node = self.get_module(module_name) if module_name else None
            if module_node:
                mid = module_node.get("id", "")
                result["modules"] = [module_node]
                result["dependent_modules"] = self.get_impacted_modules(mid)
                result["bugs"] = self.get_bugs_for_module(mid)
                result["apis"] = self.get_apis_for_module(mid)
                result["events"] = self.get_all_events()[:10]
            else:
                # Last resort: return everything from the graph
                result["bugs"] = self.get_all_bugs()[:10]
                result["events"] = self.get_all_events()[:10]
                result["apis"] = self.get_all_apis()[:10]
            return result

        fid = feature.get("id", "")
        mid = feature.get("module_id", "")

        # Find the parent module
        belongs = self._g.get_neighbors(fid, rel_type="BELONGS_TO", direction="out")
        if belongs:
            mid = belongs[0].get("id", mid)
            result["modules"] = [belongs[0]]
        elif mid:
            mod = self._g.get_node(mid)
            if mod:
                result["modules"] = [mod]

        # Impacted modules (downstream)
        if mid:
            result["dependent_modules"] = self.get_impacted_modules(mid)

        result["bugs"] = self.get_bugs_for_feature(fid)
        if mid:
            result["bugs"].extend(self.get_bugs_for_module(mid))
        result["test_cases"] = self.get_test_cases_for_feature(fid)
        result["events"] = self.get_events_triggered_by_feature(fid)
        if mid:
            result["apis"] = self.get_apis_for_module(mid)
        result["states"] = self.get_states_for_feature(fid)
        if mid:
            result["journeys"] = self.get_journeys_spanning_module(mid)

        return result

    # ─── Graph stats ──────────────────────────────────────────────────────────

    def get_graph_stats(self) -> Dict[str, int]:
        all_nodes = self._g.get_all_nodes()
        label_counts: Dict[str, int] = {}
        for node in all_nodes:
            lbl = node.get("label", "Unknown")
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        return {
            "total_nodes": len(all_nodes),
            "total_relationships": len(self._g.get_all_relationships()),
            **{f"{k.lower()}_count": v for k, v in label_counts.items()},
        }
