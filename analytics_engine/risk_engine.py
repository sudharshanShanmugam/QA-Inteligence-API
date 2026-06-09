"""
Risk Engine – Brain 2, Component 1

Risk Score = f(past_bugs, module_criticality, change_impact, bug_severity_weight)

Score range: 0.0 – 1.0
P1 ≥ 0.75 | P2 ≥ 0.50 | P3 ≥ 0.25 | P4 < 0.25
"""

from typing import Any, Dict, List, Tuple
import math


SEVERITY_WEIGHT = {
    "blocker": 1.0, "critical": 0.9, "p1": 0.9,
    "high": 0.7, "p2": 0.7,
    "medium": 0.5, "p3": 0.5,
    "low": 0.2, "p4": 0.2,
    "trivial": 0.1,
}

CRITICALITY_NORM = {1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}


class RiskEngine:
    # Weights must sum to 1.0
    W_BUG_COUNT = 0.35
    W_CRITICALITY = 0.25
    W_CHANGE_IMPACT = 0.25
    W_SEVERITY = 0.15

    def score_module(
        self,
        module: Dict[str, Any],
        bugs: List[Dict[str, Any]],
        impacted_modules: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute risk score for a single module."""
        # Factor 1: normalised bug count (sigmoid-smoothed)
        bug_count = len(bugs)
        bug_score = 1 - (1 / (1 + 0.3 * bug_count))  # sigmoid

        # Factor 2: module criticality (1-5 → 0.2-1.0)
        criticality = int(module.get("criticality", 3))
        criticality_score = CRITICALITY_NORM.get(criticality, 0.6)

        # Factor 3: change impact (downstream modules)
        impact_count = len(impacted_modules)
        impact_score = min(1.0, impact_count / 10.0)

        # Factor 4: severity-weighted bug score
        if bugs:
            sev_scores = []
            for bug in bugs:
                sev = bug.get("severity", "medium").lower()
                sev_scores.append(SEVERITY_WEIGHT.get(sev, 0.5))
            severity_score = sum(sev_scores) / len(sev_scores)
        else:
            severity_score = 0.0

        risk_score = (
            self.W_BUG_COUNT * bug_score
            + self.W_CRITICALITY * criticality_score
            + self.W_CHANGE_IMPACT * impact_score
            + self.W_SEVERITY * severity_score
        )
        risk_score = round(min(1.0, risk_score), 3)

        reasons = []
        if bug_count > 5:
            reasons.append(f"High historical bug density ({bug_count} bugs)")
        elif bug_count > 2:
            reasons.append(f"Moderate historical bug density ({bug_count} bugs)")
        if criticality >= 4:
            reasons.append(f"High criticality module (level {criticality}/5)")
        if impact_count > 3:
            reasons.append(f"Affects {impact_count} downstream modules")
        if severity_score > 0.7:
            reasons.append("Prior critical/blocker severity bugs")
        if not reasons:
            reasons.append("Standard risk profile")

        return {
            "module": module.get("name", module.get("id", "unknown")),
            "feature": "",
            "risk_score": risk_score,
            "priority": self._priority_label(risk_score),
            "reasons": reasons,
            "past_bug_count": bug_count,
        }

    def score_feature(
        self,
        feature: Dict[str, Any],
        bugs: List[Dict[str, Any]],
        module_risk: float,
        has_state_machine: bool = False,
        has_external_events: bool = False,
    ) -> Dict[str, Any]:
        """Compute risk score for a specific feature."""
        bug_count = len(bugs)
        bug_score = 1 - (1 / (1 + 0.4 * bug_count))

        # Feature priority modifier
        priority = feature.get("priority", "medium").lower()
        prio_modifier = {"critical": 1.0, "high": 0.8, "medium": 0.5, "low": 0.2}.get(priority, 0.5)

        # State machine complexity adds risk
        complexity_bonus = 0.1 if has_state_machine else 0.0
        complexity_bonus += 0.1 if has_external_events else 0.0

        risk_score = round(min(1.0,
            0.4 * bug_score
            + 0.3 * prio_modifier
            + 0.15 * module_risk
            + 0.15 * min(1.0, complexity_bonus)
        ), 3)

        reasons = []
        if bug_count > 0:
            reasons.append(f"{bug_count} historical bugs on this feature")
        if priority in ("critical", "high"):
            reasons.append(f"Priority: {priority}")
        if has_state_machine:
            reasons.append("Complex state machine – transition testing critical")
        if has_external_events:
            reasons.append("Involves external events – async risk")

        return {
            "module": feature.get("module_id", ""),
            "feature": feature.get("name", ""),
            "risk_score": risk_score,
            "priority": self._priority_label(risk_score),
            "reasons": reasons,
            "past_bug_count": bug_count,
        }

    def rank_risk_areas(self, risk_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(risk_items, key=lambda x: x["risk_score"], reverse=True)

    @staticmethod
    def _priority_label(score: float) -> str:
        if score >= 0.75:
            return "P1"
        elif score >= 0.50:
            return "P2"
        elif score >= 0.25:
            return "P3"
        return "P4"


risk_engine = RiskEngine()
