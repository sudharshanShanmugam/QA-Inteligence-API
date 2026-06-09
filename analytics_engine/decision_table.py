"""
Decision Table Builder – Brain 2, Component 6

Extracts conditions and actions from user stories and builds
a decision table to identify test combinations.
"""

from typing import Any, Dict, List, Tuple
import itertools


class DecisionTableBuilder:
    def build(self, conditions: List[str], actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build a decision table from conditions and their resulting actions.

        conditions: list of condition strings
        actions: list of {"condition_combo": [T,T,F,...], "action": "..."}
        """
        n = len(conditions)
        # Generate all 2^n combinations
        all_combos = list(itertools.product([True, False], repeat=n))

        table_rows = []
        for combo in all_combos:
            # Find matching action rule
            matched_action = "Undefined – needs specification"
            for action_rule in actions:
                rule_combo = action_rule.get("condition_combo", [])
                if len(rule_combo) == n and tuple(bool(v) for v in rule_combo) == combo:
                    matched_action = action_rule.get("action", "")
                    break

            row = {
                "conditions": {conditions[i]: combo[i] for i in range(n)},
                "action": matched_action,
                "test_case_id": f"DT-{len(table_rows)+1:03d}",
                "scenario_type": "functional" if any(combo) else "negative",
            }
            table_rows.append(row)

        return {
            "conditions": conditions,
            "total_rules": len(all_combos),
            "table": table_rows,
            "test_cases": self._to_test_cases(table_rows, conditions),
        }

    def _to_test_cases(self, rows: List[Dict], conditions: List[str]) -> List[Dict[str, Any]]:
        test_cases = []
        for row in rows:
            cond_str = ", ".join(
                f"{c}={'YES' if v else 'NO'}"
                for c, v in row["conditions"].items()
            )
            test_cases.append({
                "id": row["test_case_id"],
                "type": "decision_table",
                "scenario_type": row["scenario_type"],
                "title": f"Decision: {cond_str}",
                "description": f"Test when: {cond_str}",
                "steps": [
                    f"Set condition '{c}' to {'TRUE' if v else 'FALSE'}"
                    for c, v in row["conditions"].items()
                ] + ["Execute the operation"],
                "expected_result": row["action"],
                "risk_level": "high" if row["scenario_type"] == "negative" else "medium",
            })
        return test_cases

    def infer_conditions_from_story(self, story: str) -> Tuple[List[str], List[Dict]]:
        """
        Heuristic: extract boolean conditions from acceptance criteria / story.
        Returns (conditions, action_rules).
        """
        import re

        conditions = []
        actions = []

        # Common conditional patterns
        patterns = [
            r'if\s+([\w\s]+?)(?:\s+then|\s+,|\s+and|\.|$)',
            r'when\s+([\w\s]+?)(?:\s+then|\s+,|\s+and|\.|$)',
            r'given\s+([\w\s]+?)(?:\s+when|\s+,|\s+and|\.|$)',
        ]

        found_conditions = set()
        for pattern in patterns:
            matches = re.findall(pattern, story.lower())
            for m in matches:
                cond = m.strip()
                if 3 < len(cond) < 60:
                    found_conditions.add(cond)

        # Also check for boolean domain conditions
        domain_conditions = {
            "user is authenticated": any(w in story.lower() for w in ["login", "auth", "logged in", "authenticated"]),
            "payment is valid": any(w in story.lower() for w in ["payment", "card", "pay"]),
            "user has permission": any(w in story.lower() for w in ["permission", "role", "access", "admin"]),
            "input is valid": any(w in story.lower() for w in ["validate", "valid", "invalid", "format"]),
            "stock is available": any(w in story.lower() for w in ["stock", "inventory", "available", "quantity"]),
            "discount is applicable": any(w in story.lower() for w in ["discount", "coupon", "promo", "offer"]),
        }

        for cond, present in domain_conditions.items():
            if present:
                found_conditions.add(cond)

        conditions = list(found_conditions)[:4]  # cap at 4 conditions (16 combinations)

        if not conditions:
            conditions = ["input is valid", "user is authenticated"]

        # Generate simple action rules for the most common combinations
        n = len(conditions)
        all_true = [True] * n
        all_false = [False] * n

        actions = [
            {"condition_combo": all_true, "action": "Proceed with operation successfully"},
            {"condition_combo": all_false, "action": "Reject with appropriate error message"},
        ]

        return conditions, actions


decision_table_builder = DecisionTableBuilder()
