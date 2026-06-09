"""
Equivalence Partitioning Engine – Brain 2, Component 3

Groups inputs into valid and invalid equivalence classes.
One test per class (representative value).
"""

from typing import Any, Dict, List


class EPEngine:
    def partition(self, fields: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate one representative test per equivalence class.

        Input: same field spec as BVAEngine
        Output: list of EP test cases
        """
        test_cases = []
        for field in fields:
            test_cases.extend(self._partition_field(field))
        return test_cases

    def _partition_field(self, field: Dict[str, Any]) -> List[Dict[str, Any]]:
        cases = []
        name = field.get("name", "field")
        ft = field.get("type", "string").lower()
        required = field.get("required", True)

        if ft in ("integer", "float", "number"):
            mn = field.get("min")
            mx = field.get("max")

            # Valid classes
            if mn is not None and mx is not None:
                mid = int((mn + mx) / 2)
                cases.append(self._case(name, mid, "valid", f"Valid range [{mn}, {mx}]", "ACCEPT"))
            elif mn is not None:
                cases.append(self._case(name, mn + 1, "valid", f"Valid: above minimum {mn}", "ACCEPT"))
            elif mx is not None:
                cases.append(self._case(name, mx - 1, "valid", f"Valid: below maximum {mx}", "ACCEPT"))
            else:
                cases.append(self._case(name, 42, "valid", "Valid: typical numeric", "ACCEPT"))

            # Invalid classes
            if mn is not None:
                cases.append(self._case(name, mn - 100, "invalid", f"Below minimum {mn}", "REJECT"))
            if mx is not None:
                cases.append(self._case(name, mx + 100, "invalid", f"Above maximum {mx}", "REJECT"))
            cases.append(self._case(name, "abc",  "invalid", "Non-numeric input",  "REJECT"))
            cases.append(self._case(name, 99.9,   "invalid", "Float for integer field", "REJECT" if ft == "integer" else "ACCEPT"))

        elif ft == "string":
            mn = field.get("min", 1)
            mx = field.get("max")

            # Valid classes
            typical_len = max(mn, 5) if not mx else min(mx, max(mn, 5))
            cases.append(self._case(name, "A" * typical_len, "valid", "Valid string length", "ACCEPT"))
            cases.append(self._case(name, "valid_input_123", "valid", "Valid alphanumeric",   "ACCEPT"))

            # Invalid classes
            if required:
                cases.append(self._case(name, "",    "invalid", "Empty string for required", "REJECT"))
                cases.append(self._case(name, None,  "invalid", "Null for required field",   "REJECT"))
            if mx:
                cases.append(self._case(name, "A" * (mx + 50), "invalid", f"Exceeds max length {mx}", "REJECT"))
            cases.append(self._case(name, "   ", "invalid", "Whitespace only", "REJECT"))

        elif ft == "email":
            cases.append(self._case(name, "user@example.com",    "valid",   "Valid email",             "ACCEPT"))
            cases.append(self._case(name, "user+tag@sub.co.uk",  "valid",   "Valid complex email",     "ACCEPT"))
            cases.append(self._case(name, "notanemail",           "invalid", "Missing @ and domain",   "REJECT"))
            cases.append(self._case(name, "@domain.com",          "invalid", "Missing local part",     "REJECT"))
            cases.append(self._case(name, "user@",                "invalid", "Missing domain",         "REJECT"))
            cases.append(self._case(name, "",                     "invalid", "Empty email",            "REJECT"))

        elif ft == "phone":
            cases.append(self._case(name, "+1-800-555-0100",  "valid",   "Valid international phone", "ACCEPT"))
            cases.append(self._case(name, "5550100",           "valid",   "Valid local number",        "ACCEPT"))
            cases.append(self._case(name, "abc-def-ghij",      "invalid", "Non-numeric phone",         "REJECT"))
            cases.append(self._case(name, "123",               "invalid", "Too short",                 "REJECT"))
            cases.append(self._case(name, "1" * 20,            "invalid", "Too long",                  "REJECT"))

        elif ft == "boolean":
            cases.append(self._case(name, True,    "valid",   "Valid: true",          "ACCEPT"))
            cases.append(self._case(name, False,   "valid",   "Valid: false",         "ACCEPT"))
            cases.append(self._case(name, "yes",   "invalid", "String instead of bool", "REJECT"))
            cases.append(self._case(name, 1,       "invalid", "Integer instead of bool", "ACCEPT or REJECT per spec"))

        elif ft == "enum":
            allowed = field.get("values", [])
            if allowed:
                cases.append(self._case(name, allowed[0], "valid", f"First valid enum: {allowed[0]}", "ACCEPT"))
                if len(allowed) > 1:
                    cases.append(self._case(name, allowed[-1], "valid", f"Last valid enum: {allowed[-1]}", "ACCEPT"))
            cases.append(self._case(name, "UNDEFINED_VALUE", "invalid", "Non-existent enum value", "REJECT"))

        elif ft == "date":
            cases.append(self._case(name, "2024-06-15",  "valid",   "Valid ISO date",      "ACCEPT"))
            cases.append(self._case(name, "15-06-2024",  "invalid", "Wrong format (DD-MM-YYYY)", "REJECT"))
            cases.append(self._case(name, "2024-13-01",  "invalid", "Invalid month 13",    "REJECT"))
            cases.append(self._case(name, "not-a-date",  "invalid", "Non-date string",     "REJECT"))

        return cases

    @staticmethod
    def _case(field: str, value: Any, partition_class: str,
              description: str, expected: str) -> Dict[str, Any]:
        return {
            "field": field,
            "value": value,
            "partition_class": partition_class,
            "description": description,
            "expected_result": expected,
            "technique": "Equivalence Partitioning",
        }


ep_engine = EPEngine()
