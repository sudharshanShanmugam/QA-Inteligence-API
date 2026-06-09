"""
Boundary Value Analysis Engine – Brain 2, Component 2

For each numeric/string field, generates test values at and around boundaries.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FieldSpec:
    name: str
    field_type: str  # integer | float | string | date | enum
    min_val: Optional[Any] = None
    max_val: Optional[Any] = None
    allowed_values: Optional[List[Any]] = None
    required: bool = True


class BVAEngine:
    def analyze(self, fields: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate BVA test cases for a list of field specs.

        Input format:
        [{"name": "age", "type": "integer", "min": 0, "max": 120}, ...]
        """
        test_cases = []
        for field in fields:
            spec = self._parse_spec(field)
            cases = self._generate_for_field(spec)
            test_cases.extend(cases)
        return test_cases

    def _parse_spec(self, field: Dict[str, Any]) -> FieldSpec:
        return FieldSpec(
            name=field.get("name", "field"),
            field_type=field.get("type", "string"),
            min_val=field.get("min"),
            max_val=field.get("max"),
            allowed_values=field.get("values"),
            required=field.get("required", True),
        )

    def _generate_for_field(self, spec: FieldSpec) -> List[Dict[str, Any]]:
        cases = []
        ft = spec.field_type.lower()

        if ft in ("integer", "float", "number"):
            cases.extend(self._numeric_boundaries(spec))
        elif ft == "string":
            cases.extend(self._string_boundaries(spec))
        elif ft == "date":
            cases.extend(self._date_boundaries(spec))
        elif ft == "enum":
            cases.extend(self._enum_boundaries(spec))

        # Always add null/empty test for non-required fields
        if not spec.required:
            cases.append(self._make_case(
                spec.name, None,
                "boundary", "null/empty for optional field",
                "Should be accepted – field is optional", "edge"
            ))
        # Always add null test for required fields (negative)
        cases.append(self._make_case(
            spec.name, None,
            "boundary", "null/missing for required field",
            "Should be rejected – validation error", "negative"
        ))

        return cases

    def _numeric_boundaries(self, spec: FieldSpec) -> List[Dict[str, Any]]:
        cases = []
        mn, mx = spec.min_val, spec.max_val
        is_int = spec.field_type.lower() == "integer"
        delta = 1 if is_int else 0.001

        if mn is not None:
            cases.append(self._make_case(spec.name, mn - delta,   "boundary", f"below minimum ({mn})", "REJECT", "negative"))
            cases.append(self._make_case(spec.name, mn,           "boundary", f"at minimum ({mn})",    "ACCEPT", "functional"))
            cases.append(self._make_case(spec.name, mn + delta,   "boundary", f"just above min ({mn})", "ACCEPT", "boundary"))

        if mx is not None:
            cases.append(self._make_case(spec.name, mx - delta,   "boundary", f"just below max ({mx})", "ACCEPT", "boundary"))
            cases.append(self._make_case(spec.name, mx,           "boundary", f"at maximum ({mx})",     "ACCEPT", "functional"))
            cases.append(self._make_case(spec.name, mx + delta,   "boundary", f"above maximum ({mx})",  "REJECT", "negative"))

        if mn is not None and mx is not None:
            mid = (mn + mx) / 2
            mid = int(mid) if is_int else round(mid, 3)
            cases.append(self._make_case(spec.name, mid, "boundary", "typical valid value (midpoint)", "ACCEPT", "functional"))

        # Always test zero and negative if not already covered
        if mn is None or mn > 0:
            cases.append(self._make_case(spec.name, 0,  "boundary", "zero value",     "ACCEPT/REJECT per spec", "edge"))
        if mn is None or mn > -1:
            cases.append(self._make_case(spec.name, -1, "boundary", "negative value", "REJECT if unsigned",     "negative"))

        return cases

    def _string_boundaries(self, spec: FieldSpec) -> List[Dict[str, Any]]:
        cases = []
        mn = spec.min_val or 0
        mx = spec.max_val

        cases.append(self._make_case(spec.name, "",      "boundary", "empty string",         "REJECT if required",     "negative"))
        cases.append(self._make_case(spec.name, " ",     "boundary", "whitespace only",       "REJECT – trim & validate", "negative"))
        cases.append(self._make_case(spec.name, "A",     "boundary", "single character",      "ACCEPT if min=1",        "boundary"))

        if mn and mn > 1:
            cases.append(self._make_case(spec.name, "A" * (mn - 1), "boundary", f"below min length ({mn})", "REJECT", "negative"))
            cases.append(self._make_case(spec.name, "A" * mn,       "boundary", f"at min length ({mn})",    "ACCEPT", "boundary"))

        if mx:
            cases.append(self._make_case(spec.name, "A" * mx,       "boundary", f"at max length ({mx})",        "ACCEPT", "boundary"))
            cases.append(self._make_case(spec.name, "A" * (mx + 1), "boundary", f"exceeds max length ({mx})",   "REJECT", "negative"))

        # Special characters
        cases.append(self._make_case(spec.name, "<script>alert(1)</script>", "boundary", "XSS payload", "REJECT – sanitise", "security"))
        cases.append(self._make_case(spec.name, "'; DROP TABLE users; --",    "boundary", "SQL injection", "REJECT – sanitise", "security"))
        cases.append(self._make_case(spec.name, "àéîõü",  "boundary", "unicode characters", "ACCEPT – internationalisation", "edge"))

        return cases

    def _date_boundaries(self, spec: FieldSpec) -> List[Dict[str, Any]]:
        return [
            self._make_case(spec.name, "1900-01-01", "boundary", "far past date",    "ACCEPT/REJECT per spec", "boundary"),
            self._make_case(spec.name, "2099-12-31", "boundary", "far future date",  "ACCEPT/REJECT per spec", "boundary"),
            self._make_case(spec.name, "TODAY",      "boundary", "today's date",     "ACCEPT",                 "functional"),
            self._make_case(spec.name, "TODAY-1",    "boundary", "yesterday",        "ACCEPT/REJECT per spec", "boundary"),
            self._make_case(spec.name, "TODAY+1",    "boundary", "tomorrow",         "ACCEPT/REJECT per spec", "boundary"),
            self._make_case(spec.name, "2024-02-29", "boundary", "leap day",         "ACCEPT if leap year",    "edge"),
            self._make_case(spec.name, "2023-02-29", "boundary", "invalid leap day", "REJECT",                 "negative"),
            self._make_case(spec.name, "INVALID",    "boundary", "invalid date format", "REJECT",              "negative"),
        ]

    def _enum_boundaries(self, spec: FieldSpec) -> List[Dict[str, Any]]:
        cases = []
        if not spec.allowed_values:
            return cases
        for val in spec.allowed_values:
            cases.append(self._make_case(spec.name, val, "boundary", f"valid enum: {val}", "ACCEPT", "functional"))
        cases.append(self._make_case(spec.name, "INVALID_VALUE", "boundary", "invalid enum value", "REJECT", "negative"))
        cases.append(self._make_case(spec.name, "",              "boundary", "empty enum",          "REJECT", "negative"))
        return cases

    @staticmethod
    def _make_case(field: str, value: Any, test_type: str, description: str,
                   expected: str, scenario_type: str) -> Dict[str, Any]:
        return {
            "field": field,
            "value": value,
            "test_type": test_type,
            "description": description,
            "expected_result": expected,
            "scenario_type": scenario_type,
        }


bva_engine = BVAEngine()
