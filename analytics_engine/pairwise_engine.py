"""
Pairwise (AllPairs) Testing Engine – Brain 2, Component 5

Implements a greedy AllPairs algorithm to generate the minimum set of test cases
that covers all two-way combinations of parameter values.
"""

import random
from typing import Any, Dict, List, Set, Tuple


class PairwiseEngine:
    def generate(self, parameters: Dict[str, List[Any]], seed: int = 42) -> List[Dict[str, Any]]:
        """
        Generate a minimal pairwise test suite.

        parameters: {"param_name": [val1, val2, ...], ...}
        returns: list of test case dicts
        """
        if not parameters or len(parameters) < 2:
            return []

        random.seed(seed)
        param_names = list(parameters.keys())
        param_values = [parameters[p] for p in param_names]
        n = len(param_names)

        # Build set of all pairs to cover
        pairs_to_cover: Set[Tuple] = set()
        for i in range(n):
            for j in range(i + 1, n):
                for vi in param_values[i]:
                    for vj in param_values[j]:
                        pairs_to_cover.add((i, str(vi), j, str(vj)))

        test_cases: List[Dict[str, Any]] = []
        max_iterations = len(pairs_to_cover) * 3  # safety cap
        iteration = 0

        while pairs_to_cover and iteration < max_iterations:
            iteration += 1
            best_candidate = None
            best_coverage = -1

            # Try N random candidates, pick the one covering most uncovered pairs
            candidates = []
            for _ in range(50):
                candidate = {param_names[k]: random.choice(param_values[k]) for k in range(n)}
                candidates.append(candidate)

            for candidate in candidates:
                coverage = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        pair = (i, str(candidate[param_names[i]]), j, str(candidate[param_names[j]]))
                        if pair in pairs_to_cover:
                            coverage += 1
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_candidate = candidate

            if best_candidate:
                test_cases.append(best_candidate)
                # Remove all pairs covered by this test case
                for i in range(n):
                    for j in range(i + 1, n):
                        pair = (i, str(best_candidate[param_names[i]]), j, str(best_candidate[param_names[j]]))
                        pairs_to_cover.discard(pair)

        # Annotate with test IDs
        annotated = []
        for idx, tc in enumerate(test_cases, 1):
            annotated.append({
                "id": f"PW-{idx:03d}",
                "type": "pairwise",
                "scenario_type": "functional",
                "parameters": tc,
                "title": f"Pairwise Combination {idx}: {', '.join(f'{k}={v}' for k, v in tc.items())}",
                "description": f"Test pairwise combination covering parameters: {list(tc.keys())}",
                "expected_result": "System handles combination correctly",
            })
        return annotated

    def extract_parameters_from_story(self, story: str) -> Dict[str, List[Any]]:
        """
        Heuristic extraction of testable parameters from a user story.
        Returns a parameter map for pairwise generation.
        """
        import re

        params: Dict[str, List[Any]] = {}

        # Role/user type patterns
        role_match = re.search(r'as (?:a|an) (\w+)', story.lower())
        if role_match:
            params["user_role"] = [role_match.group(1), "admin", "guest"]

        # Status/state mentions
        statuses = re.findall(r'\b(active|inactive|pending|approved|rejected|draft|published)\b', story.lower())
        if statuses:
            params["status"] = list(set(statuses))

        # Quantity/amount mentions
        if any(w in story.lower() for w in ["amount", "quantity", "count", "number", "limit"]):
            params["quantity"] = [0, 1, 10, 100, -1]

        # Boolean flag patterns
        if any(w in story.lower() for w in ["enabled", "disabled", "true", "false", "flag", "toggle"]):
            params["flag"] = [True, False]

        # Device/platform patterns
        if any(w in story.lower() for w in ["mobile", "desktop", "app", "browser", "device"]):
            params["device"] = ["mobile", "desktop", "tablet"]

        # Payment/pricing patterns
        if any(w in story.lower() for w in ["payment", "price", "discount", "coupon", "promo"]):
            params["payment_type"] = ["credit_card", "debit_card", "upi", "wallet", "netbanking"]
            params["discount_applied"] = [True, False]

        # Auth patterns
        if any(w in story.lower() for w in ["login", "auth", "token", "session", "otp"]):
            params["auth_state"] = ["authenticated", "unauthenticated", "expired_token"]

        # Default minimal params if extraction found nothing
        if not params:
            params = {
                "input_type": ["valid", "invalid", "boundary"],
                "user_state": ["new", "existing", "inactive"],
                "network_condition": ["online", "offline", "slow"],
            }

        return params


pairwise_engine = PairwiseEngine()
