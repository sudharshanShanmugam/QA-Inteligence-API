"""
State Transition Testing Engine – Brain 2, Component 4

Generates 0-switch (single transitions) and N-switch (sequences) test cases.
Also identifies invalid transition tests (negative scenarios).
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Transition:
    from_state: str
    event: str
    to_state: str
    guard: str = ""
    action: str = ""


@dataclass
class StateMachine:
    entity: str
    states: List[str]
    initial_state: str
    final_states: List[str]
    transitions: List[Transition]


class StateTransitionEngine:
    def build_machine(self, spec: Dict[str, Any]) -> StateMachine:
        """Build a StateMachine from a dict spec."""
        transitions = [
            Transition(
                from_state=t["from"],
                event=t["event"],
                to_state=t["to"],
                guard=t.get("guard", ""),
                action=t.get("action", ""),
            )
            for t in spec.get("transitions", [])
        ]
        return StateMachine(
            entity=spec.get("entity", "Entity"),
            states=spec.get("states", []),
            initial_state=spec.get("initial_state", spec.get("states", ["start"])[0]),
            final_states=spec.get("final_states", []),
            transitions=transitions,
        )

    def generate_tests(self, machine: StateMachine) -> Dict[str, List[Dict[str, Any]]]:
        """Generate all test categories for a state machine."""
        return {
            "valid_transitions": self._valid_transition_tests(machine),
            "invalid_transitions": self._invalid_transition_tests(machine),
            "state_sequences": self._n_switch_tests(machine, max_length=3),
            "boundary_states": self._boundary_state_tests(machine),
        }

    def _valid_transition_tests(self, machine: StateMachine) -> List[Dict[str, Any]]:
        """One test per valid (from, event, to) triple – 0-switch coverage."""
        tests = []
        for i, t in enumerate(machine.transitions, 1):
            tests.append({
                "id": f"ST-VALID-{i:03d}",
                "type": "state_transition",
                "scenario_type": "functional",
                "title": f"{machine.entity}: {t.from_state} →[{t.event}]→ {t.to_state}",
                "description": f"Verify {machine.entity} transitions from '{t.from_state}' to '{t.to_state}' when '{t.event}' event occurs",
                "preconditions": [f"{machine.entity} is in state: {t.from_state}"],
                "steps": [
                    f"Set {machine.entity} to state '{t.from_state}'",
                    f"Trigger event: '{t.event}'" + (f" (guard: {t.guard})" if t.guard else ""),
                    f"Verify {machine.entity} is now in state '{t.to_state}'",
                ] + ([f"Verify action: {t.action}"] if t.action else []),
                "expected_result": f"State changes from '{t.from_state}' to '{t.to_state}'",
                "risk_level": "medium",
            })
        return tests

    def _invalid_transition_tests(self, machine: StateMachine) -> List[Dict[str, Any]]:
        """Test events that should be REJECTED from each state."""
        # Build map: state → valid events
        valid_events_per_state: Dict[str, Set[str]] = defaultdict(set)
        all_events: Set[str] = set()
        for t in machine.transitions:
            valid_events_per_state[t.from_state].add(t.event)
            all_events.add(t.event)

        tests = []
        i = 1
        for state in machine.states:
            valid_events = valid_events_per_state[state]
            invalid_events = all_events - valid_events
            for event in invalid_events:
                tests.append({
                    "id": f"ST-INVALID-{i:03d}",
                    "type": "state_transition",
                    "scenario_type": "negative",
                    "title": f"{machine.entity}: REJECT '{event}' in state '{state}'",
                    "description": f"Verify system rejects '{event}' when {machine.entity} is in '{state}' state",
                    "preconditions": [f"{machine.entity} is in state: {state}"],
                    "steps": [
                        f"Set {machine.entity} to state '{state}'",
                        f"Attempt to trigger event: '{event}'",
                        f"Verify {machine.entity} remains in state '{state}'",
                        "Verify appropriate error/rejection response",
                    ],
                    "expected_result": f"Event rejected; state remains '{state}'; error returned",
                    "risk_level": "high",
                })
                i += 1
        return tests

    def _n_switch_tests(self, machine: StateMachine, max_length: int = 3) -> List[Dict[str, Any]]:
        """DFS to find paths of length up to max_length through the state machine."""
        # Build adjacency: state → list of transitions
        adj: Dict[str, List[Transition]] = defaultdict(list)
        for t in machine.transitions:
            adj[t.from_state].append(t)

        paths: List[List[Transition]] = []
        self._dfs(adj, machine.initial_state, [], paths, max_length)

        tests = []
        for i, path in enumerate(paths[:20], 1):  # cap at 20 sequences
            if len(path) < 2:
                continue
            state_sequence = [machine.initial_state] + [t.to_state for t in path]
            event_sequence = [t.event for t in path]
            tests.append({
                "id": f"ST-SEQ-{i:03d}",
                "type": "state_sequence",
                "scenario_type": "functional",
                "title": f"{machine.entity}: Sequence [{' → '.join(state_sequence)}]",
                "description": f"End-to-end state sequence test via events: {' → '.join(event_sequence)}",
                "preconditions": [f"{machine.entity} is in initial state: {machine.initial_state}"],
                "steps": [f"Trigger '{t.event}'; expect state '{t.to_state}'" for t in path],
                "expected_result": f"Final state: {state_sequence[-1]}",
                "risk_level": "high" if len(path) >= 3 else "medium",
            })
        return tests

    def _dfs(self, adj: Dict, state: str, path: List[Transition],
             all_paths: List, max_length: int):
        if len(path) >= max_length:
            all_paths.append(list(path))
            return
        if not adj[state]:
            if path:
                all_paths.append(list(path))
            return
        for t in adj[state]:
            # Avoid infinite loops
            visited_states = {p.from_state for p in path}
            if t.to_state not in visited_states:
                path.append(t)
                self._dfs(adj, t.to_state, path, all_paths, max_length)
                path.pop()

    def _boundary_state_tests(self, machine: StateMachine) -> List[Dict[str, Any]]:
        """Test initial state and final states explicitly."""
        tests = []
        tests.append({
            "id": "ST-INIT-001",
            "type": "initial_state",
            "scenario_type": "functional",
            "title": f"{machine.entity}: Verify initial state on creation",
            "preconditions": [],
            "steps": [f"Create new {machine.entity} instance", f"Verify state is '{machine.initial_state}'"],
            "expected_result": f"State is '{machine.initial_state}'",
            "risk_level": "high",
        })
        for i, fs in enumerate(machine.final_states, 1):
            tests.append({
                "id": f"ST-FINAL-{i:03d}",
                "type": "final_state",
                "scenario_type": "functional",
                "title": f"{machine.entity}: Verify no transitions from final state '{fs}'",
                "preconditions": [f"{machine.entity} is in state: {fs}"],
                "steps": [f"Attempt any action on {machine.entity} in '{fs}' state", "Verify rejection or no-op"],
                "expected_result": f"No state change allowed from final state '{fs}'",
                "risk_level": "high",
            })
        return tests

    def infer_from_description(self, description: str) -> Optional[Dict[str, Any]]:
        """
        Heuristic: extract states/transitions from a free-text feature description.
        Returns a machine spec dict or None.
        """
        import re

        # Common state patterns
        state_patterns = [
            r'\b(pending|active|inactive|processing|completed|failed|cancelled|rejected|approved|'
            r'submitted|draft|published|archived|locked|open|closed|resolved|in_progress|paused)\b',
        ]
        transition_words = {
            "submit": ("draft", "pending"),
            "approve": ("pending", "approved"),
            "reject": ("pending", "rejected"),
            "cancel": ("active", "cancelled"),
            "complete": ("processing", "completed"),
            "activate": ("inactive", "active"),
            "deactivate": ("active", "inactive"),
            "publish": ("draft", "published"),
            "archive": ("published", "archived"),
            "reopen": ("closed", "open"),
            "close": ("open", "closed"),
            "resolve": ("open", "resolved"),
            "lock": ("active", "locked"),
            "unlock": ("locked", "active"),
            "process": ("pending", "processing"),
            "fail": ("processing", "failed"),
            "retry": ("failed", "pending"),
        }

        desc_lower = description.lower()
        found_states: Set[str] = set()
        found_transitions: List[Dict] = []

        for pattern in state_patterns:
            matches = re.findall(pattern, desc_lower)
            found_states.update(matches)

        for event, (from_s, to_s) in transition_words.items():
            if event in desc_lower:
                found_states.add(from_s)
                found_states.add(to_s)
                found_transitions.append({"from": from_s, "event": event, "to": to_s})

        if len(found_states) < 2:
            return None

        states = sorted(found_states)
        return {
            "entity": "Entity",
            "states": states,
            "initial_state": states[0],
            "final_states": [s for s in states if s in ("completed", "failed", "cancelled", "archived", "resolved", "closed")],
            "transitions": found_transitions,
        }


state_engine = StateTransitionEngine()
