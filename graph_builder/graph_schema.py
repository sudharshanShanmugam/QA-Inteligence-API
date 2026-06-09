"""
Knowledge Graph schema definitions.

Node labels:   Module | Feature | API | DBTable | Event | TestCase | Bug | UserJourney | BusinessRule | State
Relationship:  DEPENDS_ON | BELONGS_TO | TRIGGERS | UPDATES | RELATED_TO | FOUND_IN | VALIDATES |
               SPANS | PRODUCES | CONSUMES | TRANSITIONS_TO | GOVERNED_BY
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class NodeSchema:
    label: str
    required_props: List[str]
    optional_props: List[str] = field(default_factory=list)


@dataclass
class RelSchema:
    type: str
    from_label: str
    to_label: str
    properties: List[str] = field(default_factory=list)


NODES = [
    NodeSchema("Module",       ["id", "name"],        ["criticality", "description", "owner", "tech_stack"]),
    NodeSchema("Feature",      ["id", "name"],        ["module_id", "description", "priority", "status"]),
    NodeSchema("API",          ["id", "endpoint", "method"], ["module_id", "description", "auth_required"]),
    NodeSchema("DBTable",      ["id", "name"],        ["database", "schema_def"]),
    NodeSchema("Event",        ["id", "name"],        ["topic", "type", "producer", "consumer", "payload_schema"]),
    NodeSchema("TestCase",     ["id", "name"],        ["type", "status", "feature_id", "priority", "automated"]),
    NodeSchema("Bug",          ["id", "title"],       ["severity", "status", "module_id", "feature_id", "root_cause", "description"]),
    NodeSchema("UserJourney",  ["id", "name"],        ["steps", "persona"]),
    NodeSchema("BusinessRule", ["id", "name"],        ["condition", "action", "priority"]),
    NodeSchema("State",        ["id", "name"],        ["entity", "transitions"]),
]

RELATIONSHIPS = [
    RelSchema("DEPENDS_ON",     "Module",      "Module",      ["dependency_type", "strength"]),
    RelSchema("BELONGS_TO",     "Feature",     "Module",      []),
    RelSchema("BELONGS_TO",     "API",         "Module",      []),
    RelSchema("TRIGGERS",       "Feature",     "Event",       ["condition"]),
    RelSchema("UPDATES",        "API",         "DBTable",     ["operation"]),
    RelSchema("RELATED_TO",     "Bug",         "Feature",     ["relationship_type"]),
    RelSchema("FOUND_IN",       "Bug",         "Module",      ["count"]),
    RelSchema("VALIDATES",      "TestCase",    "Feature",     ["coverage_type"]),
    RelSchema("SPANS",          "UserJourney", "Module",      ["step_order"]),
    RelSchema("PRODUCES",       "Event",       "Event",       ["transformation"]),
    RelSchema("CONSUMES",       "Module",      "Event",       []),
    RelSchema("TRANSITIONS_TO", "State",       "State",       ["trigger", "guard"]),
    RelSchema("GOVERNED_BY",    "Feature",     "BusinessRule", []),
]

NODE_LABELS = {n.label for n in NODES}
REL_TYPES = {r.type for r in RELATIONSHIPS}
