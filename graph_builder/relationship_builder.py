"""
Ingests extracted entities + relationships into the graph adapter.
Also runs post-processing inference to auto-detect implied relationships.
"""

import re
import uuid
from typing import Any, Dict, List
import structlog

from graph_builder.neo4j_client import get_graph
from graph_builder.graph_schema import NODE_LABELS, REL_TYPES

log = structlog.get_logger()


class RelationshipBuilder:
    def __init__(self):
        self._graph = get_graph()

    def ingest(self, extraction_result: Dict[str, Any], source_id: str) -> Dict[str, int]:
        """
        Persist entities and relationships returned by EntityExtractor.
        Returns counts of what was written.
        """
        entities_written = 0
        rels_written = 0
        entity_id_set = set()

        # ── Write nodes ──────────────────────────────────────────────────────
        for entity in extraction_result.get("entities", []):
            label = entity.get("label", "")
            if label not in NODE_LABELS:
                log.debug("unknown_label_skipped", label=label)
                continue
            eid = entity.get("id", "")
            if not eid:
                continue
            props = {
                "name": entity.get("name", eid),
                "source_id": source_id,
                **entity.get("properties", {}),
            }
            self._graph.upsert_node(label, eid, props)
            entity_id_set.add(eid)
            entities_written += 1

        # ── Write relationships ───────────────────────────────────────────────
        for rel in extraction_result.get("relationships", []):
            from_id = rel.get("from_id", "")
            to_id = rel.get("to_id", "")
            rel_type = rel.get("type", "")
            if not from_id or not to_id or rel_type not in REL_TYPES:
                continue
            # Only write if both nodes exist
            if from_id in entity_id_set or self._graph.get_node(from_id):
                if to_id in entity_id_set or self._graph.get_node(to_id):
                    self._graph.upsert_relationship(from_id, to_id, rel_type, rel.get("properties", {}))
                    rels_written += 1

        # ── Infer additional relationships ────────────────────────────────────
        inferred = self._infer_relationships(extraction_result.get("entities", []))
        rels_written += inferred

        log.info("graph_ingested",
                 source_id=source_id,
                 entities=entities_written,
                 relationships=rels_written)
        return {"entities": entities_written, "relationships": rels_written}

    def _infer_relationships(self, entities: List[Dict[str, Any]]) -> int:
        """
        Auto-detect relationships not explicitly stated:
        - Features with same module_id → BELONGS_TO that module
        - APIs with module_id → BELONGS_TO that module
        """
        count = 0
        for entity in entities:
            label = entity.get("label", "")
            eid = entity.get("id", "")
            props = entity.get("properties", {})

            if label in ("Feature", "API") and props.get("module_id"):
                mid = props["module_id"]
                # Ensure module node exists (create minimal if not)
                if not self._graph.get_node(mid):
                    self._graph.upsert_node("Module", mid, {"name": mid, "criticality": 3})
                try:
                    self._graph.upsert_relationship(eid, mid, "BELONGS_TO", {})
                    count += 1
                except Exception:
                    pass

        return count


relationship_builder = RelationshipBuilder()
