"""MongoDB repository classes.

Three collections:
  projects          — project metadata (replaces projects.json)
  analyses          — per-project analysis history (replaces analyses/{id}.json)
  ingested_documents — metadata for every file ingested into a project's KB
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid


# ── helpers ──────────────────────────────────────────────────────────────────

def _strip(doc: dict) -> dict:
    """Remove MongoDB's internal _id before returning to callers."""
    return {k: v for k, v in doc.items() if k != "_id"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Projects ─────────────────────────────────────────────────────────────────

class ProjectRepository:
    def __init__(self, collection):
        self._col = collection
        self._col.create_index("id", unique=True)

    def create(self, name: str, description: str = "") -> dict:
        project = {
            "id": str(uuid.uuid4())[:8],
            "name": name,
            "description": description,
            "created_at": _now(),
        }
        self._col.insert_one({**project, "_id": project["id"]})
        return project

    def list_all(self) -> List[dict]:
        return [_strip(p) for p in self._col.find().sort("created_at", -1)]

    def get(self, pid: str) -> Optional[dict]:
        doc = self._col.find_one({"id": pid})
        return _strip(doc) if doc else None

    def delete(self, pid: str) -> bool:
        return self._col.delete_one({"id": pid}).deleted_count > 0


# ── Analyses ─────────────────────────────────────────────────────────────────

class AnalysisRepository:
    def __init__(self, collection):
        self._col = collection
        self._col.create_index([("project_id", 1), ("timestamp", -1)])
        self._col.create_index("analysis_id", unique=True)

    def save(self, project_id: str, story: str, result: Dict) -> dict:
        analysis_id = uuid.uuid4().hex[:12]
        doc = {
            "_id": analysis_id,
            "analysis_id": analysis_id,
            "project_id": project_id,
            "story": story,
            "result": result,
            "timestamp": _now(),
        }
        self._col.insert_one(doc)
        return {"id": analysis_id, "story": story, "result": result, "timestamp": doc["timestamp"]}

    def list(self, project_id: str) -> List[dict]:
        return [
            {"id": d["analysis_id"], "story": d["story"], "result": d["result"], "timestamp": d["timestamp"]}
            for d in self._col.find({"project_id": project_id}).sort("timestamp", -1)
        ]

    def delete_one(self, project_id: str, analysis_id: str) -> bool:
        return self._col.delete_one(
            {"project_id": project_id, "analysis_id": analysis_id}
        ).deleted_count > 0

    def delete_all(self, project_id: str) -> None:
        self._col.delete_many({"project_id": project_id})


# ── Ingested Documents ────────────────────────────────────────────────────────

class DocumentRepository:
    """Tracks every file ingested into a project's knowledge base."""

    def __init__(self, collection):
        self._col = collection
        self._col.create_index([("project_id", 1), ("ingested_at", -1)])

    def save(self, project_id: str, filename: str, doc_type: str, ingest_result: dict) -> dict:
        doc_id = uuid.uuid4().hex[:12]
        doc = {
            "_id": doc_id,
            "document_id": doc_id,
            "project_id": project_id,
            "filename": filename,
            "doc_type": ingest_result.get("detected_type", doc_type),
            "status": ingest_result.get("status", "UNKNOWN"),
            "chunks": ingest_result.get("chunks", 0),
            "entities": ingest_result.get("entities", 0),
            "relationships": ingest_result.get("rels", 0),
            "ingested_at": _now(),
        }
        self._col.insert_one(doc)
        return _strip(doc)

    def list(self, project_id: str) -> List[dict]:
        return [_strip(d) for d in self._col.find({"project_id": project_id}).sort("ingested_at", -1)]

    def delete_all(self, project_id: str) -> None:
        self._col.delete_many({"project_id": project_id})
