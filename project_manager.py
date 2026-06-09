"""Project registry.

Uses MongoDB when MONGODB_URI is configured; falls back to a local JSON file.
"""
import json
import uuid
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


class ProjectManager:
    def __init__(self):
        from db.mongo_client import get_db
        db = get_db()
        if db is not None:
            from db.repositories import ProjectRepository
            self._repo = ProjectRepository(db["projects"])
            self._mode = "mongo"
        else:
            self._mode = "file"
            from config import settings
            base = Path(settings.DATA_DIR) if settings.DATA_DIR else Path("./backend/data")
            self._path = base / "projects.json"
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._lock = threading.Lock()
            self._projects: Dict[str, dict] = self._load()

    # ── file helpers ──────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text())
            except Exception:
                return {}
        return {}

    def _save(self):
        self._path.write_text(json.dumps(self._projects, indent=2))

    # ── public API ────────────────────────────────────────────────────────────

    def create(self, name: str, description: str = "") -> dict:
        if self._mode == "mongo":
            return self._repo.create(name, description)
        with self._lock:
            pid = str(uuid.uuid4())[:8]
            project = {
                "id": pid,
                "name": name,
                "description": description,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._projects[pid] = project
            self._save()
            return project

    def list_all(self) -> List[dict]:
        if self._mode == "mongo":
            return self._repo.list_all()
        with self._lock:
            return sorted(self._projects.values(), key=lambda p: p["created_at"], reverse=True)

    def get(self, pid: str) -> Optional[dict]:
        if self._mode == "mongo":
            return self._repo.get(pid)
        with self._lock:
            return self._projects.get(pid)

    def delete(self, pid: str) -> bool:
        if self._mode == "mongo":
            return self._repo.delete(pid)
        with self._lock:
            if pid not in self._projects:
                return False
            del self._projects[pid]
            self._save()
            return True


project_manager = ProjectManager()
