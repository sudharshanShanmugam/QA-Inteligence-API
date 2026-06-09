"""MongoDB connection singleton.

Usage:
    from db.mongo_client import get_db
    db = get_db()          # None if MONGODB_URI is not configured
    col = db["projects"]
"""
from typing import Optional
import structlog

log = structlog.get_logger()

_client = None


def get_db():
    """Return the MongoDB Database, or None if MONGODB_URI is not set / unreachable."""
    global _client
    from config import settings

    if not settings.MONGODB_URI:
        return None

    if _client is None:
        try:
            from pymongo import MongoClient
            _client = MongoClient(
                settings.MONGODB_URI,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
            )
            # Cheap ping to validate connectivity at startup
            _client.admin.command("ping")
            log.info("mongodb_connected", db=settings.MONGODB_DB_NAME)
        except Exception as e:
            log.warning("mongodb_unavailable", error=str(e))
            _client = None
            return None

    return _client[settings.MONGODB_DB_NAME]
