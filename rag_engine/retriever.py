"""
RAG Retriever – combines vector search results into a structured context block
for the LLM prompt.
"""

from typing import Any, Dict, List
import structlog

from rag_engine.vector_store import vector_store
from config import settings

log = structlog.get_logger()


class RAGRetriever:
    def retrieve(self, query: str, top_k: int = None, store=None) -> Dict[str, Any]:
        """
        Retrieve relevant chunks for a query and organise by document type.
        Returns a structured context dict.
        """
        _store = store or vector_store
        top_k = top_k or settings.MAX_RETRIEVAL_DOCS
        hits = _store.search(query, top_k=top_k)

        context: Dict[str, List[str]] = {
            "brd": [],
            "srs": [],
            "user_story": [],
            "test_case": [],
            "bug_report": [],
            "api_contract": [],
            "db_schema": [],
            "event_definition": [],
            "business_rule": [],
            "other": [],
        }

        for hit in hits:
            doc_type = hit.get("doc_type", "other")
            bucket = doc_type if doc_type in context else "other"
            context[bucket].append(hit["text"])

        # Deduplicate within buckets
        context = {k: list(dict.fromkeys(v)) for k, v in context.items()}

        total_retrieved = sum(len(v) for v in context.values())
        log.info("rag_retrieval_complete", query_preview=query[:80], total_chunks=total_retrieved)

        return {
            "query": query,
            "total_retrieved": total_retrieved,
            "context_by_type": context,
            "raw_hits": hits,
        }

    def retrieve_bugs(self, query: str, top_k: int = 15, store=None) -> List[str]:
        _store = store or vector_store
        hits = _store.search(query, top_k=top_k, doc_type_filter="bug_report")
        return [h["text"] for h in hits]

    def retrieve_test_cases(self, query: str, top_k: int = 10, store=None) -> List[str]:
        _store = store or vector_store
        hits = _store.search(query, top_k=top_k, doc_type_filter="test_case")
        return [h["text"] for h in hits]

    def retrieve_api_contracts(self, query: str, top_k: int = 10, store=None) -> List[str]:
        _store = store or vector_store
        hits = _store.search(query, top_k=top_k, doc_type_filter="api_contract")
        return [h["text"] for h in hits]

    def build_context_string(self, retrieval_result: Dict[str, Any], max_chars: int = 8000) -> str:
        """
        Flatten retrieved context into a clearly structured string for the LLM.
        Spec documents are placed first so the LLM reads the most actionable
        content before any truncation kicks in.
        """
        _LABELS = {
            "user_story":       "USER STORIES / ACCEPTANCE CRITERIA",
            "brd":              "BUSINESS REQUIREMENTS DOCUMENT (BRD)",
            "srs":              "SOFTWARE REQUIREMENTS SPECIFICATION (SRS)",
            "business_rule":    "BUSINESS RULES",
            "api_contract":     "API CONTRACTS / ENDPOINTS",
            "db_schema":        "DATABASE SCHEMA / FIELD DEFINITIONS",
            "event_definition": "EVENTS / MESSAGE FORMATS",
            "test_case":        "EXISTING TEST CASES",
            "bug_report":       "HISTORICAL BUG REPORTS",
            "other":            "OTHER DOCUMENTS",
        }
        # Spec docs first — they contain field names, rules, and acceptance criteria
        priority = [
            "user_story", "brd", "srs", "business_rule",
            "api_contract", "db_schema", "event_definition",
            "test_case", "bug_report", "other",
        ]
        ctx = retrieval_result.get("context_by_type", {})
        lines: List[str] = []

        for doc_type in priority:
            chunks = ctx.get(doc_type, [])
            if not chunks:
                continue
            label = _LABELS.get(doc_type, doc_type.upper())
            lines.append(f"\n{'─'*60}")
            lines.append(f"[{label}]")
            lines.append(f"{'─'*60}")
            for i, chunk in enumerate(chunks, 1):
                lines.append(f"\n--- Excerpt {i} ---")
                lines.append(chunk.strip())

        full = "\n".join(lines)
        if not full.strip():
            return ""
        if len(full) > max_chars:
            full = full[:max_chars] + (
                "\n\n[Note: additional content was truncated — "
                "use the excerpts shown above to ground your response.]"
            )
        return full


retriever = RAGRetriever()
