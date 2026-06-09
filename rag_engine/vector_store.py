"""
ChromaDB vector store wrapper.
Handles collection creation, document upsert, and similarity search.
"""

import threading
import structlog
from typing import Any, Dict, List, Optional
from config import settings

log = structlog.get_logger()


class VectorStore:
    def __init__(self, collection_name: str = None):
        self._collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self._client = None
        self._collection = None
        self._embeddings = None
        self._lock = threading.Lock()

    def _init(self):
        if self._collection is not None:
            return
        with self._lock:
            if self._collection is not None:
                return
            import chromadb
            self._client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            log.info("chroma_collection_ready", name=self._collection_name)

    def _get_embeddings(self) -> Any:
        if self._embeddings is None:
            from langchain_openai import OpenAIEmbeddings
            self._embeddings = OpenAIEmbeddings(
                openai_api_key=settings.DEEPINFRA_API_KEY,
                openai_api_base=settings.DEEPINFRA_BASE_URL,
                model=settings.EMBED_MODEL,
                check_embedding_ctx_length=False,
                tiktoken_enabled=False,
            )
        return self._embeddings

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Embed and store document chunks. Returns number stored."""
        self._init()
        if not chunks:
            return 0

        texts = [str(c["text"]) for c in chunks if c.get("text")]
        ids = [f"{c['source_id']}_chunk_{c['chunk_index']}" for c in chunks if c.get("text")]
        metadatas = [c["metadata"] for c in chunks if c.get("text")]

        embeddings_model = self._get_embeddings()
        vectors = embeddings_model.embed_documents(texts)

        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=vectors,
            metadatas=metadatas,
        )
        log.info("chunks_stored", count=len(chunks))
        return len(chunks)

    def search(self, query: str, top_k: int = None, doc_type_filter: str = None) -> List[Dict[str, Any]]:
        """Semantic similarity search. Returns top_k most relevant chunks."""
        self._init()
        top_k = top_k or settings.MAX_RETRIEVAL_DOCS

        where = None
        if doc_type_filter:
            where = {"doc_type": doc_type_filter}

        try:
            embeddings_model = self._get_embeddings()
            query_vector = embeddings_model.embed_query(query)
            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=min(top_k, self._collection.count() or 1),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            log.warning("vector_search_failed_using_text_search", error=str(e))
            return self._text_fallback_search(query, top_k, doc_type_filter)

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "text": doc,
                "metadata": meta,
                "relevance_score": round(1 - dist, 4),
                "source_id": meta.get("source_id", ""),
                "doc_type": meta.get("doc_type", ""),
            })
        return hits

    def _text_fallback_search(self, query: str, top_k: int, doc_type_filter: Optional[str]) -> List[Dict[str, Any]]:
        """Keyword-based fallback when embedding fails."""
        self._init()
        where = {"doc_type": doc_type_filter} if doc_type_filter else None
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(top_k, self._collection.count() or 1),
                where=where,
                include=["documents", "metadatas"],
            )
            return [
                {"text": doc, "metadata": meta, "relevance_score": 0.5,
                 "source_id": meta.get("source_id", ""), "doc_type": meta.get("doc_type", "")}
                for doc, meta in zip(results["documents"][0], results["metadatas"][0])
            ]
        except Exception:
            return []

    def count(self) -> int:
        self._init()
        return self._collection.count()

    def get_all_sources(self) -> List[str]:
        self._init()
        try:
            results = self._collection.get(include=["metadatas"])
            return list({m.get("source_id", "") for m in results["metadatas"]})
        except Exception:
            return []

    def clear(self) -> None:
        self._init()
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        log.info("chroma_collection_cleared")


vector_store = VectorStore()

_store_registry: Dict[str, "VectorStore"] = {}


def get_vector_store(project_id: str = "default") -> "VectorStore":
    key = f"qa_{project_id}"
    if key not in _store_registry:
        _store_registry[key] = VectorStore(collection_name=key)
    return _store_registry[key]
