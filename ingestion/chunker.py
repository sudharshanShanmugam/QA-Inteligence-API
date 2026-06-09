"""
Adaptive chunking pipeline.

The LLM analyzes the full document and recommends one of 9 strategies.
Falls back to rule-based auto-detection if the LLM is unavailable.

Strategies
----------
recursive_meta   – Recursive splitting + rich metadata on every chunk (recommended flat)
auto             – Chunker inspects the doc itself and picks the best strategy
recursive        – Smart recursive split using multiple separator tiers
markdown_header  – Split strictly at markdown # / ## / ### / #### boundaries
structure        – Field-aware split for typed JSON docs (user stories, bugs, APIs …)
sentence         – Sentence-level split for dense technical content
paragraph        – Paragraph split for narrative prose
fixed            – Fixed character-count chunks with overlap
token            – Approximate token-count chunks with overlap (1 token ≈ 4 chars)
"""

import json
import re
from typing import Any, Dict, List, Tuple

# ── Per-strategy size config ─────────────────────────────────────────────────
_CFG: Dict[str, Dict[str, int]] = {
    "recursive_meta": {"chunk": 800,  "overlap": 80,  "min": 120},
    "auto":           {"chunk": 800,  "overlap": 80,  "min": 120},
    "recursive":      {"chunk": 1000, "overlap": 100, "min": 150},
    "markdown_header":{"chunk": 1200, "overlap": 0,   "min": 100},
    "structure":      {"chunk": 1000, "overlap": 0,   "min": 150},
    "sentence":       {"chunk": 400,  "overlap": 0,   "min":  60},
    "paragraph":      {"chunk": 800,  "overlap": 0,   "min": 100},
    "fixed":          {"chunk": 500,  "overlap": 50,  "min":   0},
    "token":          {"chunk": 1024, "overlap": 128, "min":   0},  # chars ≈ tokens*4
}
_DEFAULT_CFG = {"chunk": 800, "overlap": 80, "min": 120}

VALID_STRATEGIES = set(_CFG.keys())


class DocumentChunker:

    def chunk(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        strategy = self._get_strategy(document)
        cfg = _CFG.get(strategy, _DEFAULT_CFG)

        # resolve "auto" to a concrete strategy via heuristics
        if strategy == "auto":
            strategy = self._auto_detect(document)
            cfg = _CFG.get(strategy, _DEFAULT_CFG)

        raw_chunks = self._dispatch(strategy, document, cfg)
        refined = self._refine(raw_chunks, min_chars=cfg["min"])
        return self._finalize(refined, document, strategy)

    # ── Strategy selection ────────────────────────────────────────────────────

    @staticmethod
    def _get_strategy(document: Dict[str, Any]) -> str:
        """Ask the LLM which of the 9 strategies fits this document best."""
        try:
            from test_generator.llm_client import llm_client
            return llm_client.recommend_chunking_strategy(document)
        except Exception:
            return "auto"

    @staticmethod
    def _auto_detect(document: Dict[str, Any]) -> str:
        """Heuristic fallback — inspect the document and pick a strategy."""
        fmt = document.get("format", "text")
        doc_type = document.get("doc_type", "")
        content = document.get("content", "")

        if fmt == "json" or doc_type in {
            "user_story", "bug_report", "api_contract", "db_schema", "event_definition"
        }:
            return "structure"

        md_headers = len(re.findall(r'^#{1,4}\s+', content, re.MULTILINE))
        if md_headers >= 3:
            return "markdown_header"

        numbered = len(re.findall(r'^\d+\.\s+[A-Z]', content, re.MULTILINE))
        if numbered >= 3:
            return "recursive_meta"

        avg_para = sum(len(p) for p in content.split("\n\n") if p.strip()) / max(
            len([p for p in content.split("\n\n") if p.strip()]), 1
        )
        if avg_para > 600:
            return "recursive"

        return "recursive_meta"

    # ── Dispatcher ────────────────────────────────────────────────────────────

    def _dispatch(
        self, strategy: str, document: Dict[str, Any], cfg: Dict[str, int]
    ) -> List[Dict[str, str]]:
        if strategy == "recursive_meta":
            return self._recursive_meta_split(document, cfg)
        if strategy == "recursive":
            return self._recursive_split(document, cfg)
        if strategy == "markdown_header":
            return self._markdown_header_split(document)
        if strategy == "structure":
            return self._structure_split(document)
        if strategy == "sentence":
            return self._sentence_split(document, cfg)
        if strategy == "paragraph":
            return self._paragraph_split(document, cfg)
        if strategy == "fixed":
            return self._fixed_split(document, cfg)
        if strategy == "token":
            return self._token_split(document, cfg)
        # unknown → recursive_meta
        return self._recursive_meta_split(document, cfg)

    # ── 1. recursive_meta ─────────────────────────────────────────────────────

    def _recursive_meta_split(
        self, document: Dict[str, Any], cfg: Dict[str, int]
    ) -> List[Dict[str, str]]:
        """Recursive splitting that tracks the nearest section header as metadata."""
        content = document.get("content", "")
        chunk_size = cfg["chunk"]
        overlap = cfg["overlap"]
        separators = ["\n\n", "\n", ". ", " ", ""]

        raw_texts = self._recursive_split_text(content, chunk_size, overlap, separators)
        chunks: List[Dict[str, str]] = []

        # track which markdown/numbered header each chunk falls under
        header_pattern = re.compile(
            r'^(#{1,4}\s+.+|[0-9]+(?:\.[0-9]+)*\.?\s+[A-Z].*)$', re.MULTILINE
        )
        header_positions = [(m.start(), m.group().strip()) for m in header_pattern.finditer(content)]

        for i, text in enumerate(raw_texts):
            char_pos = content.find(text[:40]) if len(text) >= 40 else content.find(text)
            section = ""
            for pos, title in header_positions:
                if pos <= char_pos:
                    section = title
                else:
                    break
            chunks.append({
                "text": text,
                "section": section or f"chunk_{i+1}",
                "header_context": section,
            })

        return chunks

    # ── 2. recursive ─────────────────────────────────────────────────────────

    def _recursive_split(
        self, document: Dict[str, Any], cfg: Dict[str, int]
    ) -> List[Dict[str, str]]:
        content = document.get("content", "")
        separators = ["\n\n", "\n", ". ", " ", ""]
        texts = self._recursive_split_text(content, cfg["chunk"], cfg["overlap"], separators)
        return [{"text": t, "section": f"chunk_{i+1}"} for i, t in enumerate(texts)]

    # ── 3. markdown_header ────────────────────────────────────────────────────

    def _markdown_header_split(self, document: Dict[str, Any]) -> List[Dict[str, str]]:
        content = document.get("content", "")
        pattern = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)
        positions = [(m.start(), m.group(1), m.group(2).strip()) for m in pattern.finditer(content)]

        if not positions:
            return [{"text": content.strip(), "section": "body"}]

        chunks: List[Dict[str, str]] = []
        pre = content[:positions[0][0]].strip()
        if pre:
            chunks.append({"text": pre, "section": "preamble"})

        for i, (pos, hashes, title) in enumerate(positions):
            end = positions[i + 1][0] if i + 1 < len(positions) else len(content)
            body = content[pos:end].strip()
            level = len(hashes)
            chunks.append({
                "text": body,
                "section": title,
                "header_level": str(level),
                "no_merge": True,
            })

        return chunks

    # ── 4. structure ──────────────────────────────────────────────────────────

    def _structure_split(self, document: Dict[str, Any]) -> List[Dict[str, str]]:
        """Field-aware split for typed JSON; falls back to section split for text."""
        fmt = document.get("format", "text")
        doc_type = document.get("doc_type", "")
        raw = document.get("raw", {})
        chunks: List[Dict[str, str]] = []

        if fmt != "json" or not raw:
            return self._section_split(document)

        if doc_type == "user_story":
            story = (
                f"USER STORY: {raw.get('title', '')}\n"
                f"As a {raw.get('role', 'user')}, "
                f"I want {raw.get('goal', '')}, "
                f"so that {raw.get('benefit', '')}."
            )
            chunks.append({"text": story, "section": "story_body", "no_merge": True})
            acs = raw.get("acceptance_criteria", [])
            if acs:
                chunks.append({
                    "text": "ACCEPTANCE CRITERIA:\n" + "\n".join(f"- {ac}" for ac in acs),
                    "section": "acceptance_criteria", "no_merge": True,
                })
            brs = raw.get("business_rules", [])
            if brs:
                chunks.append({
                    "text": "BUSINESS RULES:\n" + "\n".join(f"- {br}" for br in brs),
                    "section": "business_rules", "no_merge": True,
                })

        elif doc_type == "bug_report":
            text = (
                f"BUG [{raw.get('severity', '?')}]: {raw.get('title', '')}\n"
                f"Module: {raw.get('module', '')} | Feature: {raw.get('feature', '')}\n"
                f"Description: {raw.get('description', '')}\n"
                f"Root Cause: {raw.get('root_cause', '')}\n"
                f"Steps: {raw.get('steps_to_reproduce', '')}\n"
                f"Status: {raw.get('status', '')}"
            )
            chunks.append({"text": text, "section": "bug_report"})

        elif doc_type == "api_contract":
            header = f"API: {raw.get('name', '')} v{raw.get('version', '1.0')}"
            for ep in raw.get("endpoints", []):
                method = ep.get("method", "GET")
                path = ep.get("path", "")
                text = (
                    f"{header}\nENDPOINT: {method} {path}\n"
                    f"Description: {ep.get('description', '')}\n"
                )
                if ep.get("request_body"):
                    text += f"Request: {json.dumps(ep['request_body'])}\n"
                if ep.get("responses"):
                    text += f"Responses: {list(ep['responses'].keys())}\n"
                if ep.get("events_published"):
                    text += f"Publishes: {ep['events_published']}"
                slug = re.sub(r"\W+", "_", f"{method}_{path}").strip("_")
                chunks.append({"text": text.strip(), "section": f"endpoint_{slug}", "no_merge": True})

        elif doc_type == "db_schema":
            db_name = raw.get("database", "")
            for table in raw.get("tables", []):
                lines = [f"TABLE: {table.get('name', '')} (DB: {db_name})"]
                for col in table.get("columns", []):
                    pk = " PK" if col.get("primary_key") else ""
                    nn = " NOT NULL" if col.get("not_null") else ""
                    lines.append(f"  {col.get('name', '')} {col.get('type', '')}{pk}{nn}")
                chunks.append({
                    "text": "\n".join(lines),
                    "section": f"table_{table.get('name', '').lower()}",
                    "no_merge": True,
                })

        elif doc_type == "event_definition":
            text = (
                f"EVENT: {raw.get('name', '')}\n"
                f"Topic: {raw.get('topic', '')} | Type: {raw.get('type', '')}\n"
                f"Producer: {raw.get('producer', '')} → Consumer: {raw.get('consumer', '')}\n"
                f"Payload: {json.dumps(raw.get('payload_schema', {}))}"
            )
            chunks.append({"text": text, "section": "event_definition"})

        else:
            return self._section_split(document)

        return chunks

    def _section_split(self, document: Dict[str, Any]) -> List[Dict[str, str]]:
        content = document.get("content", "")
        max_chars = _CFG["structure"]["chunk"]
        sections = self._split_by_headers(content)
        chunks: List[Dict[str, str]] = []
        for section_title, body in sections:
            text = f"{section_title}\n{body}".strip() if section_title else body.strip()
            if not text:
                continue
            if len(text) > max_chars:
                for i, block in enumerate(self._split_paragraphs(text, max_chars)):
                    label = f"{section_title}_p{i+1}" if section_title else f"para_{i+1}"
                    chunks.append({"text": block, "section": label})
            else:
                chunks.append({"text": text, "section": section_title or "body"})
        return chunks

    # ── 5. sentence ───────────────────────────────────────────────────────────

    def _sentence_split(
        self, document: Dict[str, Any], cfg: Dict[str, int]
    ) -> List[Dict[str, str]]:
        content = document.get("content", "")
        max_chars = cfg["chunk"]
        sentences = re.split(r"(?<=[.!?])\s+", content)
        chunks: List[Dict[str, str]] = []
        current = ""
        idx = 1
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if current and len(current) + len(sent) + 1 > max_chars:
                chunks.append({"text": current, "section": f"sent_{idx}"})
                idx += 1
                current = sent
            else:
                current = f"{current} {sent}".strip() if current else sent
        if current:
            chunks.append({"text": current, "section": f"sent_{idx}"})
        return chunks

    # ── 6. paragraph ─────────────────────────────────────────────────────────

    def _paragraph_split(
        self, document: Dict[str, Any], cfg: Dict[str, int]
    ) -> List[Dict[str, str]]:
        content = document.get("content", "")
        max_chars = cfg["chunk"]
        paras = [p.strip() for p in re.split(r"\n\n+", content) if p.strip()]
        chunks: List[Dict[str, str]] = []
        for i, para in enumerate(paras):
            if len(para) > max_chars:
                for j, sub in enumerate(self._split_paragraphs(para, max_chars)):
                    chunks.append({"text": sub, "section": f"para_{i+1}_{j+1}"})
            else:
                chunks.append({"text": para, "section": f"para_{i+1}"})
        return chunks

    # ── 7. fixed ─────────────────────────────────────────────────────────────

    def _fixed_split(
        self, document: Dict[str, Any], cfg: Dict[str, int]
    ) -> List[Dict[str, str]]:
        content = document.get("content", "")
        size = cfg["chunk"]
        overlap = cfg["overlap"]
        chunks: List[Dict[str, str]] = []
        start = 0
        idx = 1
        while start < len(content):
            end = start + size
            text = content[start:end].strip()
            if text:
                chunks.append({"text": text, "section": f"chunk_{idx}"})
                idx += 1
            start = end - overlap if overlap else end
        return chunks

    # ── 8. token ─────────────────────────────────────────────────────────────

    def _token_split(
        self, document: Dict[str, Any], cfg: Dict[str, int]
    ) -> List[Dict[str, str]]:
        """Approximate token-based split (1 token ≈ 4 chars)."""
        content = document.get("content", "")
        char_limit = cfg["chunk"]   # already in char units (~tokens*4 from _CFG)
        overlap = cfg["overlap"]

        # split on whitespace boundaries to avoid cutting mid-word
        words = content.split()
        chunks: List[Dict[str, str]] = []
        current_words: List[str] = []
        current_len = 0
        idx = 1

        for word in words:
            word_len = len(word) + 1  # +1 for space
            if current_len + word_len > char_limit and current_words:
                text = " ".join(current_words).strip()
                chunks.append({"text": text, "section": f"token_chunk_{idx}"})
                idx += 1
                # keep overlap words
                overlap_chars = 0
                overlap_words: List[str] = []
                for w in reversed(current_words):
                    overlap_chars += len(w) + 1
                    if overlap_chars > overlap:
                        break
                    overlap_words.insert(0, w)
                current_words = overlap_words + [word]
                current_len = sum(len(w) + 1 for w in current_words)
            else:
                current_words.append(word)
                current_len += word_len

        if current_words:
            chunks.append({"text": " ".join(current_words).strip(), "section": f"token_chunk_{idx}"})

        return chunks

    # ── Shared text utilities ─────────────────────────────────────────────────

    def _recursive_split_text(
        self, text: str, chunk_size: int, overlap: int, separators: List[str]
    ) -> List[str]:
        """Recursively split text trying each separator in order."""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        for sep in separators:
            if sep == "" or sep in text:
                parts = text.split(sep) if sep else list(text)
                chunks: List[str] = []
                current = ""

                for part in parts:
                    candidate = current + (sep if current else "") + part
                    if len(candidate) <= chunk_size:
                        current = candidate
                    else:
                        if current.strip():
                            chunks.append(current.strip())
                        # if this single part is still too large, recurse
                        remaining_seps = separators[separators.index(sep) + 1:]
                        if len(part) > chunk_size and remaining_seps:
                            chunks.extend(
                                self._recursive_split_text(part, chunk_size, overlap, remaining_seps)
                            )
                            current = ""
                        else:
                            current = part

                if current.strip():
                    chunks.append(current.strip())

                # apply overlap: carry last `overlap` chars into next chunk
                if overlap and len(chunks) > 1:
                    overlapped: List[str] = [chunks[0]]
                    for i in range(1, len(chunks)):
                        tail = chunks[i - 1][-overlap:] if overlap < len(chunks[i - 1]) else chunks[i - 1]
                        overlapped.append(tail + "\n" + chunks[i])
                    return overlapped

                return chunks

        return [text]

    def _split_by_headers(self, text: str) -> List[Tuple[str, str]]:
        pattern = re.compile(
            r'^(#{1,4}\s+.+|[0-9]+(?:\.[0-9]+)*\.?\s+[A-Z].*)$', re.MULTILINE
        )
        positions = [(m.start(), m.group().strip()) for m in pattern.finditer(text)]
        if not positions:
            return [("", text)]

        sections: List[Tuple[str, str]] = []
        pre = text[:positions[0][0]].strip()
        if pre:
            sections.append(("", pre))

        for i, (pos, title) in enumerate(positions):
            end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
            body = text[pos + len(title):end].strip()
            sections.append((title, body))

        return sections

    def _split_paragraphs(self, text: str, max_chars: int) -> List[str]:
        paras = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
        result: List[str] = []
        for para in paras:
            if len(para) <= max_chars:
                result.append(para)
                continue
            current = ""
            for sent in re.split(r"(?<=[.!?])\s+", para):
                if len(current) + len(sent) > max_chars and current:
                    result.append(current.strip())
                    current = sent
                else:
                    current += (" " if current else "") + sent
            if current:
                result.append(current.strip())
        return result

    # ── Refinement ────────────────────────────────────────────────────────────

    def _refine(
        self, raw_chunks: List[Dict[str, str]], min_chars: int = 120
    ) -> List[Dict[str, str]]:
        if not raw_chunks:
            return []

        refined: List[Dict[str, str]] = []
        buf: Dict[str, str] = {}

        for chunk in raw_chunks:
            text = chunk.get("text", "").strip()
            if not text:
                continue
            no_merge = chunk.get("no_merge", False)

            if no_merge:
                if buf:
                    refined.append(buf)
                    buf = {}
                refined.append(chunk)
            elif buf and len(buf.get("text", "")) < min_chars:
                buf["text"] += "\n\n" + text
            else:
                if buf:
                    refined.append(buf)
                buf = {k: v for k, v in chunk.items()}

        if buf:
            refined.append(buf)

        return refined

    # ── Finalization: metadata + overlap ──────────────────────────────────────

    def _finalize(
        self,
        chunks: List[Dict[str, str]],
        document: Dict[str, Any],
        strategy: str,
    ) -> List[Dict[str, Any]]:
        source_id = document.get("source_id", "")
        doc_type = document.get("doc_type", "")
        base_meta = document.get("metadata", {})
        total = len(chunks)
        result: List[Dict[str, Any]] = []

        for i, chunk in enumerate(chunks):
            text = chunk["text"]

            # sentence-level overlap — prepend last sentence of previous chunk
            if i > 0:
                overlap = self._last_sentence(chunks[i - 1]["text"])
                if overlap and not text.startswith(overlap[:30]):
                    text = f"[context: {overlap}]\n{text}"

            # scalar metadata only (ChromaDB requirement)
            extra: Dict[str, Any] = {
                k: v for k, v in chunk.items()
                if k not in ("text", "no_merge") and isinstance(v, (str, int, float, bool))
            }

            result.append({
                "text": text,
                "chunk_index": i,
                "total_chunks": total,
                "source_id": source_id,
                "doc_type": doc_type,
                "metadata": {
                    **{k: v for k, v in base_meta.items() if isinstance(v, (str, int, float, bool))},
                    "source_id": source_id,
                    "doc_type": doc_type,
                    "section": chunk.get("section", ""),
                    "chunk_index": i,
                    "total_chunks": total,
                    "chunking_strategy": strategy,
                    **extra,
                },
            })

        return result

    @staticmethod
    def _last_sentence(text: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        for sent in reversed(sentences):
            sent = sent.strip()
            if len(sent) > 20:
                return sent[-120:]
        return ""


chunker = DocumentChunker()
