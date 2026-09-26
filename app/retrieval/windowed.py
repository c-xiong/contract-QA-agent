"""Token-window vectors with max pooling back to unchanged source chunks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import faiss

from app.ingestion.store import ChunkStore
from app.retrieval.base import apply_filters
from app.retrieval.dense import INDEX_META, DenseIndexError, DenseRetriever
from app.retrieval.embeddings import DEFAULT_MODEL, get_embedder
from app.schemas.retrieval import RetrievalFilters, RetrievedChunk

WINDOW_INDEX_VERSION = "token-windows-v1"


def corpus_fingerprint(store: ChunkStore) -> str:
    digest = hashlib.sha256()
    for chunk in store.chunks:
        digest.update(chunk.model_dump_json().encode())
        digest.update(b"\n")
    return digest.hexdigest()


class WindowedDenseRetriever(DenseRetriever):
    """Exact window search, deduplicated before the top-k original-chunk limit."""

    name = "windowed_dense"

    @classmethod
    def build(
        cls, store: ChunkStore, *, model_id: str = DEFAULT_MODEL, batch_size: int = 64
    ) -> WindowedDenseRetriever:
        embedder = get_embedder(model_id)
        if not store.chunks:
            raise DenseIndexError("Cannot build a dense index over an empty store")
        texts: list[str] = []
        ids: list[str] = []
        for chunk in store.chunks:
            heading = chunk.section_title or (
                f"Section {chunk.section_id}" if chunk.section_id else ""
            )
            windows = embedder.window_texts(chunk.text, heading=heading)
            texts.extend(windows)
            ids.extend([chunk.chunk_id] * len(windows))
        vectors = embedder.encode(texts, batch_size=batch_size)
        index = faiss.IndexFlatIP(embedder.dimension)
        index.add(vectors)
        return cls(store, index, ids, embedder)

    def save(self, index_dir: Path) -> None:
        super().save(index_dir)
        path = index_dir / INDEX_META
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta.update(version=WINDOW_INDEX_VERSION, corpus_sha256=corpus_fingerprint(self._store))
        path.write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, store: ChunkStore, index_dir: Path) -> WindowedDenseRetriever:
        path = index_dir / INDEX_META
        if not path.exists():
            raise DenseIndexError("Missing window index; run scripts/build_index.py --windows")
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta.get("version") != WINDOW_INDEX_VERSION or meta.get(
            "corpus_sha256"
        ) != corpus_fingerprint(store):
            raise DenseIndexError("Stale window index; run scripts/build_index.py --windows")
        return cast(WindowedDenseRetriever, super().load(store, index_dir))

    async def search(
        self, query: str, top_k: int, filters: RetrievalFilters | None = None
    ) -> list[RetrievedChunk]:
        if not query.strip() or top_k <= 0:
            return []
        vector = self._embedder.encode_one(query).reshape(1, -1)
        # DECISION: exhaustive scores ensure scope filtering and window deduplication
        # cannot lose eligible candidates to a global over-fetch cutoff. The local
        # corpus is small; approximate search is unnecessary here.
        scores, positions = self._index.search(vector, len(self._chunk_ids))
        best: dict[str, float] = {}
        for score, position in zip(scores[0], positions[0], strict=True):
            if position < 0:
                continue
            cid = self._chunk_ids[int(position)]
            if cid in best:
                continue
            chunk = self._store.get_chunk(cid)
            if chunk is None:
                continue
            document = self._store.get_document(chunk.document_id)
            if apply_filters(chunk.document_id, bool(document and document.is_synthetic), filters):
                best[cid] = float(score)
        ordered = sorted(best, key=lambda cid: (-best[cid], cid))[:top_k]
        results: list[RetrievedChunk] = []
        for rank, cid in enumerate(ordered, start=1):
            chunk = self._store.get_chunk(cid)
            assert chunk is not None
            results.append(
                RetrievedChunk(
                    chunk=chunk, dense_rank=rank, fused_score=best[cid], retrieval_query=query
                )
            )
        return results
