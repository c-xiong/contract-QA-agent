"""Dense retrieval over a FAISS index.

The semantic arm of Experiment A. Strong where BM25 is weak: a question phrased in
different words from the clause ("can we walk away early?" vs "Termination for
Convenience"), which SPEC 6.8 calls the query-refinement category.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss

from app.ingestion.store import ChunkStore
from app.retrieval.base import apply_filters
from app.retrieval.embeddings import DEFAULT_MODEL, Embedder, get_embedder
from app.schemas.chunk import Chunk
from app.schemas.retrieval import RetrievalFilters, RetrievedChunk

INDEX_FILE = "dense.faiss"
INDEX_META = "dense_meta.json"


class DenseIndexError(RuntimeError):
    """The index is missing, or does not match the store it is being used with."""


class DenseRetriever:
    """FAISS inner-product search over normalized chunk embeddings."""

    name = "dense"

    def __init__(
        self,
        store: ChunkStore,
        index: faiss.Index,
        chunk_ids: list[str],
        embedder: Embedder,
    ) -> None:
        if index.ntotal != len(chunk_ids):
            raise DenseIndexError(
                f"Index holds {index.ntotal} vectors but {len(chunk_ids)} chunk ids were "
                "supplied. Rebuild with scripts/build_index.py."
            )
        self._store = store
        self._index = index
        self._chunk_ids = chunk_ids
        self._embedder = embedder

        missing = [cid for cid in chunk_ids if store.get_chunk(cid) is None]
        if missing:
            raise DenseIndexError(
                f"Index references {len(missing)} chunks absent from the store "
                f"(e.g. {missing[0]}). The index is stale; rebuild it."
            )

    # --- Construction ---------------------------------------------------------

    @staticmethod
    def index_text(chunk: Chunk) -> str:
        """What gets embedded.

        DECISION: prepend the section title and document title to the chunk body.
          A chunk reading only "shall not exceed the fees paid in the preceding twelve
          months" is semantically ambiguous in isolation; prefixed with "Limitation of
          Liability" it is not. The heading is already inside `chunk.text` when the
          chunker found one, but stating it first puts it inside the model's 256-token
          window even when the body is long enough to be truncated.
          Rejected: embedding chunk.text alone. Simpler, and measurably worse on exactly
          the chunks whose heading fell on the previous page.
        """
        parts: list[str] = []
        if chunk.section_title:
            parts.append(chunk.section_title)
        elif chunk.section_id:
            parts.append(f"Section {chunk.section_id}")
        parts.append(chunk.text)
        return "\n".join(parts)

    @classmethod
    def build(
        cls,
        store: ChunkStore,
        *,
        model_id: str = DEFAULT_MODEL,
        batch_size: int = 64,
    ) -> DenseRetriever:
        """Embed every chunk and build a flat index.

        DECISION: IndexFlatIP -- exhaustive search, no approximation.
          The corpus is a few thousand chunks. An exact scan is sub-millisecond, and an
          approximate index (IVF, HNSW) would add tuning parameters that change recall
          for reasons unrelated to the retrieval question being studied. Experiment A
          compares BM25 against dense against fusion; an ANN recall cliff in the dense
          arm would be indistinguishable from the dense model being weak.
          Revisit only if the corpus grows by two orders of magnitude, which SPEC 6.7
          explicitly says it should not.
        """
        embedder = get_embedder(model_id)
        chunks = store.chunks
        if not chunks:
            raise DenseIndexError("Cannot build a dense index over an empty store")

        vectors = embedder.encode([cls.index_text(c) for c in chunks], batch_size=batch_size)
        index = faiss.IndexFlatIP(embedder.dimension)
        index.add(vectors)
        return cls(store, index, [c.chunk_id for c in chunks], embedder)

    # --- Persistence ----------------------------------------------------------

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_dir / INDEX_FILE))
        meta = {
            "model_id": self._embedder.model_id,
            "dimension": self._embedder.dimension,
            "chunk_ids": self._chunk_ids,
        }
        (index_dir / INDEX_META).write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, store: ChunkStore, index_dir: Path) -> DenseRetriever:
        index_path = index_dir / INDEX_FILE
        meta_path = index_dir / INDEX_META
        for path in (index_path, meta_path):
            if not path.exists():
                raise DenseIndexError(
                    f"No dense index at {path}. Run: uv run python scripts/build_index.py"
                )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        index = faiss.read_index(str(index_path))
        embedder = get_embedder(meta["model_id"])
        return cls(store, index, list(meta["chunk_ids"]), embedder)

    # --- Search ---------------------------------------------------------------

    async def search(
        self,
        query: str,
        top_k: int,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievedChunk]:
        if not query.strip():
            return []

        # DECISION: over-fetch, then filter, rather than filtering inside FAISS.
        #   A flat index has no native metadata filter. Fetching exactly top_k and then
        #   dropping filtered hits silently returns fewer than requested -- and on a
        #   document-scoped request it usually returns nothing at all, because the
        #   nearest neighbours corpus-wide are rarely in the one allowed document.
        #   The over-fetch factor is bounded by the index size, so worst case is a full
        #   scan, which on this corpus is free.
        fetch = top_k if filters is None else min(len(self._chunk_ids), max(top_k * 20, 200))

        vector = self._embedder.encode_one(query).reshape(1, -1)
        scores, positions = self._index.search(vector, fetch)

        results: list[RetrievedChunk] = []
        for score, position in zip(scores[0], positions[0], strict=True):
            if position < 0:  # FAISS pads with -1 when fewer than `fetch` exist
                continue
            chunk = self._store.get_chunk(self._chunk_ids[int(position)])
            if chunk is None:
                continue
            document = self._store.get_document(chunk.document_id)
            if not apply_filters(
                chunk.document_id, bool(document and document.is_synthetic), filters
            ):
                continue

            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    bm25_rank=None,
                    dense_rank=len(results) + 1,
                    fused_score=float(score),
                    rerank_score=None,
                    retrieval_query=query,
                    pulled_by="search",
                )
            )
            if len(results) >= top_k:
                break

        return results

    @property
    def total_candidates(self) -> int:
        return len(self._chunk_ids)
