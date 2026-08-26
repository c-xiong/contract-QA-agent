"""The retriever interface. See docs/SPEC.md section 10.1.

Every retriever implements this and nothing else. The agent must not know which one
is active: that is what makes Experiment A a swap of one object rather than a
rewrite of the agent, and what keeps "the agent got better" separable from "the
retriever got better".
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.schemas.retrieval import RetrievalFilters, RetrievedChunk


@runtime_checkable
class Retriever(Protocol):
    """Async because retrieval is I/O in every implementation but the in-memory one.

    BM25 over a few thousand chunks is CPU-bound and returns immediately, but dense
    retrieval, a reranker, and any hosted index are all network calls. Making the
    interface async now means adding those later does not change a single caller.
    """

    name: str

    async def search(
        self,
        query: str,
        top_k: int,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievedChunk]: ...


def apply_filters(
    chunk_document_id: str,
    is_synthetic: bool,
    filters: RetrievalFilters | None,
) -> bool:
    """Return True if a chunk survives the filters. Shared so every retriever agrees."""
    if filters is None:
        return True
    if filters.document_ids is not None and chunk_document_id not in filters.document_ids:
        return False
    return not (filters.exclude_synthetic and is_synthetic)
