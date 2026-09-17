"""Retrieval interface types. See docs/decisions.md"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.chunk import Chunk

PulledBy = Literal["search", "cross_reference"]


class RetrievalFilters(BaseModel):
    """Constraints applied inside the retriever, not requested of a model.

    ``document_ids`` is the request allowlist. It is enforced here and re-checked by
    the citation verifier, because a constraint that exists in only one place is a
    constraint that silently stops existing when that place is refactored.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_ids: list[str] | None = None
    exclude_synthetic: bool = False


class RetrievedChunk(BaseModel):
    """A chunk returned by retrieval, with the diagnostics needed to explain why.

    Component ranks are retained so a retrieval failure can be classified as "BM25
    missed it" versus "dense missed it" versus "fusion demoted it". Collapsing them
    into one score makes Experiment A uninterpretable. See SPEC 10.3.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk: Chunk
    bm25_rank: int | None = None
    dense_rank: int | None = None
    fused_score: float
    rerank_score: float | None = None
    retrieval_query: str
    pulled_by: PulledBy = "search"

    @property
    def document_id(self) -> str:
        return self.chunk.document_id

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


class SearchOutcome(BaseModel):
    """One search call's results plus how many candidates were considered."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    results: list[RetrievedChunk] = Field(default_factory=list)
    total_candidates: int = 0
