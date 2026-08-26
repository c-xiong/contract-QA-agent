"""Typed tool contracts. See docs/SPEC.md section 12.

Each tool is an ordinary, testable Python function underneath its LLM-facing wrapper.
Input validation is Pydantic, so an out-of-range top_k or a malformed locator is
rejected before any work happens, and the failure is a validation error with a field
name rather than an exception three layers down.

Sprint 0 implements `search_contracts` only. `open_document_section` and
`get_document_metadata` arrive in Sprint 2 with the agent loop that needs them; the
schemas are here because the ChunkStore already supports them and the shape is
settled.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.ingestion.store import ChunkStore
from app.retrieval.base import Retriever
from app.schemas.retrieval import RetrievalFilters, RetrievedChunk


class SearchContractsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)
    document_ids: list[str] | None = None


class SearchContractsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[RetrievedChunk] = Field(default_factory=list)
    total_candidates: int = 0


class OpenDocumentSectionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    page_number: int | None = Field(default=None, ge=1)
    section_id: str | None = None

    @model_validator(mode="after")
    def exactly_one_locator(self) -> OpenDocumentSectionInput:
        if (self.page_number is None) == (self.section_id is None):
            raise ValueError("provide exactly one of page_number or section_id")
        return self


class DocumentSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    title: str
    section_path: list[str] = Field(default_factory=list)
    page_number: int
    text: str


class GetDocumentMetadataInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str


async def search_contracts(
    payload: SearchContractsInput,
    retriever: Retriever,
    *,
    allowed_document_ids: list[str] | None = None,
) -> SearchContractsOutput:
    """Search the corpus.

    The allowlist is intersected here rather than trusted from the model's arguments.
    A model asking for a document outside the allowlist gets no results from it, and
    the citation verifier rejects any citation to it as a second, independent check.
    """
    requested = payload.document_ids
    if allowed_document_ids is not None:
        requested = (
            [d for d in requested if d in allowed_document_ids]
            if requested is not None
            else list(allowed_document_ids)
        )

    filters = RetrievalFilters(document_ids=requested) if requested is not None else None
    results = await retriever.search(payload.query, top_k=payload.top_k, filters=filters)
    return SearchContractsOutput(results=results, total_candidates=len(results))


def open_document_section(
    payload: OpenDocumentSectionInput, store: ChunkStore
) -> DocumentSection | None:
    """Fetch a section or page by locator. Wired into the agent in Sprint 2."""
    document = store.get_document(payload.document_id)
    if document is None:
        return None

    chunks = (
        store.find_page(payload.document_id, payload.page_number)
        if payload.page_number is not None
        else store.find_section(payload.document_id, payload.section_id or "")
    )
    if not chunks:
        return None

    return DocumentSection(
        document_id=payload.document_id,
        title=document.title,
        section_path=chunks[0].section_path,
        page_number=chunks[0].page_number,
        text="\n\n".join(c.text for c in chunks),
    )
