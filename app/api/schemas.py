"""Request and response models for the HTTP API. See docs/decisions.md"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.agent.runner import ResearchResult
from app.schemas.evidence import Evidence


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)
    document_ids: list[str] | None = Field(
        default=None,
        description="Restrict the search to these documents. Enforced in the retriever "
        "and re-checked by the citation verifier.",
    )
    include_trace: bool = False


class RenderedCitation(BaseModel):
    """A citation in both forms.

    SPEC 13.3: the internal format is keyed on document_id and is what the verifier
    checks; the display form uses the title and is what a person reads. Both are
    returned so a caller never has to reconstruct either, and so the human-readable
    form can never drift from the checked one.
    """

    model_config = ConfigDict(extra="forbid")

    internal: str
    display: str
    document_id: str
    document_title: str
    page_number: int | None = None
    section_id: str | None = None


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    document_id: str
    document_title: str
    page_number: int | None = None
    section_id: str | None = None
    excerpt: str
    pulled_by: str


class TraceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: str
    detail: str


class ResearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str
    status: str
    abstained: bool
    citations: list[RenderedCitation] = Field(default_factory=list)
    citation_errors: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    documents_searched: list[str] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    model_id: str = "stub"
    trace: list[TraceItem] | None = None


class DocumentSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    title: str
    agreement_type: str | None = None
    page_count: int
    corpus_source: str
    is_synthetic: bool
    chunk_count: int


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    documents: int
    chunks: int
    retriever: str
    model_id: str
    live_model: bool
    note: str | None = Field(
        default=None,
        description="Set when the service started in a degraded state -- a missing index, "
        "or a live model that was requested and could not be built.",
    )


def render_citations(result: ResearchResult, titles: dict[str, str]) -> list[RenderedCitation]:
    rendered: list[RenderedCitation] = []
    for citation in result.citations:
        title = titles.get(citation.document_id, citation.document_id)
        parts = [title]
        if citation.page_number is not None:
            parts.append(f"p. {citation.page_number}")
        if citation.section_id is not None:
            parts.append(f"§{citation.section_id}")
        rendered.append(
            RenderedCitation(
                internal=citation.render(),
                display=", ".join(parts),
                document_id=citation.document_id,
                document_title=title,
                page_number=citation.page_number,
                section_id=citation.section_id,
            )
        )
    return rendered


def render_evidence(evidence: list[Evidence]) -> list[EvidenceItem]:
    return [
        EvidenceItem(
            evidence_id=e.evidence_id,
            document_id=e.document_id,
            document_title=e.document_title,
            page_number=e.page_number,
            section_id=e.section_id,
            excerpt=e.excerpt,
            pulled_by=e.pulled_by,
        )
        for e in evidence
    ]
