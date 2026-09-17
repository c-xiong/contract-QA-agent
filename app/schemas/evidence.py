"""Evidence and citation types. See docs/decisions.md"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.retrieval import PulledBy


class Evidence(BaseModel):
    """One excerpt selected to support an answer, with provenance intact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str
    topic: str
    document_id: str = Field(pattern=r"^doc-\d{3,}$")
    document_title: str
    section_path: list[str] = Field(default_factory=list)
    section_title: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    excerpt: str
    source_chunk_id: str
    pulled_by: PulledBy = "search"

    @property
    def section_id(self) -> str | None:
        return self.section_path[-1] if self.section_path else None


class Citation(BaseModel):
    """A parsed citation. Not necessarily a valid one -- validity is the verifier's job."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    page_number: int | None = None
    section_id: str | None = None
    raw: str = Field(description="The exact source text, kept so errors can quote it.")

    def render(self) -> str:
        """Re-render in canonical internal form (SPEC 13.3)."""
        parts = [self.document_id]
        if self.page_number is not None:
            parts.append(f"p. {self.page_number}")
        if self.section_id is not None:
            parts.append(f"§{self.section_id}")
        return f"[{', '.join(parts)}]"


CitationErrorCode = Literal[
    "unparseable",
    "unknown_document",
    "document_not_allowed",
    "page_not_found",
    "section_not_found",
    "not_in_evidence",
]


class CitationError(BaseModel):
    """A single failed citation check, specific enough to act on.

    The code is a closed set rather than free text so that failures can be counted
    per category across an eval run. "citation verification failed" as a bare string
    tells you nothing about whether the model hallucinated a document, drifted a page
    number, or cited real evidence it was never shown.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: CitationErrorCode
    raw: str
    detail: str
    citation: Citation | None = None
