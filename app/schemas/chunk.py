"""Chunk-level schema. See docs/decisions.md"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    """A retrievable span of one document, carrying its full provenance.

    ``section_path`` and ``outbound_references`` are what make cross-reference
    resolution (SPEC 9.5) possible. They are populated even when empty rather
    than omitted, so a downstream consumer can distinguish "no references found"
    from "this pipeline stage dropped the field."
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    document_id: str = Field(pattern=r"^doc-\d{3,}$")
    document_title: str
    section_path: list[str] = Field(
        default_factory=list,
        description='Numbering components, outermost first: ["8", "8.1"] for Section 8.1.',
    )
    section_title: str | None = None
    page_number: int | None = Field(
        default=None,
        ge=1,
        description=(
            "1-based, matching how a reader cites a page. None when the source has no "
            "pagination at all -- ContractNLI ships extracted text, not paginated PDFs "
            "(SPEC 6.3). Rendering a page number for such a document would be fabricated "
            "provenance in a system whose entire claim is provenance, so the field is "
            "honestly empty and the citation falls back to its section form."
        ),
    )
    text: str
    token_count: int = Field(ge=0)
    outbound_references: list[str] = Field(
        default_factory=list,
        description='Section identifiers cited inside this chunk, e.g. ["8.3"] for '
        '"subject to Section 8.3". Feeds cross-reference resolution in Sprint 2.',
    )

    @property
    def section_id(self) -> str | None:
        """The most specific section identifier, or None if numbering was not detected."""
        return self.section_path[-1] if self.section_path else None

    def citation(self) -> str:
        """Render this chunk's provenance in the internal citation format (SPEC 13.3).

        Three forms, all accepted by the parser and the verifier:
            [doc-014, p. 12, §8.1]   paginated, numbered
            [doc-014, p. 12]         paginated, numbering not detected
            [doc-051, §s37]          unpaginated source (ContractNLI)
        """
        parts = [self.document_id]
        if self.page_number is not None:
            parts.append(f"p. {self.page_number}")
        if self.section_id is not None:
            parts.append(f"§{self.section_id}")
        return f"[{', '.join(parts)}]"
