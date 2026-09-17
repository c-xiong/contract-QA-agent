"""Document-level schema. See docs/decisions.md"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CorpusSource = Literal["cuad", "contractnli", "edgar", "synthetic"]


class Document(BaseModel):
    """One contract, as ingested.

    ``document_id`` is assigned by the corpus manifest and is the only stable
    handle on a document anywhere in the system. Citations key on it, eval tasks
    key on it, and the retrieval allowlist keys on it. Titles are for display
    only: the corpus deliberately contains multiple versions of the same
    agreement, so a title is not an identifier. See docs/decisions.md.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(pattern=r"^doc-\d{3,}$")
    title: str
    agreement_type: str | None = None
    effective_date: date | None = None
    version: str | None = None
    source_path: str = Field(
        description="Original path as acquired, kept so any ingestion result can be re-derived."
    )
    page_count: int = Field(ge=0)
    corpus_source: CorpusSource
    is_synthetic: bool = False

    @property
    def is_empty(self) -> bool:
        """True for a document that yielded no pages.

        At least one CUAD PDF is a scanned image with no text layer (SPEC 6.2).
        Such a document must never become an unanswerable eval task, because it
        would be unanswerable for a parser reason rather than a contract reason.
        """
        return self.page_count == 0
