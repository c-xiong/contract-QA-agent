"""The deterministic citation gate. SPEC 13.4 layer 1.

AUTHOR-OWNED (CLAUDE.md rule 2). Every `# DECISION:` is a choice to defend.

This module is the concrete answer to "how do you stop it hallucinating citations".
The answer is not a prompt instruction. It is this: the model's output is parsed,
each citation is checked against the store and against the evidence actually
retrieved, and an answer whose citations fail is not returned as written.

Five checks, in the order that produces the most specific error:

  1. parseable          -- the span is a well-formed citation at all
  2. document exists    -- doc-014 was ingested
  3. document allowed   -- doc-014 is within this request's allowlist
  4. locator exists     -- that page, and that section, exist in that document
  5. grounded           -- the citation points at evidence that was actually retrieved

Check 5 is the one that matters most and the one a naive implementation omits. A
model handed evidence from doc-001 can emit a citation to doc-002 that is real,
paginated correctly, and entirely fabricated as support. Checks 1-4 all pass. Only
check 5 catches it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.evidence.citation_parser import extract_citations, normalize_compound_citations
from app.ingestion.store import ChunkStore
from app.schemas.evidence import Citation, CitationError
from app.schemas.retrieval import RetrievedChunk


@dataclass(frozen=True)
class VerificationResult:
    """The outcome of verifying one answer's citations."""

    citations: list[Citation] = field(default_factory=list)
    errors: list[CitationError] = field(default_factory=list)
    uncited: bool = False
    normalized_answer: str | None = None

    @property
    def ok(self) -> bool:
        return not self.errors and not self.uncited

    @property
    def valid_citations(self) -> list[Citation]:
        bad = {error.raw for error in self.errors}
        return [c for c in self.citations if c.raw not in bad]

    def summary(self) -> str:
        if self.uncited:
            return "no citations found in answer"
        if not self.errors:
            return f"{len(self.citations)} citation(s), all verified"
        codes = ", ".join(sorted({e.code for e in self.errors}))
        return f"{len(self.errors)} of {len(self.citations)} citation(s) failed: {codes}"


class CitationVerifier:
    """Checks an answer's citations against the store and the retrieved evidence."""

    def __init__(self, store: ChunkStore) -> None:
        self._store = store

    def verify(
        self,
        answer: str,
        retrieved: list[RetrievedChunk],
        *,
        allowed_document_ids: list[str] | None = None,
    ) -> VerificationResult:
        """Verify every citation in `answer`.

        `retrieved` is what the agent actually saw. Verification is against that set,
        not against the whole corpus, which is what makes check 5 meaningful.
        """
        normalized_answer = normalize_compound_citations(answer)
        citations, malformed = extract_citations(normalized_answer)

        errors: list[CitationError] = [
            CitationError(
                code="unparseable",
                raw=span,
                detail="Looks like a citation but does not match the format [doc-NNN, p. N, §N.N].",
            )
            for span in malformed
        ]

        # DECISION: grounding is keyed on (document_id, page_number), not on chunk_id.
        #   The writer is shown excerpts and asked to cite them; it never sees a
        #   chunk_id, so it cannot echo one back. Page is the finest locator the model
        #   is actually given, so page is what can honestly be checked.
        #   Rejected: requiring the model to emit chunk_ids. It would make check 5
        #   exact, but it puts an opaque internal token into user-facing output and
        #   invites the model to invent plausible ones.
        #   Consequence to be aware of: two chunks on the same page are
        #   indistinguishable to this check. A citation to the right page but the
        #   wrong clause on that page passes. That is a real gap, and it is precisely
        #   what SPEC 13.4 layer 2 (model-based claim support) exists to close.
        grounded_pages: set[tuple[str, int | None]] = {
            (item.chunk.document_id, item.chunk.page_number) for item in retrieved
        }
        grounded_documents: set[str] = {item.chunk.document_id for item in retrieved}

        for citation in citations:
            error = self._check(
                citation,
                grounded_pages=grounded_pages,
                grounded_documents=grounded_documents,
                allowed_document_ids=allowed_document_ids,
            )
            if error is not None:
                errors.append(error)

        # DECISION: an answer with no citations at all is a distinct failure, not a
        # passing answer with zero errors.
        #   Without this, the cheapest way for the model to satisfy the gate is to
        #   cite nothing, and the gate silently rewards exactly the behavior it exists
        #   to prevent. `uncited` is reported separately from `errors` because the
        #   remedy differs: a bad citation gets one repair attempt, while an uncited
        #   answer means the writer ignored its instructions entirely.
        uncited = not citations and not malformed and bool(answer.strip())

        return VerificationResult(
            citations=citations,
            errors=errors,
            uncited=uncited,
            normalized_answer=normalized_answer,
        )

    def _check(
        self,
        citation: Citation,
        *,
        grounded_pages: set[tuple[str, int | None]],
        grounded_documents: set[str],
        allowed_document_ids: list[str] | None,
    ) -> CitationError | None:
        document_id = citation.document_id

        if not self._store.has_document(document_id):
            return CitationError(
                code="unknown_document",
                raw=citation.raw,
                citation=citation,
                detail=f"{document_id} is not in the corpus.",
            )

        if allowed_document_ids is not None and document_id not in allowed_document_ids:
            return CitationError(
                code="document_not_allowed",
                raw=citation.raw,
                citation=citation,
                detail=f"{document_id} is outside this request's allowlist.",
            )

        if citation.page_number is not None and not self._store.has_page(
            document_id, citation.page_number
        ):
            pages = self._store.pages_of(document_id)
            extent = f"1-{max(pages)}" if pages else "none"
            return CitationError(
                code="page_not_found",
                raw=citation.raw,
                citation=citation,
                detail=(
                    f"{document_id} has no indexed page {citation.page_number} "
                    f"(pages with text: {extent})."
                ),
            )

        # DECISION: a wrong section is an error even when the page is right.
        #   A citation is a claim about a location. Half-right is wrong, and letting
        #   it pass would mean the §N.N component is decorative.
        #   Rejected: warning instead of failing. The chunker's section detection is
        #   imperfect, so this will occasionally reject a citation whose section is
        #   correct in the document but absent from our index -- a false positive
        #   caused by our own parser. That is the right direction to err in a system
        #   whose claim is provenance, but it is a real cost, and it is why
        #   `has_section` checks the whole document rather than only the cited page.
        if citation.section_id is not None and not self._store.has_section(
            document_id, citation.section_id
        ):
            return CitationError(
                code="section_not_found",
                raw=citation.raw,
                citation=citation,
                detail=f"{document_id} has no indexed section {citation.section_id}.",
            )

        if citation.page_number is not None:
            if (document_id, citation.page_number) not in grounded_pages:
                return CitationError(
                    code="not_in_evidence",
                    raw=citation.raw,
                    citation=citation,
                    detail=(
                        f"Nothing from {document_id} page {citation.page_number} was "
                        "retrieved for this question."
                    ),
                )
        elif document_id not in grounded_documents:
            return CitationError(
                code="not_in_evidence",
                raw=citation.raw,
                citation=citation,
                detail=f"Nothing from {document_id} was retrieved for this question.",
            )

        return None
