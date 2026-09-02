"""The deterministic citation gate.

These tests are the concrete answer to "how do you stop it hallucinating citations."
Each covers one failure mode with a distinct error code, because "citation
verification failed" as a single bucket tells you nothing about which of six very
different things went wrong.
"""

from __future__ import annotations

from app.evidence.citation_verifier import CitationVerifier
from app.ingestion.store import ChunkStore
from tests.conftest import make_retrieved


def retrieved_from(store: ChunkStore, *chunk_ids: str):
    items = []
    for rank, chunk_id in enumerate(chunk_ids, start=1):
        chunk = store.get_chunk(chunk_id)
        assert chunk is not None
        items.append(make_retrieved(chunk, rank=rank))
    return items


class TestPasses:
    def test_citation_to_retrieved_evidence_verifies(self, store: ChunkStore) -> None:
        verifier = CitationVerifier(store)
        result = verifier.verify(
            "The cap is limited [doc-900, p. 1, §8.1].",
            retrieved_from(store, "doc-900-c0001"),
        )
        assert result.ok is True
        assert len(result.citations) == 1

    def test_page_only_citation_verifies(self, store: ChunkStore) -> None:
        result = CitationVerifier(store).verify(
            "Delaware law applies [doc-900, p. 3].",
            retrieved_from(store, "doc-900-c0003"),
        )
        assert result.ok is True

    def test_compound_citation_is_normalized_then_verified(self, store: ChunkStore) -> None:
        result = CitationVerifier(store).verify(
            "The cap and carve-out both apply [doc-900, p. 1, §8.1; doc-900, p. 2, §8.3].",
            retrieved_from(store, "doc-900-c0001", "doc-900-c0002"),
        )

        assert result.ok is True
        assert [citation.section_id for citation in result.citations] == ["8.1", "8.3"]
        assert result.normalized_answer is not None
        assert ";" not in result.normalized_answer
        assert "[doc-900, p. 1, §8.1] [doc-900, p. 2, §8.3]" in result.normalized_answer


class TestFailures:
    def test_hallucinated_document(self, store: ChunkStore) -> None:
        result = CitationVerifier(store).verify(
            "Something [doc-999, p. 1, §8.1].", retrieved_from(store, "doc-900-c0001")
        )
        assert [e.code for e in result.errors] == ["unknown_document"]

    def test_page_beyond_what_was_indexed(self, store: ChunkStore) -> None:
        result = CitationVerifier(store).verify(
            "Something [doc-900, p. 99, §8.1].", retrieved_from(store, "doc-900-c0001")
        )
        assert [e.code for e in result.errors] == ["page_not_found"]

    def test_section_that_does_not_exist(self, store: ChunkStore) -> None:
        """Half-right is wrong: a citation is a claim about a location."""
        result = CitationVerifier(store).verify(
            "Something [doc-900, p. 1, §99.9].", retrieved_from(store, "doc-900-c0001")
        )
        assert [e.code for e in result.errors] == ["section_not_found"]

    def test_document_outside_the_request_allowlist(self, store: ChunkStore) -> None:
        result = CitationVerifier(store).verify(
            "Something [doc-901, p. 1, §1].",
            retrieved_from(store, "doc-901-c0001"),
            allowed_document_ids=["doc-900"],
        )
        assert [e.code for e in result.errors] == ["document_not_allowed"]

    def test_real_page_that_was_never_retrieved(self, store: ChunkStore) -> None:
        """The check a naive verifier omits, and the one that matters most.

        doc-900 page 3 exists and is correctly paginated. It was simply never shown to
        the model for this question. Checks 1-4 all pass; only grounding catches it.
        """
        result = CitationVerifier(store).verify(
            "Delaware law applies [doc-900, p. 3, §14.2].",
            retrieved_from(store, "doc-900-c0001"),
        )
        assert [e.code for e in result.errors] == ["not_in_evidence"]

    def test_malformed_citation_is_reported_as_unparseable(self, store: ChunkStore) -> None:
        result = CitationVerifier(store).verify(
            "Something [doc-9, p. 1].", retrieved_from(store, "doc-900-c0001")
        )
        assert [e.code for e in result.errors] == ["unparseable"]

    def test_answer_with_no_citations_is_a_distinct_failure(self, store: ChunkStore) -> None:
        """Otherwise the cheapest way to pass the gate is to cite nothing.

        The gate would then reward exactly the behavior it exists to prevent.
        """
        result = CitationVerifier(store).verify(
            "The cap is one million dollars.", retrieved_from(store, "doc-900-c0001")
        )
        assert result.uncited is True
        assert result.ok is False
        assert result.errors == []

    def test_empty_answer_is_not_reported_as_uncited(self, store: ChunkStore) -> None:
        result = CitationVerifier(store).verify("", retrieved_from(store, "doc-900-c0001"))
        assert result.uncited is False


class TestMixedResults:
    def test_valid_citations_are_separable_from_failed_ones(self, store: ChunkStore) -> None:
        """Enables returning only the supported portion (SPEC 13.4)."""
        result = CitationVerifier(store).verify(
            "Cap [doc-900, p. 1, §8.1] and elsewhere [doc-999, p. 1].",
            retrieved_from(store, "doc-900-c0001"),
        )
        assert len(result.errors) == 1
        assert [c.document_id for c in result.valid_citations] == ["doc-900"]

    def test_summary_names_the_error_codes(self, store: ChunkStore) -> None:
        result = CitationVerifier(store).verify(
            "Something [doc-999, p. 1].", retrieved_from(store, "doc-900-c0001")
        )
        assert "unknown_document" in result.summary()
