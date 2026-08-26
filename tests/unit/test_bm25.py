"""BM25 retrieval behavior."""

from __future__ import annotations

import pytest

from app.ingestion.store import ChunkStore
from app.retrieval.base import Retriever
from app.retrieval.bm25 import Bm25Retriever, tokenize
from app.schemas.retrieval import RetrievalFilters
from tests.conftest import make_chunk, make_document


class TestTokenize:
    def test_dotted_section_numbers_stay_one_token(self) -> None:
        """Splitting "8.1" into "8" and "1" makes a section query match every chunk
        containing the digit 8."""
        assert "8.1" in tokenize("see Section 8.1 below")

    def test_decimal_amounts_stay_intact(self) -> None:
        assert "1.5" in tokenize("a fee of 1.5 percent")

    def test_casefolds(self) -> None:
        assert tokenize("Governing LAW") == ["governing", "law"]


class TestSearch:
    async def test_finds_the_lexically_matching_chunk(self, store: ChunkStore) -> None:
        results = await Bm25Retriever(store).search("governing law Delaware", top_k=3)
        assert results
        assert results[0].chunk.chunk_id == "doc-900-c0003"

    async def test_component_rank_is_recorded_for_diagnosis(self, store: ChunkStore) -> None:
        """Retained so a failure is diagnosable as "BM25 missed it" vs "fusion demoted it"."""
        results = await Bm25Retriever(store).search("governing law", top_k=3)
        assert results[0].bm25_rank == 1
        assert results[0].dense_rank is None
        assert results[0].retrieval_query == "governing law"
        assert results[0].pulled_by == "search"

    async def test_respects_top_k(self, store: ChunkStore) -> None:
        assert len(await Bm25Retriever(store).search("liability exceed negligence", top_k=1)) == 1

    async def test_high_document_frequency_terms_suppress_themselves(self) -> None:
        """Okapi IDF is log((N-n+0.5)/(n+0.5)) with no +1 smoothing.

        A term in more than about half the chunks therefore scores negative, and
        rank_bm25's epsilon floor scales with average_idf, which is itself negative
        here. This is stopword removal without a stopword list -- usually desirable,
        but it means a query of entirely common vocabulary retrieves nothing.
        Returning nothing and abstaining beats returning chunks BM25 could not rank.
        """
        documents = [make_document(f"doc-90{i}") for i in range(4)]
        chunks = [
            make_chunk(f"doc-90{i}-c1", f"doc-90{i}", text="liability cap provision")
            for i in range(4)
        ]
        results = await Bm25Retriever(ChunkStore(documents, chunks)).search("liability", top_k=5)
        assert results == []

    async def test_zero_scoring_chunks_are_not_returned_to_pad_top_k(
        self, store: ChunkStore
    ) -> None:
        """Padding gives the writer irrelevant evidence and invites a confident answer
        built on nothing."""
        results = await Bm25Retriever(store).search("Delaware", top_k=10)
        assert len(results) < 4
        assert all(r.fused_score > 0 for r in results)

    async def test_query_with_no_indexable_tokens_returns_nothing(self, store: ChunkStore) -> None:
        assert await Bm25Retriever(store).search("!!! ???", top_k=5) == []

    async def test_results_are_deterministic_across_runs(self, store: ChunkStore) -> None:
        """Non-deterministic ordering makes eval runs unreproducible for a reason
        unrelated to the system under test."""
        retriever = Bm25Retriever(store)
        first = await retriever.search("liability", top_k=5)
        second = await retriever.search("liability", top_k=5)
        assert [r.chunk.chunk_id for r in first] == [r.chunk.chunk_id for r in second]


class TestFilters:
    async def test_document_allowlist_is_enforced_in_the_retriever(self, store: ChunkStore) -> None:
        results = await Bm25Retriever(store).search(
            "definitions confidential information",
            top_k=5,
            filters=RetrievalFilters(document_ids=["doc-900"]),
        )
        assert all(r.chunk.document_id == "doc-900" for r in results)

    async def test_synthetic_exclusion(self) -> None:
        """Mirrors SPEC 6.6: a synthetic near-duplicate paraphrase of a real clause.

        Five documents with the query term in two of them, so its IDF stays positive
        and this exercises the filter rather than the negative-IDF path above.
        """
        documents = [make_document(f"doc-90{i}") for i in range(5)]
        documents[1] = documents[1].model_copy(update={"is_synthetic": True})
        chunks = [
            make_chunk("doc-900-c1", "doc-900", text="aggregate liability shall not exceed fees"),
            make_chunk("doc-901-c1", "doc-901", text="total liability shall not exceed amounts"),
            make_chunk("doc-902-c1", "doc-902", text="governing law of the state of delaware"),
            make_chunk("doc-903-c1", "doc-903", text="notice period for renewal is thirty days"),
            make_chunk("doc-904-c1", "doc-904", text="insurance coverage requirements apply"),
        ]
        retriever = Bm25Retriever(ChunkStore(documents, chunks))

        unfiltered = await retriever.search("liability exceed", top_k=5)
        assert {r.chunk.document_id for r in unfiltered} == {"doc-900", "doc-901"}

        results = await retriever.search(
            "liability exceed", top_k=5, filters=RetrievalFilters(exclude_synthetic=True)
        )
        assert [r.chunk.document_id for r in results] == ["doc-900"]


class TestConstruction:
    def test_empty_store_is_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="empty store"):
            Bm25Retriever(ChunkStore([], []))

    def test_satisfies_the_retriever_protocol(self, store: ChunkStore) -> None:
        """The agent must not depend on which retriever is active."""
        assert isinstance(Bm25Retriever(store), Retriever)
