"""Dense retrieval and reranking.

These load real models, so they are slower than the rest of the unit suite. They stay
here rather than in integration because they exercise one module each and need no
corpus on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.store import ChunkStore
from app.retrieval.base import Retriever
from app.retrieval.dense import DenseIndexError, DenseRetriever
from app.retrieval.rerank import CrossEncoderReranker
from app.schemas.retrieval import RetrievalFilters
from tests.conftest import make_chunk, make_document


@pytest.fixture(scope="module")
def dense(request: pytest.FixtureRequest) -> DenseRetriever:
    documents = [make_document("doc-900", page_count=3), make_document("doc-901", page_count=1)]
    chunks = [
        make_chunk(
            "doc-900-c1",
            "doc-900",
            page_number=1,
            section_title="Limitation of Liability",
            text="Neither party shall be liable for amounts exceeding the fees paid.",
        ),
        make_chunk(
            "doc-900-c2",
            "doc-900",
            page_number=2,
            section_title="Governing Law",
            text="This Agreement is governed by the laws of the State of Delaware.",
        ),
        make_chunk(
            "doc-901-c1",
            "doc-901",
            page_number=1,
            section_title="Insurance",
            text="Contractor shall maintain commercial general liability insurance.",
        ),
    ]
    return DenseRetriever.build(ChunkStore(documents, chunks))


class TestIndexText:
    def test_heading_is_prepended_to_the_body(self) -> None:
        """A chunk reading only "shall not exceed the fees paid" is ambiguous alone;
        prefixed with its heading it is not, and the heading lands inside the model's
        256-token window even when the body would be truncated."""
        chunk = make_chunk(section_title="Limitation of Liability", text="body text")
        assert DenseRetriever.index_text(chunk).startswith("Limitation of Liability")

    def test_falls_back_to_the_section_number(self) -> None:
        chunk = make_chunk(section_title=None, section_path=["8", "8.1"], text="body")
        assert DenseRetriever.index_text(chunk).startswith("Section 8.1")


class TestSearch:
    async def test_matches_on_meaning_not_wording(self, dense: DenseRetriever) -> None:
        """The reason a dense arm exists: no shared vocabulary with the clause."""
        results = await dense.search("which state's courts decide disputes", top_k=1)
        assert results and results[0].chunk.chunk_id == "doc-900-c2"

    async def test_records_the_dense_rank_only(self, dense: DenseRetriever) -> None:
        results = await dense.search("governing law", top_k=2)
        assert results[0].dense_rank == 1
        assert results[0].bm25_rank is None

    async def test_empty_query_returns_nothing(self, dense: DenseRetriever) -> None:
        assert await dense.search("   ", top_k=5) == []

    async def test_allowlist_is_honoured(self, dense: DenseRetriever) -> None:
        """Over-fetch then filter: fetching exactly top_k and dropping filtered hits
        returns nothing on a document-scoped request, because the global nearest
        neighbours are rarely inside the one allowed document."""
        results = await dense.search(
            "liability", top_k=5, filters=RetrievalFilters(document_ids=["doc-901"])
        )
        assert results
        assert all(r.chunk.document_id == "doc-901" for r in results)

    def test_satisfies_the_retriever_protocol(self, dense: DenseRetriever) -> None:
        assert isinstance(dense, Retriever)


class TestPersistence:
    def test_round_trips(self, dense: DenseRetriever, tmp_path: Path) -> None:
        dense.save(tmp_path)
        reloaded = DenseRetriever.load(dense._store, tmp_path)
        assert reloaded.total_candidates == dense.total_candidates

    def test_missing_index_names_the_build_command(self, tmp_path: Path) -> None:
        store = ChunkStore([make_document()], [make_chunk()])
        with pytest.raises(DenseIndexError, match=r"build_index\.py"):
            DenseRetriever.load(store, tmp_path)

    def test_stale_index_is_refused_rather_than_silently_used(
        self, dense: DenseRetriever, tmp_path: Path
    ) -> None:
        """The index is keyed on chunk ids. After a chunker change they no longer exist,
        and serving results from it would return chunks the store cannot resolve."""
        dense.save(tmp_path)
        other = ChunkStore([make_document("doc-950")], [make_chunk("doc-950-c1", "doc-950")])
        with pytest.raises(DenseIndexError, match="stale"):
            DenseRetriever.load(other, tmp_path)


class TestReranker:
    async def test_reorders_and_preserves_the_fused_score(self, dense: DenseRetriever) -> None:
        """Both numbers are kept so a trace can show fusion ranked a chunk 12th and the
        reranker promoted it to 1st -- the evidence for whether reranking helped."""
        reranker = CrossEncoderReranker(dense, depth=3)
        results = await reranker.search("what law governs this contract", top_k=2)
        assert results
        assert results[0].rerank_score is not None
        assert results[0].fused_score is not None
        assert results[0].chunk.chunk_id == "doc-900-c2"

    async def test_empty_base_result_returns_empty(self, dense: DenseRetriever) -> None:
        reranker = CrossEncoderReranker(dense, depth=3)
        assert await reranker.search("   ", top_k=3) == []

    def test_name_records_the_composition(self, dense: DenseRetriever) -> None:
        assert CrossEncoderReranker(dense).name == "dense+rerank"
