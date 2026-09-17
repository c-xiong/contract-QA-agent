"""Dense index and reranker mechanics using controlled model outputs, entirely offline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.ingestion.store import ChunkStore
from app.retrieval.base import Retriever
from app.retrieval.dense import DenseIndexError, DenseRetriever
from app.retrieval.embeddings import Embedder
from app.retrieval.rerank import CrossEncoderReranker
from app.schemas.retrieval import RetrievalFilters
from tests.conftest import make_chunk, make_document


class FakeEmbedder(Embedder):
    """Orthogonal fixture vectors; these assert index behavior, not model quality."""

    def __init__(self) -> None:
        self.model_id = "fixture-vectors"
        self.dimension = 3

    def encode(self, texts: list[str], *, batch_size: int = 64) -> np.ndarray:
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "law" in lowered:
                vectors.append([0.0, 1.0, 0.0])
            elif "insurance" in lowered:
                vectors.append([0.0, 0.0, 1.0])
            else:
                vectors.append([1.0, 0.0, 0.0])
        return np.asarray(vectors, dtype="float32")


@pytest.fixture
def dense(monkeypatch: pytest.MonkeyPatch) -> DenseRetriever:
    monkeypatch.setattr("app.retrieval.dense.get_embedder", lambda model_id: FakeEmbedder())
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
    async def test_orders_by_inner_product(self, dense: DenseRetriever) -> None:
        results = await dense.search("governing law", top_k=1)
        assert results and results[0].chunk.chunk_id == "doc-900-c2"
        assert results[0].fused_score == pytest.approx(1.0)

    async def test_faiss_padding_does_not_duplicate_results(self, dense: DenseRetriever) -> None:
        results = await dense.search("law", top_k=10)
        assert len(results) == 3
        assert len({item.chunk.chunk_id for item in results}) == 3

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

    async def test_allowlist_with_no_candidates_returns_empty(self, dense: DenseRetriever) -> None:
        assert await dense.search("law", 2, RetrievalFilters(document_ids=["doc-999"])) == []

    def test_satisfies_the_retriever_protocol(self, dense: DenseRetriever) -> None:
        assert isinstance(dense, Retriever)


class TestPersistence:
    async def test_round_trips(self, dense: DenseRetriever, tmp_path: Path) -> None:
        dense.save(tmp_path)
        reloaded = DenseRetriever.load(dense._store, tmp_path)
        assert reloaded.total_candidates == dense.total_candidates
        assert await reloaded.search("law", 3) == await dense.search("law", 3)

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
    @pytest.fixture(autouse=True)
    def cross_encoder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class ControlledCrossEncoder:
            def __init__(self, model_id: str) -> None:
                pass

            def predict(
                self, pairs: list[tuple[str, str]], *, show_progress_bar: bool
            ) -> list[float]:
                return [3.0 if "Governing Law" in text else -2.0 for _, text in pairs]

        monkeypatch.setattr("app.retrieval.rerank.CrossEncoder", ControlledCrossEncoder)

    async def test_reorders_and_preserves_the_fused_score(self, dense: DenseRetriever) -> None:
        """Both numbers are kept so a trace can show fusion ranked a chunk 12th and the
        reranker promoted it to 1st -- the evidence for whether reranking helped."""
        reranker = CrossEncoderReranker(dense, depth=3)
        baseline = await dense.search("liability", top_k=3)
        assert baseline[0].chunk.chunk_id == "doc-900-c1"
        results = await reranker.search("liability", top_k=2)
        assert results
        assert results[0].rerank_score is not None
        assert results[0].fused_score is not None
        assert results[0].chunk.chunk_id == "doc-900-c2"
        assert results[0].rerank_score == 3.0
        assert results[0].fused_score == 0.0
        assert all(item.rerank_score is None for item in baseline)

    async def test_forwards_document_filter(self, dense: DenseRetriever) -> None:
        results = await CrossEncoderReranker(dense, depth=3).search(
            "law", top_k=2, filters=RetrievalFilters(document_ids=["doc-901"])
        )
        assert [item.chunk.document_id for item in results] == ["doc-901"]

    async def test_ties_use_chunk_id_order(self, dense: DenseRetriever) -> None:
        results = await CrossEncoderReranker(dense, depth=3).search("liability", top_k=3)
        assert [item.chunk.chunk_id for item in results[1:]] == ["doc-900-c1", "doc-901-c1"]

    async def test_empty_base_result_returns_empty(self, dense: DenseRetriever) -> None:
        reranker = CrossEncoderReranker(dense, depth=3)
        assert await reranker.search("   ", top_k=3) == []

    def test_name_records_the_composition(self, dense: DenseRetriever) -> None:
        assert CrossEncoderReranker(dense).name == "dense+rerank"
