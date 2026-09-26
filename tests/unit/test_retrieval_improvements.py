"""Recall mechanics: tail coverage, scope, deduplication and comparison coverage."""

from pathlib import Path

import faiss
import numpy as np
import pytest

from app.ingestion.store import ChunkStore
from app.retrieval.coverage import comparison_documents, select_with_coverage
from app.retrieval.dense import DenseIndexError
from app.retrieval.rerank import CrossEncoderReranker
from app.retrieval.windowed import WindowedDenseRetriever
from app.schemas.retrieval import RetrievalFilters, RetrievedChunk
from tests.conftest import make_chunk, make_document
from tests.unit.test_dense_and_rerank import FakeEmbedder


class WordTokenizer:
    def __init__(self):
        self.words = {}

    def encode(self, text, *, add_special_tokens):
        tokens = []
        for word in text.split():
            self.words.setdefault(word, len(self.words))
            tokens.append(self.words[word])
        return tokens

    def decode(self, ids, *, skip_special_tokens):
        reverse = {value: word for word, value in self.words.items()}
        return " ".join(reverse[token] for token in ids)

    def num_special_tokens_to_add(self, *, pair):
        return 2


def test_windows_cover_tail_without_exceeding_model_budget():
    class Model:
        tokenizer = WordTokenizer()
        max_seq_length = 10

    embedder = FakeEmbedder()
    embedder._model = Model()
    body = " ".join(f"word{i}" for i in range(25))
    windows = embedder.window_texts(body, heading="Liability", overlap=2)
    assert all(text.startswith("Liability ") for text in windows)
    assert all(len(text.split()) + 2 <= 10 for text in windows)
    assert set(body.split()) <= set(" ".join(windows).split())
    assert "word24" in windows[-1]
    with pytest.raises(ValueError, match="overlap"):
        embedder.window_texts(body, overlap=100)


def window_retriever():
    documents = [make_document("doc-900"), make_document("doc-901")]
    chunks = [
        make_chunk("doc-900-a", "doc-900"),
        make_chunk("doc-900-b", "doc-900"),
        make_chunk("doc-901-a", "doc-901"),
    ]
    store = ChunkStore(documents, chunks)
    # Repeated windows for one chunk must not crowd out another chunk or scope.
    ids = ["doc-900-a"] * 250 + ["doc-900-b", "doc-901-a"]
    vectors = np.asarray([[1, 0, 0]] * 250 + [[0.8, 0.6, 0], [0.6, 0.8, 0]], dtype="float32")
    index = faiss.IndexFlatIP(3)
    index.add(vectors)
    return WindowedDenseRetriever(store, index, ids, FakeEmbedder())


async def test_windows_are_deduplicated_before_top_k_and_scope_is_exact():
    retriever = window_retriever()
    hits = await retriever.search("query", 3)
    assert [hit.chunk_id for hit in hits] == ["doc-900-a", "doc-900-b", "doc-901-a"]
    scoped = await retriever.search("query", 1, RetrievalFilters(document_ids=["doc-901"]))
    assert [hit.chunk_id for hit in scoped] == ["doc-901-a"]
    assert await retriever.search("query", 0) == []
    assert await retriever.search("query", 5, RetrievalFilters(document_ids=[])) == []


async def test_window_index_checks_text_changes_not_just_ids(tmp_path: Path, monkeypatch):
    retriever = window_retriever()
    monkeypatch.setattr("app.retrieval.dense.get_embedder", lambda _: FakeEmbedder())
    retriever.save(tmp_path)
    loaded = WindowedDenseRetriever.load(retriever._store, tmp_path)
    assert await loaded.search("query", 3) == await retriever.search("query", 3)
    changed = ChunkStore(
        retriever._store.documents,
        [chunk.model_copy(update={"text": "Changed body"}) for chunk in retriever._store.chunks],
    )
    with pytest.raises(DenseIndexError, match="Stale"):
        WindowedDenseRetriever.load(changed, tmp_path)


def test_coverage_only_activates_for_explicit_comparison():
    filters = RetrievalFilters(document_ids=["doc-900", "doc-901", "doc-900"])
    assert comparison_documents("Compare each agreement", filters) == ["doc-900", "doc-901"]
    assert comparison_documents("Find the governing law", filters) == []
    assert comparison_documents("Compare agreements", None) == []


def test_coverage_preserves_total_budget_and_best_per_document():
    ranked = [
        RetrievedChunk(
            chunk=make_chunk(f"doc-900-{i}", "doc-900"), fused_score=10 - i, retrieval_query="q"
        )
        for i in range(5)
    ]
    ranked += [
        RetrievedChunk(chunk=make_chunk("doc-901-a", "doc-901"), fused_score=1, retrieval_query="q")
    ]
    hits = select_with_coverage(ranked, ["doc-900", "doc-901"], 5)
    assert len(hits) == 5
    assert hits[0] == ranked[0]
    assert hits[-1] == ranked[-1]
    assert ranked[4] not in hits
    assert select_with_coverage(ranked, ["doc-900", "doc-901"], 1) == [ranked[0]]


async def test_balanced_candidate_pool_respects_budget_and_exclusions(monkeypatch):
    class Encoder:
        def __init__(self, model_id):
            pass

        def predict(self, pairs, **kwargs):
            return list(reversed(range(len(pairs))))

    class Base:
        name = "fixture"

        def __init__(self):
            self.calls = []

        async def search(self, query, top_k, filters=None):
            self.calls.append(filters)
            doc = filters.document_ids[0]
            return [
                RetrievedChunk(
                    chunk=make_chunk(f"{doc}-{i}", doc), fused_score=1, retrieval_query=query
                )
                for i in range(top_k)
            ]

    monkeypatch.setattr("app.retrieval.rerank.CrossEncoder", Encoder)
    base = Base()
    reranker = CrossEncoderReranker(base, depth=7, balance_documents=True)
    filters = RetrievalFilters(
        document_ids=["doc-900", "doc-901", "doc-902"], exclude_synthetic=True
    )
    candidates = await reranker.retrieve_candidates("Compare each contract", filters)
    assert len(candidates) == 7
    assert {c.document_id for c in candidates} == set(filters.document_ids)
    assert all(call.exclude_synthetic for call in base.calls)
    hits = await reranker.rerank_candidates("Compare each contract", candidates, 5, filters)
    assert len(hits) == 5
    assert {c.document_id for c in hits} == set(filters.document_ids)
