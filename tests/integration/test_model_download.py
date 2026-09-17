"""Opt-in semantic smoke checks using the actual Hugging Face retrieval models.

Run with ``uv run pytest -m model_download``. Downloads are permitted in the dedicated
CI job; API calls remain disabled. These synthetic examples are not evaluation data.
"""

from __future__ import annotations

import pytest

from app.ingestion.store import ChunkStore
from app.retrieval.dense import DenseRetriever
from app.retrieval.rerank import CrossEncoderReranker

pytestmark = pytest.mark.model_download


@pytest.fixture
def dense(store: ChunkStore) -> DenseRetriever:
    return DenseRetriever.build(store)


async def test_dense_matches_paraphrased_query(dense: DenseRetriever) -> None:
    results = await dense.search("which state's courts decide disputes", top_k=1)
    assert results and results[0].chunk.chunk_id == "doc-900-c0003"
    assert results[0].dense_rank == 1


async def test_real_cross_encoder_ranks_governing_law(dense: DenseRetriever) -> None:
    results = await CrossEncoderReranker(dense, depth=4).search(
        "what law governs this contract", top_k=2
    )
    assert len(results) == 2
    assert results[0].chunk.chunk_id == "doc-900-c0003"
    assert results[0].rerank_score is not None
    assert results[0].fused_score is not None
