"""Construct retrieval arms by name.

One place that knows how to build every arm, so Experiment A is a loop over names and
the agent, the API, and the eval runner all resolve a retriever the same way. The agent
must never know which arm is active (SPEC 10.1) -- this is what keeps that true.
"""

from __future__ import annotations

from pathlib import Path

from app.ingestion.store import ChunkStore
from app.retrieval.base import Retriever
from app.retrieval.bm25 import Bm25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import DEFAULT_RRF_K, RrfHybridRetriever
from app.retrieval.rerank import CrossEncoderReranker

# The arms compared in Experiment A, in the order they are reported.
ARM_NAMES = ("bm25", "dense", "rrf_hybrid", "rrf_hybrid_rerank")


class UnknownArmError(ValueError):
    """The requested retrieval arm does not exist."""


def build_retriever(
    name: str,
    store: ChunkStore,
    index_dir: Path,
    *,
    rrf_k: int = DEFAULT_RRF_K,
) -> Retriever:
    """Build one retrieval arm.

    Dense-backed arms load a prebuilt FAISS index rather than embedding on the fly:
    embedding 7,000 chunks per arm per experiment run would make Experiment A slow
    enough to discourage re-running it, and re-running it is the point.
    """
    if name == "bm25":
        return Bm25Retriever(store)

    if name == "dense":
        return DenseRetriever.load(store, index_dir)

    if name == "rrf_hybrid":
        return RrfHybridRetriever(
            [Bm25Retriever(store), DenseRetriever.load(store, index_dir)], k=rrf_k
        )

    if name == "rrf_hybrid_rerank":
        base = RrfHybridRetriever(
            [Bm25Retriever(store), DenseRetriever.load(store, index_dir)], k=rrf_k
        )
        return CrossEncoderReranker(base)

    raise UnknownArmError(f"Unknown retrieval arm {name!r}. Known arms: {', '.join(ARM_NAMES)}")
