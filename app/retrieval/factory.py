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
from app.retrieval.query_expansion import QueryExpandedRetriever
from app.retrieval.rerank import CrossEncoderReranker
from app.retrieval.windowed import WindowedDenseRetriever

# The arms compared in Experiment A, in the order they are reported.
LEGACY_ARM_NAMES = ("bm25", "dense", "rrf_hybrid", "rrf_hybrid_rerank")
# The selected improvement is available explicitly; validation showed modest gains
# with regressions, so it does not silently replace existing runtime defaults.
ARM_NAMES = (*LEGACY_ARM_NAMES, "legal_query_coverage_rerank")
EXPERIMENTAL_ARM_NAMES = (
    "coverage_rerank",
    "windowed_rerank",
    "windowed_coverage_rerank",
    "legal_query_rerank",
)


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

    if name in {"coverage_rerank", "windowed_rerank", "windowed_coverage_rerank"}:
        dense = (
            WindowedDenseRetriever.load(store, index_dir / "windows")
            if name.startswith("windowed")
            else DenseRetriever.load(store, index_dir)
        )
        base = RrfHybridRetriever([Bm25Retriever(store), dense], k=rrf_k)
        return CrossEncoderReranker(base, balance_documents="coverage" in name)

    if name in {"legal_query_rerank", "legal_query_coverage_rerank"}:
        hybrid = RrfHybridRetriever(
            [Bm25Retriever(store), DenseRetriever.load(store, index_dir)], k=rrf_k
        )
        return CrossEncoderReranker(
            QueryExpandedRetriever(hybrid), balance_documents="coverage" in name, expand_query=True
        )

    raise UnknownArmError(
        f"Unknown retrieval arm {name!r}. Known arms: {', '.join((*ARM_NAMES, *EXPERIMENTAL_ARM_NAMES))}"
    )
