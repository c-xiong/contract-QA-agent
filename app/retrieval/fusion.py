"""Reciprocal Rank Fusion over BM25 and dense retrieval.

DECISION-BEARING MODULE. See docs/decisions.md.

    RRF(d) = sum over lists of 1 / (k + rank_i(d))

The hybrid arm of Experiment A. It exists because BM25 and dense fail on complementary
inputs: BM25 misses a question phrased in different words from the clause; dense misses
an exact section label, a dollar figure, or a party name. Fusing them should beat both
on a category mix that contains each kind of question -- and Experiment A is what says
whether it actually does.
"""

from __future__ import annotations

import asyncio

from app.retrieval.base import Retriever
from app.schemas.retrieval import RetrievalFilters, RetrievedChunk

# DECISION: k = 60.
#   This is the constant from Cormack, Clarke and Buettcher (SIGIR 2009), the paper that
#   introduced RRF, and it is the value nearly every implementation uses. Choosing it
#   because it is conventional is a weak reason on its own, so here is what it actually
#   does and why the value is defensible.
#
#   k controls how sharply rank position is discounted. The score contribution of rank 1
#   is 1/(k+1) and of rank 10 is 1/(k+10); the RATIO between them is what matters:
#
#       k=0    rank 1 is 10.0x rank 10   -- top hit dominates, fusion barely blends
#       k=10   rank 1 is  1.8x rank 10
#       k=60   rank 1 is  1.15x rank 10  -- ranks are nearly flat within the top 10
#       k=200  rank 1 is  1.04x rank 10  -- position almost stops mattering
#
#   At k=60 a document appearing at rank 8 in BOTH lists outscores one appearing at
#   rank 1 in only one list (2/68 = 0.0294 vs 1/61 = 0.0164). That is the behaviour we
#   want: agreement between two independent retrievers is stronger evidence than a
#   single retriever's confidence. Low k inverts that and makes fusion little better
#   than "whichever arm was more confident".
#
#   Rejected: k=0, which reduces fusion to a max-of-reciprocal-ranks and reproduces
#   whichever arm happened to rank something first. Rejected: tuning k on the eval set,
#   which would fit the constant to 44 tasks and report the result as a property of the
#   method. If k is ever tuned, it must be tuned on a split the reported numbers do not
#   come from, and the report must say so.
#
#   TO THE AUTHOR: this is the single most likely "why did you pick that number"
#   question in the whole repository. The table above is the answer. Consider running
#   Experiment A at k in {10, 60, 200} once -- three arms is cheap, retrieval-only
#   experiments call no model, and a measured sensitivity curve beats citing a paper.
DEFAULT_RRF_K = 60

# How deep to take each arm before fusing.
#   DECISION: fetch 4x the requested top_k from each arm.
#     Fusion can only rank documents it was given. If each arm supplies exactly top_k,
#     a chunk at rank 6 in both lists -- a strong consensus signal -- is invisible.
#     Rejected: fetching the whole corpus from each arm, which makes RRF a full re-rank
#     and drowns consensus in noise from arbitrarily-ranked tail results.
DEFAULT_DEPTH_MULTIPLIER = 4


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    *,
    k: int = DEFAULT_RRF_K,
) -> list[RetrievedChunk]:
    """Fuse ranked lists into one, preserving each arm's component rank.

    Component ranks survive fusion (SPEC 10.3) so that a retrieval failure can be
    diagnosed as "BM25 missed it" versus "dense missed it" versus "fusion demoted it".
    Collapsing them into a single score makes Experiment A uninterpretable -- you would
    see that hybrid lost without being able to say why.
    """
    scores: dict[str, float] = {}
    merged: dict[str, RetrievedChunk] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            chunk_id = item.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

            existing = merged.get(chunk_id)
            if existing is None:
                merged[chunk_id] = item
            else:
                # Carry whichever component rank each arm assigned. An item found by
                # both arms ends up with both populated; one found by a single arm keeps
                # None for the other, which is itself the diagnostic.
                merged[chunk_id] = existing.model_copy(
                    update={
                        "bm25_rank": existing.bm25_rank
                        if existing.bm25_rank is not None
                        else item.bm25_rank,
                        "dense_rank": existing.dense_rank
                        if existing.dense_rank is not None
                        else item.dense_rank,
                    }
                )

    # Sort by fused score descending, then chunk_id ascending. The tiebreak is not
    # cosmetic: RRF produces exact ties constantly (any two chunks at the same rank in
    # the same single list score identically), and an unstable order would make eval
    # runs irreproducible for a reason unrelated to the system under test.
    ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))

    return [
        merged[chunk_id].model_copy(update={"fused_score": score}) for chunk_id, score in ordered
    ]


class RrfHybridRetriever:
    """Runs BM25 and dense concurrently, then fuses."""

    name = "rrf_hybrid"

    def __init__(
        self,
        arms: list[Retriever],
        *,
        k: int = DEFAULT_RRF_K,
        depth_multiplier: int = DEFAULT_DEPTH_MULTIPLIER,
    ) -> None:
        if not arms:
            raise ValueError("RrfHybridRetriever needs at least one arm")
        self._arms = arms
        self._k = k
        self._depth_multiplier = depth_multiplier
        self.name = f"rrf_hybrid(k={k})" if k != DEFAULT_RRF_K else "rrf_hybrid"

    async def search(
        self,
        query: str,
        top_k: int,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievedChunk]:
        depth = top_k * self._depth_multiplier

        # Arms are independent, so run them concurrently. This is the payoff for the
        # Retriever protocol being async even though BM25 alone is CPU-bound: adding a
        # network-backed arm later changes nothing here.
        ranked_lists = await asyncio.gather(
            *(arm.search(query, top_k=depth, filters=filters) for arm in self._arms)
        )

        fused = reciprocal_rank_fusion(list(ranked_lists), k=self._k)
        return fused[:top_k]
