"""RRF fusion. The k=60 behaviour is the thing to pin down."""

from __future__ import annotations

import pytest

from app.retrieval.fusion import (
    DEFAULT_RRF_K,
    RrfHybridRetriever,
    reciprocal_rank_fusion,
)
from tests.conftest import make_chunk, make_retrieved


def ranked(*chunk_ids: str, dense: bool = False) -> list:
    items = []
    for rank, chunk_id in enumerate(chunk_ids, start=1):
        item = make_retrieved(make_chunk(chunk_id, "doc-900"), rank=rank)
        if dense:
            item = item.model_copy(update={"bm25_rank": None, "dense_rank": rank})
        items.append(item)
    return items


def consensus_lists() -> tuple[list, list]:
    """`solo` is rank 1 in one list only; `both` is rank 8 in each list.

    At k=60:  solo = 1/61 = 0.0164   both = 2/68 = 0.0294  -> consensus wins
    At k=0:   solo = 1/1  = 1.0      both = 2/8  = 0.25    -> confidence wins
    """
    filler = [f"f{i}" for i in range(1, 7)]
    left = ranked("solo", *filler, "both")
    right = ranked(*[f"g{i}" for i in range(1, 8)], "both", dense=True)
    return left, right


class TestScoring:
    def test_agreement_beats_single_list_confidence(self) -> None:
        """The property k=60 is chosen for, stated as a test.

        Consensus between two independent retrievers is stronger evidence than one
        retriever's confidence, and at k=60 the arithmetic says so.
        """
        left, right = consensus_lists()
        fused = reciprocal_rank_fusion([left, right], k=DEFAULT_RRF_K)
        assert fused[0].chunk.chunk_id == "both", (
            "a chunk found deep in BOTH arms should outrank a solo rank-1 hit"
        )

    def test_low_k_inverts_that_preference(self) -> None:
        """Why k matters, demonstrated rather than asserted.

        This is the whole justification for not leaving k at 0 or 10, and it is the
        question most likely to be asked about this module.
        """
        left, right = consensus_lists()

        at_60 = reciprocal_rank_fusion([left, right], k=60)
        at_0 = reciprocal_rank_fusion([left, right], k=0)

        assert at_60[0].chunk.chunk_id == "both", "k=60 rewards agreement"
        assert at_0[0].chunk.chunk_id != "both", "k=0 rewards whichever arm was loudest"

        # Concretely: at k=0 the deep-consensus chunk is beaten by rank-1 solo hits,
        # which is exactly the behaviour k=60 is chosen to avoid.
        rank_of_both = [i.chunk.chunk_id for i in at_0].index("both")
        assert rank_of_both > 0

    def test_score_matches_the_formula(self) -> None:
        fused = reciprocal_rank_fusion([ranked("a", "b")], k=60)
        assert fused[0].fused_score == pytest.approx(1 / 61)
        assert fused[1].fused_score == pytest.approx(1 / 62)


class TestComponentRanks:
    def test_both_ranks_are_preserved_after_fusion(self) -> None:
        """SPEC 10.3: a failure must be diagnosable as BM25-missed vs dense-missed vs
        fusion-demoted. Collapsing to one score makes Experiment A uninterpretable."""
        fused = reciprocal_rank_fusion([ranked("shared"), ranked("shared", dense=True)])
        assert fused[0].bm25_rank == 1
        assert fused[0].dense_rank == 1

    def test_single_arm_hit_keeps_the_other_rank_as_none(self) -> None:
        fused = reciprocal_rank_fusion([ranked("only_bm25"), ranked("only_dense", dense=True)])
        by_id = {f.chunk.chunk_id: f for f in fused}
        assert by_id["only_bm25"].dense_rank is None
        assert by_id["only_dense"].bm25_rank is None


class TestDeterminism:
    def test_ties_break_on_chunk_id(self) -> None:
        """RRF produces exact ties constantly; unstable order would make eval runs
        irreproducible for a reason unrelated to the system under test."""
        first = reciprocal_rank_fusion([ranked("b", "a")])
        second = reciprocal_rank_fusion([ranked("b", "a")])
        assert [f.chunk.chunk_id for f in first] == [f.chunk.chunk_id for f in second]

    def test_equal_scores_sort_by_chunk_id(self) -> None:
        fused = reciprocal_rank_fusion([ranked("zeta"), ranked("alpha", dense=True)])
        assert [f.chunk.chunk_id for f in fused] == ["alpha", "zeta"]


class TestRetriever:
    def test_requires_at_least_one_arm(self) -> None:
        with pytest.raises(ValueError, match="at least one arm"):
            RrfHybridRetriever([])

    def test_name_records_a_non_default_k(self) -> None:
        """The arm's identity must carry its parameters, or two experiment runs with
        different k are indistinguishable in the artifact."""
        assert RrfHybridRetriever([object()], k=10).name == "rrf_hybrid(k=10)"  # type: ignore[list-item]
        assert RrfHybridRetriever([object()]).name == "rrf_hybrid"  # type: ignore[list-item]
