"""Retrieval graders. Pure functions over (task, result), tested against fixtures."""

from __future__ import annotations

from evals.graders.retrieval import (
    document_recall,
    evidence_span_recall,
    grade_retrieval,
    reciprocal_rank,
)
from evals.schema import EvalTask
from tests.conftest import make_chunk, make_retrieved

CAP_TEXT = (
    "8.1 Limitation of Liability. Except as set out in Section 8.3, the aggregate "
    "liability of either party shall not exceed the fees paid in the twelve months "
    "preceding the claim."
)


def task(**overrides: object) -> EvalTask:
    payload: dict[str, object] = {
        "task_id": "t-001",
        "category": "single_document_fact_lookup",
        "question": "What is the liability cap?",
        "expected_behavior": "answer",
        "dataset_version": "test-v1",
        "expected_document_ids": ["doc-900"],
        "expected_evidence": [
            {
                "document_id": "doc-900",
                "page_number": 1,
                "span_text": CAP_TEXT,
                "source": "cuad",
            }
        ],
    }
    payload.update(overrides)
    return EvalTask.model_validate(payload)


def hits(*specs: tuple[str, str]) -> list:
    return [
        make_retrieved(make_chunk(f"{doc}-c{i}", doc, text=text), rank=i)
        for i, (doc, text) in enumerate(specs, start=1)
    ]


class TestDocumentRecall:
    def test_full_recall(self) -> None:
        result = document_recall(task(), hits(("doc-900", CAP_TEXT)), k=5)
        assert result.score == 1.0 and result.passed

    def test_missing_document_scores_zero_and_names_it(self) -> None:
        result = document_recall(task(), hits(("doc-901", "unrelated")), k=5)
        assert result.score == 0.0
        assert result.data["missing"] == ["doc-900"]

    def test_k_bounds_what_counts_as_retrieved(self) -> None:
        """k counts chunks the agent actually saw, not distinct documents."""
        retrieved = hits(("doc-901", "a"), ("doc-901", "b"), ("doc-900", CAP_TEXT))
        assert document_recall(task(), retrieved, k=2).score == 0.0
        assert document_recall(task(), retrieved, k=3).score == 1.0

    def test_partial_recall_across_multiple_expected_documents(self) -> None:
        multi = task(expected_document_ids=["doc-900", "doc-901"], expected_evidence=[])
        result = document_recall(multi, hits(("doc-900", CAP_TEXT)), k=5)
        assert result.score == 0.5
        assert not result.passed

    def test_abstain_task_is_a_pass_with_an_explicit_note(self) -> None:
        """Scoring 0 would punish correct behavior; skipping would shrink n silently."""
        abstain = task(expected_behavior="abstain", expected_document_ids=[], expected_evidence=[])
        result = document_recall(abstain, [], k=5)
        assert result.passed and result.score == 1.0
        assert "abstain" in result.detail


class TestEvidenceSpanRecall:
    def test_finds_the_expected_span(self) -> None:
        result = evidence_span_recall(task(), hits(("doc-900", CAP_TEXT)), k=5)
        assert result.score == 1.0

    def test_right_document_wrong_clause_scores_zero(self) -> None:
        """The failure a document-level metric cannot see, and a page citation would
        make look authoritative."""
        wrong = "14.2 Governing Law. This Agreement is governed by the laws of Delaware."
        assert document_recall(task(), hits(("doc-900", wrong)), k=5).score == 1.0
        assert evidence_span_recall(task(), hits(("doc-900", wrong)), k=5).score == 0.0

    def test_matching_survives_whitespace_and_typographic_differences(self) -> None:
        """CUAD annotations are ASCII; PDF extraction is not."""
        reflowed = CAP_TEXT.replace("Liability.", "Liability.\n").replace(" the ", "  the ")
        assert evidence_span_recall(task(), hits(("doc-900", reflowed)), k=5).score == 1.0

    def test_span_split_across_chunks_still_matches_via_the_probe(self) -> None:
        """Whole-span containment would score 0 here, penalizing our own chunking."""
        first_half = CAP_TEXT[:90]
        assert evidence_span_recall(task(), hits(("doc-900", first_half)), k=5).score == 1.0

    def test_no_expected_spans_is_not_applicable(self) -> None:
        result = evidence_span_recall(task(expected_evidence=[]), [], k=5)
        assert result.passed and "not applicable" in result.detail


class TestReciprocalRank:
    def test_rank_one(self) -> None:
        assert reciprocal_rank(task(), hits(("doc-900", CAP_TEXT))).score == 1.0

    def test_rank_three(self) -> None:
        retrieved = hits(("doc-901", "a"), ("doc-901", "b"), ("doc-900", CAP_TEXT))
        result = reciprocal_rank(task(), retrieved)
        assert result.score == 1 / 3
        assert result.data["rank"] == 3

    def test_absent_scores_zero(self) -> None:
        assert reciprocal_rank(task(), hits(("doc-901", "a"))).score == 0.0

    def test_nothing_retrieved_scores_zero(self) -> None:
        assert reciprocal_rank(task(), []).score == 0.0


class TestGradeRetrieval:
    def test_runs_every_grader_and_versions_each(self) -> None:
        results = grade_retrieval(task(), hits(("doc-900", CAP_TEXT)), k=5)
        assert {r.name for r in results} == {
            "document_recall",
            "evidence_span_recall",
            "reciprocal_rank",
        }
        assert all(r.version for r in results), "every grader is versioned"

    def test_graders_are_pure_and_repeatable(self) -> None:
        retrieved = hits(("doc-900", CAP_TEXT))
        first = [r.score for r in grade_retrieval(task(), retrieved, k=5)]
        second = [r.score for r in grade_retrieval(task(), retrieved, k=5)]
        assert first == second
