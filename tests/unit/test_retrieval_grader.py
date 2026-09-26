"""Independent evidence diagnostics reject wrong documents and partial clauses."""

import pytest

from evals.graders.retrieval import (
    document_aware_evidence_recall,
    evidence_complete_span_recall,
    evidence_fragment_recall,
    evidence_span_recall,
)
from tests.unit.test_graders import CAP_TEXT, hits, task

GRADERS = (
    document_aware_evidence_recall,
    evidence_fragment_recall,
    evidence_complete_span_recall,
)


def test_identical_text_from_wrong_document_is_not_evidence() -> None:
    retrieved = hits(("doc-901", CAP_TEXT))
    assert evidence_span_recall(task(), retrieved, k=5).score == 1.0  # legacy unchanged
    assert all(grader(task(), retrieved, k=5).score == 0.0 for grader in GRADERS)


def test_prefix_match_does_not_imply_complete_clause() -> None:
    retrieved = hits(("doc-900", CAP_TEXT[:90]))
    assert document_aware_evidence_recall(task(), retrieved, k=5).score == 1.0
    assert evidence_fragment_recall(task(), retrieved, k=5).score == 0.0
    assert evidence_complete_span_recall(task(), retrieved, k=5).score == 0.0


def test_explicit_omissions_require_every_fragment_for_complete_span() -> None:
    first = "The recipient must return confidential documents."
    second = "Copies retained for regulatory compliance remain confidential."
    annotated = task(
        expected_evidence=[
            {
                "document_id": "doc-900",
                "span_text": first + "<omitted>" + second,
                "source": "cuad",
            }
        ]
    )
    partial = hits(("doc-900", first), ("doc-901", second))
    assert evidence_fragment_recall(annotated, partial, k=5).score == 0.5
    assert evidence_complete_span_recall(annotated, partial, k=5).score == 0.0
    complete = hits(("doc-900", first), ("doc-900", second))
    assert evidence_fragment_recall(annotated, complete, k=5).score == 1.0
    assert evidence_complete_span_recall(annotated, complete, k=5).score == 1.0


@pytest.mark.parametrize("grader", GRADERS)
def test_normalization_and_top_k_are_consistent(grader) -> None:
    retrieved = hits(("doc-900", "irrelevant"), ("doc-900", CAP_TEXT.upper().replace(" ", "\n")))
    assert grader(task(), retrieved, k=1).score == 0.0
    assert grader(task(), retrieved, k=2).score == 1.0
    assert grader(task(), retrieved, k=0).score == 0.0
    with pytest.raises(ValueError, match="nonnegative"):
        grader(task(), retrieved, k=-1)


@pytest.mark.parametrize("grader", GRADERS)
def test_absent_evidence_is_explicitly_not_applicable(grader) -> None:
    result = grader(task(expected_evidence=[]), [], k=5)
    assert result.score == 1.0
    assert "not applicable" in result.detail


def test_complete_span_recall_keeps_individual_annotation_denominator() -> None:
    annotated = task(
        expected_evidence=[
            {"document_id": "doc-900", "span_text": "First clause.", "source": "manual"},
            {
                "document_id": "doc-900",
                "span_text": "Second clause.<omitted>Third clause.",
                "source": "manual",
            },
        ]
    )
    retrieved = hits(("doc-900", "First clause. Second clause."))
    assert evidence_fragment_recall(annotated, retrieved, k=5).score == 2 / 3
    assert evidence_complete_span_recall(annotated, retrieved, k=5).score == 1 / 2
