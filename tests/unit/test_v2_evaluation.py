"""Keep deterministic diagnostics separate from human semantic judgments."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agent.runner import ResearchResult
from evals.graders.v2 import ManualLabel, diagnostics, ratio
from evals.loader import load_suite
from evals.schema import EvalTask


def test_provisional_suite_is_labelled_and_old_suite_unchanged() -> None:
    tasks = load_suite("v2")
    assert len(tasks) == 24
    assert all(
        t.review_status == "provisional" and t.dataset_version.startswith("provisional-generated-")
        for t in tasks
    )
    assert len(load_suite("hand")) == 41
    assert {t.dataset_version for t in load_suite("hand")} == {"hand-v1"}
    assert {t.category for t in tasks} >= {
        "simple_lookup",
        "definition_chain",
        "version_comparison",
        "notice_deadline",
    }


def test_failure_cannot_pass_unanswerable_case() -> None:
    task = EvalTask(
        task_id="fixture",
        category="unanswerable",
        question="missing?",
        expected_behavior="abstain",
        dataset_version="provisional-generated-test",
    )
    result = ResearchResult(
        question=task.question,
        answer="",
        status="failed",
        citations=[],
        citation_errors=[],
        retrieved=[],
        evidence=[],
        trace=[],
        input_tokens=0,
        output_tokens=0,
        usage_known=False,
    )
    data = diagnostics(task, result)
    assert data["task_success"] is False
    assert data["answer_abstain_decision"] is False
    assert data["semantic_citation_support"] is None
    assert data["input_tokens"] is None
    assert data["evidence_recall_at_5"] is None


def test_manual_labels_require_reviewer_and_single_failure() -> None:
    with pytest.raises(ValidationError):
        ManualLabel(task_id="t", mode="v2", task_success=True)
    with pytest.raises(ValidationError):
        ManualLabel(task_id="t", mode="v2", task_success=False, reviewer="author")
    with pytest.raises(ValidationError):
        ManualLabel(task_id="t", mode="v2", supported_citations=2, inspected_citations=1)
    label = ManualLabel(
        task_id="t",
        mode="v2",
        task_success=False,
        primary_failure="unsupported_claim",
        reviewer="author",
    )
    assert label.primary_failure == "unsupported_claim"
    assert ratio(0, 0)["rate"] is None


def test_review_aggregator_rejects_partial_or_false_success() -> None:
    from scripts.score_v2_review import score

    report = {
        "provenance": {"reporting_eligible": False},
        "rows": [
            {"task_id": "t", "mode": mode, "category": "x", "diagnostics": {"status": "failed"}}
            for mode in ("v1", "v2")
        ],
    }
    with pytest.raises(ValueError):
        score(report, [])
    labels = [
        ManualLabel(
            task_id="t",
            mode=mode,
            task_success=False,
            reviewer="author",
            primary_failure="infrastructure_failure",
        )
        for mode in ("v1", "v2")
    ]
    result = score(report, labels)
    assert not result["reporting_eligible"]
    assert result["paired_success"]["ties"] == 1
    labels[0] = ManualLabel(task_id="t", mode="v1", task_success=True, reviewer="author")
    with pytest.raises(ValueError):
        score(report, labels)
