"""Eval schema and loader validation.

The loader refuses rather than skips. Silently dropping a task changes the
denominator of every reported metric, and a metric whose n moved for an unrecorded
reason is not reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evals.loader import DatasetError, load_tasks
from evals.schema import EvalTask, ExpectedEvidence

VALID = {
    "task_id": "t-001",
    "category": "single_document_fact_lookup",
    "question": "Which state's law governs this agreement?",
    "expected_behavior": "answer",
    "dataset_version": "test-v1",
    "expected_document_ids": ["doc-900"],
    "expected_evidence": [
        {"document_id": "doc-900", "page_number": 3, "span_text": "governed by", "source": "cuad"}
    ],
}


def write_tasks(path: Path, *tasks: dict) -> Path:
    path.write_text("\n".join(json.dumps(t) for t in tasks) + "\n", encoding="utf-8")
    return path


class TestExpectedEvidence:
    def test_requires_at_least_one_locator(self) -> None:
        """A document_id alone cannot be graded at evidence level."""
        with pytest.raises(ValidationError, match="at least one of"):
            ExpectedEvidence(document_id="doc-900", source="cuad")

    def test_any_single_locator_suffices(self) -> None:
        assert ExpectedEvidence(document_id="doc-900", page_number=3, source="cuad")
        assert ExpectedEvidence(document_id="doc-900", section_id="8.1", source="cuad")
        assert ExpectedEvidence(document_id="doc-900", span_text="text", source="cuad")


class TestTaskConsistency:
    def test_valid_task_loads(self) -> None:
        assert EvalTask.model_validate(VALID).task_id == "t-001"

    def test_abstain_task_must_not_carry_expected_evidence(self) -> None:
        """If evidence exists, the correct behavior is 'answer' or 'partial'."""
        payload = {**VALID, "expected_behavior": "abstain"}
        with pytest.raises(ValidationError, match="must not carry expected evidence"):
            EvalTask.model_validate(payload)

    def test_abstain_task_with_no_evidence_is_valid(self) -> None:
        payload = {
            **VALID,
            "expected_behavior": "abstain",
            "expected_document_ids": [],
            "expected_evidence": [],
        }
        assert EvalTask.model_validate(payload).expected_behavior == "abstain"

    def test_answer_task_needs_expected_documents(self) -> None:
        payload = {**VALID, "expected_document_ids": [], "expected_evidence": []}
        with pytest.raises(ValidationError, match="requires at least one"):
            EvalTask.model_validate(payload)

    def test_evidence_cannot_name_an_unexpected_document(self) -> None:
        payload = {
            **VALID,
            "expected_evidence": [{"document_id": "doc-999", "page_number": 1, "source": "cuad"}],
        }
        with pytest.raises(ValidationError, match="absent from"):
            EvalTask.model_validate(payload)

    def test_allowlist_excluding_the_answer_is_rejected(self) -> None:
        """Otherwise the task is unpassable by construction and scores 0 forever."""
        payload = {**VALID, "allowed_document_ids": ["doc-901"]}
        with pytest.raises(ValidationError, match="unpassable by construction"):
            EvalTask.model_validate(payload)

    def test_unknown_field_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvalTask.model_validate({**VALID, "expected_anser": "typo"})


class TestPlaceholderRejection:
    @pytest.mark.parametrize(
        "question",
        [
            "TODO(author): write the question here",
            "<REPLACE WITH YOUR QUESTION>",
            "FIXME",
        ],
    )
    def test_unfilled_template_is_refused(self, question: str) -> None:
        """A number derived from a placeholder is worse than no number: it looks like data."""
        with pytest.raises(ValidationError, match="placeholder"):
            EvalTask.model_validate({**VALID, "question": question})


class TestLoader:
    def test_loads_jsonl(self, tmp_path: Path) -> None:
        path = write_tasks(tmp_path / "tasks.jsonl", VALID)
        assert len(load_tasks(path)) == 1

    def test_comments_and_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.jsonl"
        path.write_text(f"# a note\n\n{json.dumps(VALID)}\n\n", encoding="utf-8")
        assert len(load_tasks(path)) == 1

    def test_duplicate_task_id_is_refused(self, tmp_path: Path) -> None:
        path = write_tasks(tmp_path / "tasks.jsonl", VALID, VALID)
        with pytest.raises(DatasetError, match="duplicate task_id"):
            load_tasks(path)

    def test_mixed_dataset_versions_are_refused(self, tmp_path: Path) -> None:
        """A suite is versioned as a whole so results compare across runs."""
        other = {**VALID, "task_id": "t-002", "dataset_version": "test-v2"}
        path = write_tasks(tmp_path / "tasks.jsonl", VALID, other)
        with pytest.raises(DatasetError, match="mixed dataset_version"):
            load_tasks(path)

    def test_malformed_json_reports_the_line_number(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.jsonl"
        path.write_text(f"{json.dumps(VALID)}\n{{not json\n", encoding="utf-8")
        with pytest.raises(DatasetError, match=":2:"):
            load_tasks(path)

    def test_invalid_task_stops_the_run_rather_than_being_skipped(self, tmp_path: Path) -> None:
        bad = {**VALID, "task_id": "t-002", "expected_behavior": "nonsense"}
        path = write_tasks(tmp_path / "tasks.jsonl", VALID, bad)
        with pytest.raises(DatasetError, match="invalid task"):
            load_tasks(path)

    def test_empty_file_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.jsonl"
        path.write_text("# only a comment\n", encoding="utf-8")
        with pytest.raises(DatasetError, match="no tasks"):
            load_tasks(path)
