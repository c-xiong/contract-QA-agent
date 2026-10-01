"""Human labels for standalone V2 answers, and the reports built from them.

DECISION: human labels are the reported answer-quality outcome for all 41 answers. Judge
  verdicts are compared against frozen labels and reported as evaluator agreement,
  never merged into or used to overwrite labels. Label corrections after unsealing are
  explicit adjudication records, not silent edits.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evals.graders.v2 import FailureCategory, ratio

Outcome = Literal[
    "correct_answer",
    "partial_answer",
    "incorrect_answer",
    "unsupported_answer",
    "correct_abstention",
    "false_abstention",
    "execution_failure",
]
CompletenessError = Literal[
    "omitted_exception",
    "omitted_condition",
    "wrong_party",
    "incorrect_conclusion",
    "other",
]


class HumanLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    reviewer: str | None = None
    reviewer_type: Literal["author", "independent"] | None = None
    labeled_at: str | None = None
    outcome: Outcome | None = None
    task_success: bool | None = None
    required_points: list[Literal["present", "absent", "contradicted"]] | None = None
    forbidden_claims: list[Literal["made", "not_made"]] | None = None
    claims_inspected: int | None = Field(default=None, ge=0)
    claims_supported: int | None = Field(default=None, ge=0)
    claims_contradicted: int | None = Field(default=None, ge=0)
    claims_insufficient: int | None = Field(default=None, ge=0)
    completeness_errors: list[CompletenessError] = Field(default_factory=list)
    primary_failure: FailureCategory | None = None
    notes: str = ""

    @property
    def complete(self) -> bool:
        return self.task_success is not None

    @model_validator(mode="after")
    def consistent(self) -> HumanLabel:
        if self.task_success is None:
            return self
        if not (self.reviewer and self.reviewer_type and self.labeled_at and self.outcome):
            raise ValueError(
                f"{self.task_id}: a completed label needs reviewer, type, time, outcome"
            )
        if self.task_success is False and self.primary_failure is None:
            raise ValueError(f"{self.task_id}: a failed task needs one primary failure")
        if self.task_success is True and self.primary_failure is not None:
            raise ValueError(f"{self.task_id}: a successful task cannot have a primary failure")
        if self.outcome == "execution_failure" and self.task_success:
            raise ValueError(f"{self.task_id}: execution failures are never successes")
        parts = (self.claims_supported, self.claims_contradicted, self.claims_insufficient)
        if self.claims_inspected is None:
            if any(p is not None for p in parts):
                raise ValueError(f"{self.task_id}: claim counts need claims_inspected")
        elif sum(p or 0 for p in parts) != self.claims_inspected:
            raise ValueError(f"{self.task_id}: claim counts must sum to claims_inspected")
        return self


def load_labels(path: Path) -> list[HumanLabel]:
    return [
        HumanLabel.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind_source_report(labels_path: Path, report_sha256: str) -> None:
    """Record the exact report bytes used to create the review sheet."""
    source_path = labels_path.with_suffix(".source.json")
    if source_path.exists():
        raise ValueError(f"{source_path} exists; review bindings cannot be overwritten")
    source_path.write_text(json.dumps({"report_sha256": report_sha256}, indent=2) + "\n")


def freeze(labels_path: Path, report_path: Path) -> dict[str, Any]:
    report_bytes = report_path.read_bytes()
    report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    source_path = labels_path.with_suffix(".source.json")
    if not source_path.exists():
        raise ValueError(
            "Labels have no source-report binding; create a new review with `v2_labels sheet`"
        )
    source = json.loads(source_path.read_text())
    if not source.get("report_sha256"):
        raise ValueError("Unsupported legacy review binding: source-report hash is missing")
    if source["report_sha256"] != report_sha256:
        raise ValueError("Source report changed since the review sheet was created")
    report = json.loads(report_bytes)
    labels = load_labels(labels_path)
    ids = [label.task_id for label in labels]
    expected = [row["task_id"] for row in report["rows"]]
    if sorted(ids) != sorted(expected) or len(set(ids)) != len(ids):
        raise ValueError("Labels must cover every report row exactly once, including failures")
    incomplete = [label.task_id for label in labels if not label.complete]
    if incomplete:
        raise ValueError(f"Unlabelled cases: {', '.join(incomplete)}")
    rows = {row["task_id"]: row for row in report["rows"]}
    for label in labels:
        status = rows[label.task_id]["result"]["status"]
        if status in ("failed", "cancelled") and label.task_success:
            raise ValueError(f"{label.task_id}: an execution failure cannot be labelled a success")
    record = {
        "labels_sha256": sha256(labels_path),
        "report_sha256": report_sha256,
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "reviewer_types": dict(Counter(label.reviewer_type for label in labels)),
        "reviewers": sorted({label.reviewer for label in labels if label.reviewer}),
        "task_count": len(labels),
    }
    freeze_path = labels_path.with_suffix(".freeze.json")
    if freeze_path.exists():
        raise ValueError(f"{freeze_path} exists; corrections go through adjudication records")
    freeze_path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def check_frozen(labels_path: Path, report_sha256: str) -> dict[str, Any]:
    freeze_path = labels_path.with_suffix(".freeze.json")
    if not freeze_path.exists():
        raise ValueError("Labels are not frozen; run `v2_labels freeze` before unsealing verdicts")
    record: dict[str, Any] = json.loads(freeze_path.read_text())
    if not record.get("report_sha256"):
        raise ValueError("Unsupported legacy frozen labels: source-report hash is missing")
    if record["report_sha256"] != report_sha256:
        raise ValueError("Source report does not match the report used to freeze these labels")
    if sha256(labels_path) != record["labels_sha256"]:
        raise ValueError(
            "Labels changed after freezing. Restore them and record corrections in an "
            "adjudication file instead."
        )
    return record


class Adjudication(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str
    field: str
    old: Any
    new: Any
    reason: str = Field(min_length=1)
    adjudicator: str = Field(min_length=1)
    adjudicated_at: str = Field(min_length=1)


def apply_adjudications(labels: list[HumanLabel], path: Path | None) -> list[HumanLabel]:
    if path is None or not path.exists():
        return labels
    by_id = {label.task_id: label for label in labels}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = Adjudication.model_validate_json(line)
        label = by_id[record.task_id]
        if getattr(label, record.field) != record.old:
            raise ValueError(f"{record.task_id}.{record.field}: adjudication 'old' does not match")
        by_id[record.task_id] = HumanLabel.model_validate(
            label.model_dump() | {record.field: record.new}
        )
    return [by_id[label.task_id] for label in labels]


def load_sealed(path: Path, report_sha256: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    with path.open() as handle:
        header = json.loads(handle.readline())
        if not header.get("report_sha256"):
            raise ValueError("Unsupported legacy sealed verdicts: source-report hash is missing")
        if header["report_sha256"] != report_sha256:
            raise ValueError("Sealed verdicts belong to a different source report")
        records = [json.loads(line) for line in handle if line.strip()]
    return header, {r["task_id"]: r for r in records}


def agreement(
    labels: list[HumanLabel],
    verdicts: dict[str, dict[str, Any]],
    task_ids: list[str],
    *,
    seed: int = 20260928,
    agreement_sample: int = 5,
) -> dict[str, Any]:
    """Criterion-level human/judge comparison. Unknown or unjudged cases stay visible."""
    by_id = {label.task_id: label for label in labels}
    success = {"agree": 0, "disagree": 0, "false_pass": 0, "false_fail": 0}
    points = {"agree": 0, "disagree": 0}
    forbidden = {"agree": 0, "disagree": 0}
    claims = {"human_supported": 0, "judge_supported": 0, "human_inspected": 0, "judge_graded": 0}
    unknown: list[str] = []
    not_judged: list[str] = []
    disagreements: list[dict[str, Any]] = []
    agreements: list[str] = []
    for task_id in task_ids:
        label, verdict = by_id[task_id], verdicts.get(task_id)
        if verdict is None or verdict.get("not_judged"):
            not_judged.append(task_id)
            continue
        rubric = verdict["answer_rubric"]
        judged_claims = [c for c in verdict.get("claims", []) if c["verdict"] != "unknown"]
        claims["judge_graded"] += len(judged_claims)
        claims["judge_supported"] += sum(c["verdict"] == "supported" for c in judged_claims)
        claims["human_inspected"] += label.claims_inspected or 0
        claims["human_supported"] += label.claims_supported or 0
        if rubric["verdict"] != "graded":
            unknown.append(task_id)
            continue
        judge_success = rubric["task_success"]
        mismatched: list[str] = []
        if judge_success == label.task_success:
            success["agree"] += 1
        else:
            success["disagree"] += 1
            success["false_pass" if judge_success else "false_fail"] += 1
            mismatched.append("task_success")
        for field, bucket, key in (
            ("required_points", points, "required_points"),
            ("forbidden_claims", forbidden, "forbidden_claims"),
        ):
            human = getattr(label, field)
            judged = [v["verdict"] for v in sorted(rubric[key], key=lambda v: v["index"])]
            if human is None:
                continue
            for h, j in zip(human, judged, strict=True):
                bucket["agree" if h == j else "disagree"] += 1
            if list(human) != judged:
                mismatched.append(field)
        if mismatched:
            disagreements.append({"task_id": task_id, "fields": mismatched})
        else:
            agreements.append(task_id)
    rng = random.Random(seed)
    sample = sorted(rng.sample(agreements, min(agreement_sample, len(agreements))))
    graded = success["agree"] + success["disagree"]
    return {
        "n_cases": len(task_ids),
        "graded_by_judge": graded,
        "judge_unknown": unknown,
        "not_judged": not_judged,
        "task_success_agreement": ratio(success["agree"], graded),
        "false_pass": ratio(success["false_pass"], graded),
        "false_fail": ratio(success["false_fail"], graded),
        "required_point_agreement": ratio(points["agree"], points["agree"] + points["disagree"]),
        "forbidden_claim_agreement": ratio(
            forbidden["agree"], forbidden["agree"] + forbidden["disagree"]
        ),
        "claim_support_totals": claims,
        "disagreements_to_review": disagreements,
        "agreement_sample_to_review": {"seed": seed, "task_ids": sample},
    }


def quality(labels: list[HumanLabel], report: dict[str, Any]) -> dict[str, Any]:
    """Answer quality from human labels only; every rate has its denominator."""
    rows = {row["task_id"]: row for row in report["rows"]}

    def group(selected: list[HumanLabel]) -> dict[str, Any]:
        answerable = [
            label for label in selected if rows[label.task_id]["diagnostics"]["answerable"]
        ]
        inspected = sum(label.claims_inspected or 0 for label in selected)
        return {
            "n": len(selected),
            "task_success": ratio(
                sum(label.task_success is True for label in selected), len(selected)
            ),
            "outcomes": dict(Counter(label.outcome for label in selected)),
            "false_abstention": ratio(
                sum(label.outcome == "false_abstention" for label in answerable), len(answerable)
            ),
            "unsupported_answer": ratio(
                sum(label.outcome == "unsupported_answer" for label in selected), len(selected)
            ),
            "execution_failure": ratio(
                sum(label.outcome == "execution_failure" for label in selected), len(selected)
            ),
            "claims_supported": ratio(
                sum(label.claims_supported or 0 for label in selected), inspected
            ),
            "claims_contradicted": ratio(
                sum(label.claims_contradicted or 0 for label in selected), inspected
            ),
            "completeness_errors": dict(
                Counter(e for label in selected for e in label.completeness_errors)
            ),
            "primary_failures": dict(
                Counter(label.primary_failure for label in selected if label.primary_failure)
            ),
        }

    subsets: dict[str, Any] = {"all": group(labels)}
    for category in sorted({rows[label.task_id]["category"] for label in labels}):
        subsets[f"category:{category}"] = group(
            [label for label in labels if rows[label.task_id]["category"] == category]
        )
    return subsets
