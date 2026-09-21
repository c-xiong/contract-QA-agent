"""Deterministic V2 diagnostics; semantic outcomes require independent human labels."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.agent.runner import ResearchResult
from evals.graders.retrieval import evidence_span_recall
from evals.schema import EvalTask

VERSION = "v2-diagnostics@1"
FailureCategory = Literal[
    "retrieval_miss",
    "wrong_tool_or_arguments",
    "missing_definition",
    "version_or_precedence_confusion",
    "incomplete_evidence",
    "calculation_error",
    "unsupported_claim",
    "invalid_citation",
    "unnecessary_loop",
    "failed_repair",
    "wrong_abstention",
    "scope_or_injection_violation",
    "infrastructure_failure",
]


class ManualLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str
    mode: Literal["v1", "v2"]
    task_success: bool | None = None
    supported_citations: int | None = Field(default=None, ge=0)
    inspected_citations: int | None = Field(default=None, ge=0)
    useful_calls: int | None = Field(default=None, ge=0)
    inspected_calls: int | None = Field(default=None, ge=0)
    primary_failure: FailureCategory | None = None
    reviewer: str | None = None
    notes: str = ""

    @model_validator(mode="after")
    def consistent(self) -> ManualLabel:
        for hits, count in (
            (self.supported_citations, self.inspected_citations),
            (self.useful_calls, self.inspected_calls),
        ):
            if (hits is None) != (count is None) or (
                hits is not None and count is not None and hits > count
            ):
                raise ValueError("Counts require a denominator and cannot exceed it")
        if self.task_success is not None and not self.reviewer:
            raise ValueError("Completed judgments must identify a reviewer")
        if self.task_success is False and self.primary_failure is None:
            raise ValueError("Assign exactly one primary cause to each failed task")
        if self.task_success is True and self.primary_failure is not None:
            raise ValueError("Successful tasks cannot have a primary failure")
        return self


def ratio(hits: int, total: int) -> dict[str, int | float | None]:
    return {"count": hits, "denominator": total, "rate": hits / total if total else None}


def diagnostics(
    task: EvalTask, result: ResearchResult, snapshot: dict[str, Any] | None = None
) -> dict[str, Any]:
    state = snapshot or {}
    tools = list(state.get("tool_results_by_call_id", {}).values())
    names = [t["tool"] for t in tools if t["status"] == "ok"]
    coverage = set(task.required_capabilities) & set(names)
    order_ok = all(
        a in names and b in names and names.index(a) < names.index(b)
        for a, b in task.capability_order
    )
    delivered = result.citations if result.status == "completed" else []
    answerable = task.expected_behavior != "abstain"
    if result.status in ("failed", "cancelled"):
        decision_correct = False
    else:
        decision_correct = result.status == ("completed" if answerable else "abstained")
    return {
        "grader_version": VERSION,
        "answerable": answerable,
        "status": result.status,
        "answer_abstain_decision": decision_correct,
        "false_abstention": answerable and result.status == "abstained",
        "citation_validity": ratio(
            len(delivered) if not result.citation_errors else 0,
            len(delivered),
        ),
        "structural_gate_pass": bool(state.get("verification_ok"))
        if snapshot
        else result.status == "completed" and bool(result.citations) and not result.citation_errors,
        "semantic_citation_support": None,
        "task_success": False if result.status in ("failed", "cancelled") else None,
        "evidence_recall_at_5": evidence_span_recall(task, result.retrieved, 5).score
        if answerable
        else None,
        "trajectory_evidence_recall": evidence_span_recall(
            task, result.retrieved, len(result.retrieved)
        ).score
        if answerable
        else None,
        "required_capability_coverage": ratio(len(coverage), len(task.required_capabilities))
        if snapshot
        else None,
        "capability_order_satisfied": order_ok if snapshot and task.capability_order else None,
        "tool_usefulness": None,
        "derived_date_matches": any(
            c.get("date") == task.expected_date for c in state.get("calculations", [])
        )
        if task.expected_date and snapshot
        else None,
        "retries": state.get("retries", 0),
        "tool_calls": result.tool_calls if snapshot else None,
        "unnecessary_calls": None,  # Human labels distinguish justified retries from waste.
        "model_calls": result.model_calls,
        "input_tokens": result.input_tokens if result.usage_known else None,
        "output_tokens": result.output_tokens if result.usage_known else None,
        "elapsed_seconds": result.elapsed_seconds,
    }
