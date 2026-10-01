"""Offline answer-rubric judge: criterion-level verdicts against a frozen task rubric.

DECISION: the judge sees the question, reviewed reference evidence, the rubric and the
  delivered answer, so it can detect facts the system never retrieved. It never sees the
  runtime verifier's verdict, the system's internal reasoning or a system/version label.
  Its prompt is independent of every runtime prompt.

DECISION: verdicts are per criterion (each required point, each forbidden claim, the
  answer/abstain behaviour), not an unexplained 1-10 score. Any output that does not
  cover every criterion is `unknown` and routes to human review; it is never a pass.

This prompt is provisional until calibrated against human labels. Its verdicts are
evaluator measurements, never the reported answer-quality outcome.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from app.agent.llm import ModelClient, ModelError
from evals.schema import EvalTask

ANSWER_RUBRIC_VERSION = "answer_rubric@1-provisional"
# TODO(author): the grading rules and pass criteria in ANSWER_RUBRIC_SYSTEM were drafted
# with coding-agent assistance. .claude/rules/evals.md forbids generated pass criteria in
# reporting. Review or rewrite them, bump ANSWER_RUBRIC_VERSION (drop "-provisional"),
# then set this flag. Until then the frozen judge pass refuses to run; calibration rounds
# may run as pipeline checks only.
ANSWER_RUBRIC_AUTHOR_REVIEWED = False

ANSWER_RUBRIC_SYSTEM = """You grade one answer to a contract question against a fixed rubric.

You receive the question, the expected behaviour (answer, abstain or partial), numbered
required points, numbered forbidden claims, reference excerpts from the contract, and the
answer under review. Judge only from these inputs; do not use outside knowledge.

Return JSON only:
{"required_points": [{"index": 0, "verdict": "present" | "absent" | "contradicted"}],
 "forbidden_claims": [{"index": 0, "verdict": "made" | "not_made"}],
 "behavior": "answered" | "abstained" | "partial",
 "conclusion_correct": true | false | null,
 "task_success": true | false,
 "reason": "<one sentence>"}

Rules:
- Include every required point and every forbidden claim index exactly once.
- A point is "present" only if the answer states it or plainly entails it. Paraphrase is
  fine; a vaguer or weaker statement is "absent".
- "conclusion_correct" is null when the task expects abstention.
- If the task expects abstention, task_success is true only if the answer declines to
  answer and asserts no unsupported contract facts.
- Otherwise task_success is true only if every required point is present, no forbidden
  claim is made, and no required point is contradicted. Omitted exceptions or conditions
  in the reference excerpts that change the conclusion make task_success false."""


class PointVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int
    verdict: Literal["present", "absent", "contradicted"]


class ForbiddenVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int
    verdict: Literal["made", "not_made"]


class RubricVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    required_points: list[PointVerdict]
    forbidden_claims: list[ForbiddenVerdict]
    behavior: Literal["answered", "abstained", "partial"]
    conclusion_correct: bool | None
    task_success: bool
    reason: str


def rubric_prompt(task: EvalTask, answer: str) -> str:
    references = "\n".join(
        f"- [{e.document_id}"
        + (f", p. {e.page_number}" if e.page_number is not None else "")
        + (f", §{e.section_id}" if e.section_id else "")
        + f"] {e.span_text}"
        for e in task.expected_evidence
        if e.span_text
    )
    points = "\n".join(f"{i}. {p}" for i, p in enumerate(task.required_points)) or "(none)"
    forbidden = "\n".join(f"{i}. {c}" for i, c in enumerate(task.forbidden_claims)) or "(none)"
    return (
        f"Question:\n{task.question}\n\nExpected behaviour: {task.expected_behavior}\n\n"
        f"Required points:\n{points}\n\nForbidden claims:\n{forbidden}\n\n"
        f"Reference excerpts:\n{references or '(none)'}\n\nAnswer under review:\n{answer}"
    )


def _extract_json(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if 0 <= start < end else text


async def judge_answer(task: EvalTask, answer: str, client: ModelClient) -> dict[str, Any]:
    """Return a criterion-level verdict, or `{"verdict": "unknown", ...}` for human review."""
    try:
        response = await client.complete(ANSWER_RUBRIC_SYSTEM, rubric_prompt(task, answer))
    except ModelError as exc:
        return {"verdict": "unknown", "reason": f"judge call failed: {type(exc).__name__}"}
    try:
        parsed = RubricVerdict.model_validate(json.loads(_extract_json(response.text)))
    except (json.JSONDecodeError, ValidationError) as exc:
        return {"verdict": "unknown", "reason": f"unparseable judge output: {type(exc).__name__}"}
    if sorted(p.index for p in parsed.required_points) != list(
        range(len(task.required_points))
    ) or sorted(c.index for c in parsed.forbidden_claims) != list(
        range(len(task.forbidden_claims))
    ):
        return {"verdict": "unknown", "reason": "judge did not cover every rubric criterion"}
    return {"verdict": "graded", **parsed.model_dump()}
