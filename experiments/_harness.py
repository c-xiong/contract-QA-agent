"""Shared reporting for experiments B and C.

Both compare two configurations of the same system on the same tasks, so both need the
same statistical discipline: paired counts, per-category breakdown, inspected failures,
and n in every caption (SPEC 15.7, .claude/rules/evals.md).
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from app.agent.runner import ResearchResult
from evals.graders.answer import grade_answer
from evals.graders.retrieval import GradeResult, grade_retrieval
from evals.schema import EvalTask


@dataclass
class ConditionResult:
    """One configuration's scores across a suite."""

    name: str
    per_task: dict[str, dict[str, float]] = field(default_factory=dict)
    category: dict[str, str] = field(default_factory=dict)
    answers: dict[str, str] = field(default_factory=dict)
    statuses: dict[str, str] = field(default_factory=dict)
    tokens: int = 0
    searches: int = 0
    model_calls: int = 0
    elapsed_seconds: float = 0.0
    grader_versions: dict[str, str] = field(default_factory=dict)
    grade_data: dict[str, dict[str, dict[str, object]]] = field(default_factory=dict)
    costs: dict[str, dict[str, int | float]] = field(default_factory=dict)

    def mean(self, metric: str) -> float:
        values = [s[metric] for s in self.per_task.values() if metric in s]
        return sum(values) / len(values) if values else 0.0

    def by_category(self, metric: str) -> dict[str, tuple[float, int]]:
        buckets: dict[str, list[float]] = defaultdict(list)
        for task_id, scores in self.per_task.items():
            if metric in scores:
                buckets[self.category[task_id]].append(scores[metric])
        return {c: (sum(v) / len(v), len(v)) for c, v in sorted(buckets.items())}


def record(
    condition: ConditionResult,
    task: EvalTask,
    result: ResearchResult,
    k: int,
    *,
    excluded_metrics: frozenset[str] = frozenset(),
    elapsed_seconds: float = 0.0,
    extra_grades: list[GradeResult] | None = None,
) -> None:
    grades = grade_retrieval(task, result.retrieved, k) + grade_answer(task, result)
    grades += extra_grades or []
    condition.per_task[task.task_id] = {
        grade.name: grade.score for grade in grades if grade.name not in excluded_metrics
    }
    condition.category[task.task_id] = task.category
    condition.answers[task.task_id] = result.answer
    condition.statuses[task.task_id] = result.status
    condition.tokens += result.input_tokens + result.output_tokens
    condition.searches += sum(1 for e in result.trace if e.step == "search")
    condition.grader_versions.update(
        {g.name: g.version for g in grades if g.name not in excluded_metrics}
    )
    condition.grade_data[task.task_id] = {
        g.name: g.data for g in grades if g.name not in excluded_metrics
    }
    calls = getattr(result, "model_calls", sum(e.step == "write" for e in result.trace))
    condition.model_calls += calls
    condition.elapsed_seconds += elapsed_seconds
    condition.costs[task.task_id] = {
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "model_calls": calls,
        "elapsed_seconds": elapsed_seconds,
    }


def paired(
    left: ConditionResult, right: ConditionResult, metric: str
) -> tuple[int, int, int, list[str]]:
    wins = losses = ties = 0
    lost: list[str] = []
    for task_id, scores in sorted(left.per_task.items()):
        if task_id not in right.per_task:
            continue
        if metric not in scores or metric not in right.per_task[task_id]:
            continue
        a, b = scores[metric], right.per_task[task_id][metric]
        if a > b:
            wins += 1
        elif a < b:
            losses += 1
            lost.append(task_id)
        else:
            ties += 1
    return wins, losses, ties, lost


METRICS = (
    "document_recall",
    "evidence_span_recall",
    "abstention_correctness",
    "forbidden_claims",
    "required_points",
    "post_gate_citation_validity",
    "citation_coverage",
    "claim_support",
)


def print_report(
    title: str,
    baseline: ConditionResult,
    variant: ConditionResult,
    tasks: list[EvalTask],
    *,
    k: int,
    generated: bool,
    notes: list[str],
) -> None:
    print("\n" + "=" * 78)
    print(f"AGGREGATE  (n={len(tasks)} tasks, 1 trial, k={k})")
    print("=" * 78)
    width = max(len(baseline.name), len(variant.name)) + 2
    print(f"{'metric':<26}{baseline.name:>{width}}{variant.name:>{width}}{'delta':>10}")
    for metric in METRICS:
        baseline_has = any(metric in scores for scores in baseline.per_task.values())
        variant_has = any(metric in scores for scores in variant.per_task.values())
        if not baseline_has or not variant_has:
            left_cell = f"{baseline.mean(metric):.3f}" if baseline_has else "not applicable"
            right_cell = f"{variant.mean(metric):.3f}" if variant_has else "not applicable"
            print(f"{metric:<26}{left_cell:>{width}}{right_cell:>{width}}{'':>10}")
            continue
        a, b = baseline.mean(metric), variant.mean(metric)
        print(f"{metric:<26}{a:>{width}.3f}{b:>{width}.3f}{b - a:>+10.3f}")
    print(f"{'total tokens':<26}{baseline.tokens:>{width}}{variant.tokens:>{width}}")
    print(f"{'total searches':<26}{baseline.searches:>{width}}{variant.searches:>{width}}")

    print("\n" + "=" * 78)
    print("PAIRED COMPARISONS -- same tasks, variant vs baseline")
    print("=" * 78)
    for metric in METRICS:
        wins, losses, ties, lost = paired(variant, baseline, metric)
        if wins == 0 and losses == 0:
            print(f"{metric:<26} all {ties} tasks tied")
            continue
        print(f"{metric:<26} win {wins:2d}  loss {losses:2d}  tie {ties:2d}")
        if lost:
            print(f"{'':<26}   lost on: {', '.join(lost[:6])}")

    print("\n" + "=" * 78)
    print("PER CATEGORY -- evidence_span_recall  (mean, n)")
    print("=" * 78)
    left, right = (
        baseline.by_category("evidence_span_recall"),
        variant.by_category("evidence_span_recall"),
    )
    print(f"{'category':<34}{baseline.name:>20}{variant.name:>20}")
    for cat in sorted(set(left) | set(right)):
        la = f"{left[cat][0]:.2f} (n={left[cat][1]})" if cat in left else "-"
        ra = f"{right[cat][0]:.2f} (n={right[cat][1]})" if cat in right else "-"
        print(f"{cat:<34}{la:>20}{ra:>20}")

    print("\n" + "=" * 78)
    print("FAILURE INSPECTION")
    print("=" * 78)
    task_by_id = {t.task_id: t for t in tasks}
    shown = 0
    for task_id in sorted(variant.per_task):
        if shown >= 3:
            break
        scores = variant.per_task[task_id]
        weak = [m for m in METRICS if scores.get(m, 1.0) < 1.0]
        if not weak:
            continue
        task = task_by_id[task_id]
        print(f"  {task_id}  [{task.category}]  weak on: {', '.join(weak)}")
        print(f"    question : {task.question[:68]}")
        print(f"    expected : {task.expected_behavior}, got {variant.statuses[task_id]}")
        print(f"    answer   : {' '.join(variant.answers[task_id].split())[:100]}")
        print()
        shown += 1
    if shown == 0:
        print("  No task scored below 1.0 on any metric in the variant condition.\n")

    print("=" * 78)
    print("LIMITATIONS")
    print("=" * 78)
    print(f"  - n = {len(tasks)}. A one- or two-task difference is noise. Directional only;")
    print("    no significance claim is supported at this sample size.")
    for note in notes:
        print(f"  - {note}")
    if generated:
        print("  - Questions are model-generated: a pipeline check, not a capability")
        print("    measurement. See CLAUDE.md rule 1.")


def write_artifact(
    out: Path,
    experiment: str,
    baseline: ConditionResult,
    variant: ConditionResult,
    tasks: list[EvalTask],
    extra: dict[str, object],
    *,
    provenance: dict[str, object] | None = None,
    output: Path | None = None,
) -> Path:
    return write_conditions_artifact(
        out, experiment, [baseline, variant], tasks, extra, provenance=provenance, output=output
    )


def write_conditions_artifact(
    out: Path,
    experiment: str,
    conditions: list[ConditionResult],
    tasks: list[EvalTask],
    extra: dict[str, object],
    *,
    provenance: dict[str, object] | None = None,
    output: Path | None = None,
) -> Path:
    versions = {k: v for c in conditions for k, v in c.grader_versions.items()}
    payload = {
        **(provenance or {}),
        "experiment": experiment,
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "task_count": len(tasks),
        "trials": 1,
        "grader_versions": versions,
        "dataset_version": tasks[0].dataset_version if tasks else "unknown",
        "questions_generated": bool(
            tasks and tasks[0].dataset_version.startswith("provisional-generated")
        ),
        "conditions": {
            c.name: {
                "means": {
                    metric: c.mean(metric)
                    for metric in METRICS
                    if any(metric in scores for scores in c.per_task.values())
                },
                "by_category": {
                    k: {"mean": v[0], "n": v[1]}
                    for k, v in c.by_category("evidence_span_recall").items()
                },
                "tokens": c.tokens,
                "searches": c.searches,
                "model_calls": c.model_calls,
                "elapsed_seconds": c.elapsed_seconds,
                "grader_versions": c.grader_versions,
                "per_task": c.per_task,
                "answers": c.answers,
                "statuses": c.statuses,
                "grade_data": c.grade_data,
                "costs": c.costs,
            }
            for c in conditions
        },
        **extra,
    }
    stamp = str(payload["run_at"]).replace(":", "").replace("-", "")
    suffix = str(payload.get("run_id", ""))[:8]
    path = output or out / f"{experiment}-{stamp}{'-' + suffix if suffix else ''}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def limited_tasks(tasks: list[EvalTask], limit: int | None) -> list[EvalTask]:
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    return tasks[:limit] if limit is not None else tasks


def paired_summary(baseline: ConditionResult, variant: ConditionResult) -> dict[str, object]:
    summary: dict[str, object] = {}
    for metric in METRICS:
        wins, losses, ties, lost = paired(variant, baseline, metric)
        if wins + losses + ties:
            summary[metric] = {
                "delta": variant.mean(metric) - baseline.mean(metric),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "lost_on": lost,
            }
    return summary
