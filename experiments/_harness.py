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
from evals.graders.retrieval import grade_retrieval
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

    def mean(self, metric: str) -> float:
        values = [s[metric] for s in self.per_task.values() if metric in s]
        return sum(values) / len(values) if values else 0.0

    def by_category(self, metric: str) -> dict[str, tuple[float, int]]:
        buckets: dict[str, list[float]] = defaultdict(list)
        for task_id, scores in self.per_task.items():
            if metric in scores:
                buckets[self.category[task_id]].append(scores[metric])
        return {c: (sum(v) / len(v), len(v)) for c, v in sorted(buckets.items())}


def record(condition: ConditionResult, task: EvalTask, result: ResearchResult, k: int) -> None:
    grades = grade_retrieval(task, result.retrieved, k) + grade_answer(task, result)
    condition.per_task[task.task_id] = {g.name: g.score for g in grades}
    condition.category[task.task_id] = task.category
    condition.answers[task.task_id] = result.answer
    condition.statuses[task.task_id] = result.status
    condition.tokens += result.input_tokens + result.output_tokens
    condition.searches += sum(1 for e in result.trace if e.step == "search")


def paired(
    left: ConditionResult, right: ConditionResult, metric: str
) -> tuple[int, int, int, list[str]]:
    wins = losses = ties = 0
    lost: list[str] = []
    for task_id, scores in sorted(left.per_task.items()):
        if task_id not in right.per_task:
            continue
        a, b = scores.get(metric, 0.0), right.per_task[task_id].get(metric, 0.0)
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
    "citation_validity",
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
) -> Path:
    payload = {
        "experiment": experiment,
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "task_count": len(tasks),
        "trials": 1,
        "dataset_version": tasks[0].dataset_version if tasks else "unknown",
        "questions_generated": bool(
            tasks and tasks[0].dataset_version.startswith("provisional-generated")
        ),
        "conditions": {
            c.name: {
                "means": {m: c.mean(m) for m in METRICS},
                "by_category": {
                    k: {"mean": v[0], "n": v[1]}
                    for k, v in c.by_category("evidence_span_recall").items()
                },
                "tokens": c.tokens,
                "searches": c.searches,
                "per_task": c.per_task,
            }
            for c in (baseline, variant)
        },
        **extra,
    }
    out.mkdir(parents=True, exist_ok=True)
    stamp = str(payload["run_at"]).replace(":", "").replace("-", "")
    path = out / f"{experiment}-{stamp}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
