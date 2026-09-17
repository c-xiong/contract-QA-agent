"""Experiment B: isolate the refinement loop and cross-reference resolution (2 x 2).

CRA_LIVE_MODEL=0 uv run python experiments/agentic_comparison.py --suite hand --limit 2
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

from app.agent.runner import ResearchAgent
from app.agent.state import Budget
from app.config import get_settings
from app.ingestion.store import ChunkStore, StoreError
from app.retrieval.factory import build_retriever
from evals.graders.claim_support import claim_support
from evals.loader import DatasetError, load_suite
from evals.provenance import capture_provenance
from experiments._harness import (
    METRICS,
    ConditionResult,
    limited_tasks,
    paired_summary,
    record,
    write_conditions_artifact,
)


def factorial_budgets(top_k: int) -> dict[str, Budget]:
    # DECISION: vary only loop and xref budgets. A two-arm comparison changes both
    # and cannot attribute a gain; all other limits and the graph remain identical.
    base = Budget(max_chunks_per_query=top_k)
    return {
        f"{loop}_{xref}": replace(
            base,
            max_searches=base.max_searches if loop == "loop" else 1,
            max_research_iterations=base.max_research_iterations if loop == "loop" else 1,
            max_cross_reference_depth=base.max_cross_reference_depth if xref == "xref" else 0,
            max_cross_reference_chunks=base.max_cross_reference_chunks if xref == "xref" else 0,
        )
        for loop in ("single", "loop")
        for xref in ("no_xref", "xref")
    }


def factorial_effects(conditions: dict[str, ConditionResult]) -> dict[str, dict[str, object]]:
    """Paired main effects average over the other factor; interaction is diff-in-diff."""
    effects: dict[str, dict[str, object]] = {}
    names = ("single_no_xref", "single_xref", "loop_no_xref", "loop_xref")
    rows = [conditions[name] for name in names]
    common_tasks = set.intersection(*(set(row.per_task) for row in rows))
    for metric in METRICS:
        per_task: dict[str, dict[str, float]] = {}
        for task_id in sorted(common_tasks):
            if not all(metric in row.per_task[task_id] for row in rows):
                continue
            a, b, c, d = (row.per_task[task_id][metric] for row in rows)
            per_task[task_id] = {
                "loop": ((c - a) + (d - b)) / 2,
                "xref": ((b - a) + (d - c)) / 2,
                "interaction": (d - b) - (c - a),
            }
        if per_task:
            effects[metric] = {
                "n": len(per_task),
                "means": {
                    factor: sum(r[factor] for r in per_task.values()) / len(per_task)
                    for factor in ("loop", "xref", "interaction")
                },
                "per_task": per_task,
            }
    return effects


async def main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    try:
        tasks = limited_tasks(load_suite(args.suite), args.limit)
        store = ChunkStore.load(settings.processed_dir)
    except (DatasetError, StoreError, ValueError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    retriever = build_retriever(args.arm, store, settings.index_dir)
    agent = ResearchAgent(store, settings, retriever=retriever, answerability_gate=False)
    budgets = factorial_budgets(settings.retrieval_top_k)
    provenance = capture_provenance(
        store,
        dataset_name=args.suite,
        dataset_version=tasks[0].dataset_version,
        retrieval_arm=retriever.name,
        top_k=settings.retrieval_top_k,
        model_id=agent.client.model_id,
    )
    print(f"Experiment B: 4 arms x {len(tasks)} tasks, 1 trial; model={agent.client.model_id}")
    print(
        f"Projected research calls: {4 * len(tasks)}; writer calls <= {sum(1 + b.max_repair_attempts for b in budgets.values()) * len(tasks)}"
    )
    if args.claim_support:
        print(
            "Additional offline grading: up to one model call per factual claim (output-dependent)."
        )
    conditions = {name: ConditionResult(name=name) for name in budgets}

    async def run_condition(name: str, budget: Budget) -> None:
        print(f"running {name}...", flush=True)
        for task in tasks:
            start = perf_counter()
            result = await agent.research(
                task.question,
                allowed_document_ids=task.allowed_document_ids,
                budget=budget,
            )
            elapsed = perf_counter() - start
            grades = (
                [await claim_support(task, result, agent.client, live=settings.live_model)]
                if args.claim_support
                else []
            )
            record(
                conditions[name],
                task,
                result,
                settings.retrieval_top_k,
                elapsed_seconds=elapsed,
                extra_grades=grades,
            )

    items = list(budgets.items())
    for start in range(0, len(items), 2):
        await asyncio.gather(
            *(run_condition(name, budget) for name, budget in items[start : start + 2])
        )
    contrasts = {
        label: paired_summary(conditions[left], conditions[right])
        for label, left, right in (
            ("loop_without_xref", "single_no_xref", "loop_no_xref"),
            ("loop_with_xref", "single_xref", "loop_xref"),
            ("xref_without_loop", "single_no_xref", "single_xref"),
            ("xref_with_loop", "loop_no_xref", "loop_xref"),
        )
    }
    effects = factorial_effects(conditions)
    print("Paired effects (one trial; directional, no significance claim):")
    for metric, effect in effects.items():
        print(f"  {metric}: {effect['means']}")
    path = write_conditions_artifact(
        args.out,
        "experiment-b",
        list(conditions.values()),
        tasks,
        {
            "retriever": retriever.name,
            "model_id": agent.client.model_id,
            "budgets": {name: asdict(budget) for name, budget in budgets.items()},
            "answerability_gate": False,
            "factorial_effects": effects,
            "paired_contrasts": contrasts,
            "claim_support_enabled": args.claim_support,
        },
        provenance=provenance,
        output=args.output,
    )
    print(f"Artifact: {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="hand")
    parser.add_argument("--arm", default="rrf_hybrid_rerank")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--claim-support", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("experiments/runs"))
    parser.add_argument("--output", type=Path, help="exact output filename")
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
