"""Matched V1/V2 runs, retaining all failures and blank manual adjudication fields."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Any

from app.agent.runner import ResearchAgent
from app.agent.state import Budget
from app.agent.tool_registry import digest
from app.config import Settings
from app.ingestion.store import ChunkStore
from app.observability.events import index_hash, json_value
from app.retrieval.factory import ARM_NAMES, build_retriever
from evals.graders.v2 import VERSION, ManualLabel, diagnostics, ratio
from evals.loader import load_suite
from evals.provenance import capture_provenance


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for mode in ("v1", "v2"):
        chosen = [r for r in rows if r["mode"] == mode]
        categories = sorted({r["category"] for r in chosen})
        groups[mode] = {}
        for subset in ["all", "answerable", "unanswerable", *categories]:
            selected = [
                r["diagnostics"]
                for r in chosen
                if subset == "all"
                or (subset == "answerable" and r["diagnostics"]["answerable"])
                or (subset == "unanswerable" and not r["diagnostics"]["answerable"])
                or r["category"] == subset
            ]

            def med(key: str, selected: list[dict[str, Any]] = selected) -> float | None:
                values = [d[key] for d in selected if d[key] is not None]
                return median(values) if values else None

            groups[mode][subset] = {
                "n": len(selected),
                "statuses": dict(Counter(d["status"] for d in selected)),
                "answer_abstain_decision": ratio(
                    sum(d["answer_abstain_decision"] for d in selected), len(selected)
                ),
                "false_abstention": ratio(
                    sum(d["false_abstention"] for d in selected),
                    sum(d["answerable"] for d in selected),
                ),
                "structural_gate_pass": ratio(
                    sum(d["structural_gate_pass"] for d in selected), len(selected)
                ),
                "task_success": None,
                "semantic_support": None,
                "median_latency_seconds": med("elapsed_seconds"),
                "median_model_calls": med("model_calls"),
                "median_tool_calls": med("tool_calls"),
                "median_input_tokens": med("input_tokens"),
                "median_output_tokens": med("output_tokens"),
            }
    pairs: dict[str, Any] = {}
    for metric in ("evidence_recall_at_5", "trajectory_evidence_recall", "answer_abstain_decision"):
        wins = losses = ties = 0
        for task_id in sorted({r["task_id"] for r in rows}):
            pair = {r["mode"]: r["diagnostics"][metric] for r in rows if r["task_id"] == task_id}
            if pair.get("v1") is None or pair.get("v2") is None:
                continue
            wins += pair["v2"] > pair["v1"]
            losses += pair["v2"] < pair["v1"]
            ties += pair["v2"] == pair["v1"]
        pairs[metric] = {"wins": wins, "losses": losses, "ties": ties, "n": wins + losses + ties}
    return {"groups": groups, "paired": pairs}


async def run(args: argparse.Namespace) -> None:
    settings = Settings()
    tasks = load_suite(args.suite)
    if args.limit:
        tasks = tasks[: args.limit]
    store = ChunkStore.load(settings.processed_dir)
    retriever = build_retriever(args.arm, store, settings.index_dir)
    budget = Budget()
    provenance = capture_provenance(
        store,
        dataset_name=args.suite,
        dataset_version=tasks[0].dataset_version,
        retrieval_arm=args.arm,
        top_k=budget.max_chunks_per_query,
        model_id=settings.model_id if settings.live_model else "stub",
        grader_versions={"diagnostics": VERSION},
    )
    provenance.update(
        {
            "budget": asdict(budget),
            "document_metadata_hash": digest([d.model_dump(mode="json") for d in store.documents]),
            "index_hash": index_hash(settings.index_dir),
            "trial_count": 1,
            "task_count": len(tasks),
            "live_model": settings.live_model,
            "reporting_eligible": settings.live_model
            and all(
                t.review_status == "author-reviewed"
                and not t.dataset_version.startswith("provisional-generated-")
                for t in tasks
            ),
            "manual_review_complete": False,
            "cache": "shared retriever/model cache; no response cache; V1 runs before V2 per case",
        }
    )
    output = Path(
        args.output
        or f"reports/v2-{args.suite}-{'live' if settings.live_model else 'offline'}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    agents = {
        mode: ResearchAgent(
            store,
            settings.model_copy(update={"agent_mode": mode}),
            retriever=retriever,
            answerability_gate=False,
        )
        for mode in ("v1", "v2")
    }
    rows: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    # Write each case immediately, so interruption cannot erase previous failed runs.
    predictions = output.with_suffix(".predictions.jsonl")
    with predictions.open("w") as handle:
        for task in tasks:
            for mode, agent in agents.items():
                result = await agent.research(
                    task.question,
                    allowed_document_ids=task.allowed_document_ids,
                    budget=budget,
                    requested_versions=task.requested_versions,
                )
                snapshot = None
                if result.trace_path:
                    snapshot = json.loads(
                        (await asyncio.to_thread(Path(result.trace_path).read_text)).splitlines()[
                            -1
                        ]
                    )["state"]
                row = {
                    "task_id": task.task_id,
                    "mode": mode,
                    "category": task.category,
                    "dataset_version": task.dataset_version,
                    "diagnostics": diagnostics(task, result, snapshot),
                    "result": json_value(asdict(result)),
                }
                rows.append(row)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                labels.append(ManualLabel(task_id=task.task_id, mode=mode).model_dump())
    await asyncio.to_thread(
        output.write_text,
        json.dumps(
            {
                "provenance": provenance,
                "summary": summarize(rows),
                "manual_metrics": "Pending independent human annotation; stub results are wiring checks only.",
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
    )
    output.with_suffix(".review.jsonl").write_text("".join(json.dumps(r) + "\n" for r in labels))
    print(
        f"Saved {len(rows)} runs ({len(tasks)} matched cases) to {output}; semantic judgments remain unscored."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["hand", "v2"], default="hand")
    parser.add_argument("--arm", choices=ARM_NAMES, default="bm25")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
