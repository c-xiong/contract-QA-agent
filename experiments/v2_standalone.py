"""Standalone V2 evaluation run: no V1 execution, incremental, budgeted and resumable.

Offline (default) this is a wiring check with the deterministic stub and makes no
provider calls. A live run requires CRA_LIVE_MODEL=1 and an application-phase preflight;
every provider attempt then passes through a persisted budget ledger.

    CRA_LIVE_MODEL=0 uv run python -m experiments.v2_standalone --suite hand
    CRA_LIVE_MODEL=1 uv run python -m experiments.v2_standalone --suite hand \
        --arm rrf_hybrid_rerank --preflight evals/preflight/v2-hand-application.json \
        --output reports/v2-standalone-hand-live.json [--resume]

Semantic quality is not scored here. Human labels (see scripts/v2_label_sheet.py) are
the reported answer-quality outcome; the offline judge is measured against them.
"""

# ruff: noqa: ASYNC240 -- sequential local artifact I/O between awaited runs; no concurrency

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Any

from app.agent.llm import MAX_OUTPUT_TOKENS, AnthropicClient, ModelClient
from app.agent.runner import ResearchAgent
from app.agent.state import Budget
from app.agent.tool_registry import TOOL_VERSION, digest
from app.agent.v2 import PLANNER_VERSION
from app.config import Settings
from app.ingestion.store import ChunkStore
from app.observability.events import index_hash, json_value
from app.retrieval.factory import ARM_NAMES, build_retriever
from evals.budget import BudgetedClient, Ledger, load_preflight
from evals.graders.v2 import VERSION, diagnostics, ratio
from evals.loader import load_suite, suite_path
from evals.provenance import capture_provenance

SPLIT = Path("evals/datasets/hand/judge_split-v1.json")
CLAIM_CEILING = Path("evals/datasets/hand/claim_ceiling-v1.json")


def sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def split_sides(suite: str) -> dict[str, str]:
    if suite != "hand" or not SPLIT.exists():
        return {}
    manifest = json.loads(SPLIT.read_text())
    return {t: side for side in ("calibration", "audit") for t in manifest[side]}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Deterministic diagnostics only, each with an explicit denominator."""

    def group(selected: list[dict[str, Any]]) -> dict[str, Any]:
        d = [r["diagnostics"] for r in selected]

        def med(key: str) -> float | None:
            values = [x[key] for x in d if x[key] is not None]
            return median(values) if values else None

        return {
            "n": len(d),
            "statuses": dict(Counter(x["status"] for x in d)),
            "execution": dict(Counter(r["execution"] for r in selected)),
            "answer_abstain_decision": ratio(sum(x["answer_abstain_decision"] for x in d), len(d)),
            "false_abstention": ratio(
                sum(x["false_abstention"] for x in d), sum(x["answerable"] for x in d)
            ),
            "structural_gate_pass": ratio(sum(x["structural_gate_pass"] for x in d), len(d)),
            "failed_or_cancelled": ratio(
                sum(x["status"] in ("failed", "cancelled") for x in d), len(d)
            ),
            "task_success": "human label required",
            "semantic_support": "human label required",
            "median_latency_seconds": med("elapsed_seconds"),
            "median_model_calls": med("model_calls"),
            "median_tool_calls": med("tool_calls"),
            "median_input_tokens": med("input_tokens"),
            "median_output_tokens": med("output_tokens"),
            "unknown_usage_runs": sum(x["input_tokens"] is None for x in d),
        }

    subsets: dict[str, Any] = {"all": group(rows)}
    for key in ("category", "split"):
        for value in sorted({str(r[key]) for r in rows}):
            subsets[f"{key}:{value}"] = group([r for r in rows if str(r[key]) == value])
    return subsets


def frozen_config(
    args: argparse.Namespace, settings: Settings, store: ChunkStore, budget: Budget
) -> dict[str, Any]:
    """Everything that must not change between a run and its resumption."""
    return {
        "suite": args.suite,
        "dataset_sha256": sha256(suite_path(args.suite)),
        "split_sha256": sha256(SPLIT) if args.suite == "hand" else None,
        "claim_ceiling_sha256": sha256(CLAIM_CEILING),
        "retrieval_arm": args.arm,
        "budget": asdict(budget),
        "live_model": settings.live_model,
        "model_id": settings.model_id if settings.live_model else "stub",
        "planner_version": PLANNER_VERSION,
        "tool_version": TOOL_VERSION,
        "corpus_hash": digest([d.model_dump(mode="json") for d in store.documents]),
        "index_hash": index_hash(settings.index_dir),
        "answerability_gate": False,
        "limit": args.limit,
    }


def build_client(
    args: argparse.Namespace, settings: Settings, output: Path
) -> tuple[ModelClient | None, Ledger | None]:
    if not settings.live_model:
        return None, None
    if args.preflight is None:
        raise SystemExit("A live run requires --preflight; see evals/preflight/README.md.")
    preflight = load_preflight(args.preflight)
    if preflight.phase != "application":
        raise SystemExit("This runner needs an application-phase preflight.")
    if preflight.max_output_tokens_per_call < MAX_OUTPUT_TOKENS:
        raise SystemExit(
            f"max_output_tokens_per_call must cover the client's {MAX_OUTPUT_TOKENS}-token ceiling"
        )
    ledger = Ledger(preflight, output.with_suffix(".ledger.json"))
    return BudgetedClient(AnthropicClient(settings), ledger), ledger


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def run(args: argparse.Namespace) -> Path:
    settings = Settings().model_copy(update={"agent_mode": "v2"})
    tasks = load_suite(args.suite)
    if args.limit:
        tasks = tasks[: args.limit]
    mode = "live" if settings.live_model else "offline"
    output = Path(args.output or f"reports/v2-standalone-{args.suite}-{mode}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions = output.with_suffix(".predictions.jsonl")
    config_path = output.with_suffix(".config.json")

    store = ChunkStore.load(settings.processed_dir)
    retriever = build_retriever(args.arm, store, settings.index_dir)
    budget = Budget()
    config = frozen_config(args, settings, store, budget)
    if predictions.exists() or config_path.exists():
        if not args.resume:
            raise SystemExit(f"{predictions} exists; pass --resume or choose a new --output.")
        if json.loads(config_path.read_text()) != config:
            raise SystemExit("Frozen configuration changed since this run started; not resuming.")
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n")

    client, ledger = build_client(args, settings, output)
    agent = ResearchAgent(
        store, settings, retriever=retriever, answerability_gate=False, client=client
    )
    sides = split_sides(args.suite)
    done = {r["task_id"] for r in load_rows(predictions) if r["execution"] != "budget_stopped"}
    with predictions.open("a") as handle:
        for task in tasks:
            if task.task_id in done:
                continue
            if ledger and ledger.exhausted:
                break
            refused_before = ledger.state.refused if ledger else 0
            result = await agent.research(
                task.question,
                allowed_document_ids=task.allowed_document_ids,
                budget=budget,
                requested_versions=task.requested_versions,
            )
            snapshot = None
            if result.trace_path:
                text = await asyncio.to_thread(Path(result.trace_path).read_text)
                snapshot = json.loads(text.splitlines()[-1])["state"]
            execution = "completed"
            if ledger and ledger.exhausted:
                # The experiment cap interrupted this case; it is re-run on resume.
                execution = "budget_stopped"
            elif ledger and ledger.state.refused > refused_before:
                # One oversized prompt was refused; the case stays in denominators as a failure.
                execution = "budget_refused_call"
            row = {
                "task_id": task.task_id,
                "category": task.category,
                "split": sides.get(task.task_id),
                "dataset_version": task.dataset_version,
                "execution": execution,
                "diagnostics": diagnostics(task, result, snapshot),
                "result": json_value(asdict(result)),
            }
            handle.write(json.dumps(row) + "\n")
            handle.flush()

    # The last row per task wins; earlier budget_stopped attempts stay in predictions.jsonl.
    latest: dict[str, dict[str, Any]] = {}
    for row in load_rows(predictions):
        latest[row["task_id"]] = row
    rows = [latest[t.task_id] for t in tasks if t.task_id in latest]
    executed = {r["task_id"] for r in rows if r["execution"] != "budget_stopped"}
    unexecuted = [
        {
            "task_id": t.task_id,
            "reason": "budget_exhausted" if ledger and ledger.exhausted else "not_reached",
        }
        for t in tasks
        if t.task_id not in executed
    ]
    provenance = capture_provenance(
        store,
        dataset_name=args.suite,
        dataset_version=tasks[0].dataset_version,
        retrieval_arm=args.arm,
        top_k=budget.max_chunks_per_query,
        model_id=config["model_id"],
        grader_versions={"diagnostics": VERSION},
    )
    provenance.update(
        {
            "frozen_config": config,
            "frozen_config_sha256": digest(config),
            "trial_count": 1,
            "task_count": len(tasks),
            "reporting_eligible": settings.live_model
            and all(
                t.review_status == "author-reviewed"
                and not t.dataset_version.startswith("provisional-generated-")
                for t in tasks
            ),
            "human_review_complete": False,
        }
    )
    report = {
        "provenance": provenance,
        "summary": summarize([r for r in rows if r["execution"] != "budget_stopped"]),
        "unexecuted": unexecuted,
        "budget_ledger": ledger.summary() if ledger else "offline: no provider calls",
        "answer_quality": "Pending human labels; stub results are wiring checks only.",
        "rows": rows,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Saved {len(executed)}/{len(tasks)} executed cases to {output}"
        + (f"; {len(unexecuted)} unexecuted" if unexecuted else "")
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", choices=["hand", "v2"], default="hand")
    parser.add_argument("--arm", choices=ARM_NAMES, default="bm25")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output")
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--resume", action="store_true")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
