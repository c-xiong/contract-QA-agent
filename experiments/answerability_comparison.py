"""Experiment D: pre-write answerability off/on with an independent offline judge.

Run the identical script with CRA_LIVE_MODEL=0 before an explicitly enabled live run.
The frozen hand suite supplies capability measurements; generated suites are diagnostics.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import cast

from app.agent.answerability import ANSWERABILITY_VERSION
from app.agent.llm import ModelClient, ModelError, ModelResponse, build_client
from app.agent.runner import ResearchAgent, ResearchResult
from app.config import get_settings
from app.evidence.claim_support import CLAIM_SUPPORT_VERSION
from app.ingestion.store import ChunkStore, StoreError
from app.retrieval.factory import ARM_NAMES, build_retriever
from evals.graders.claim_support import claim_support
from evals.graders.retrieval import GradeResult
from evals.loader import DatasetError, load_suite
from evals.provenance import capture_provenance
from evals.schema import EvalTask
from experiments._harness import (
    ConditionResult,
    limited_tasks,
    paired_summary,
    record,
    write_conditions_artifact,
)


@dataclass
class JudgeUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_seconds: float = 0.0
    errors: int = 0


class MeteredJudge:
    """Keep offline scoring costs out of application latency and token totals."""

    def __init__(self, client: ModelClient, max_calls: int) -> None:
        self.client = client
        self.model_id = client.model_id
        self.max_calls = max_calls
        self.usage = JudgeUsage()

    async def complete(self, system: str, user: str) -> ModelResponse:
        if self.usage.calls >= self.max_calls:
            self.usage.errors += 1
            raise ModelError("offline judge call budget exhausted")
        self.usage.calls += 1
        start = perf_counter()
        try:
            response = await self.client.complete(system, user)
        except ModelError:
            self.usage.errors += 1
            raise
        finally:
            self.usage.elapsed_seconds += perf_counter() - start
        self.usage.input_tokens += response.input_tokens
        self.usage.output_tokens += response.output_tokens
        return response


def observed_behavior(result: ResearchResult) -> str:
    if result.status == "failed":
        return "failed"
    if result.abstained:
        return "abstain"
    if result.answerability and result.answerability.verdict == "partial":
        return "partial"
    return "answer" if result.status == "completed" else "failed"


def behavior_summary(
    tasks: list[EvalTask], results: dict[str, ResearchResult], condition: ConditionResult
) -> dict[str, object]:
    """Separate failures from false answers/refusals; preserve every denominator."""
    negative = [task for task in tasks if task.expected_behavior == "abstain"]
    positive = [task for task in tasks if task.expected_behavior != "abstain"]
    false_answers = sum(results[t.task_id].status == "completed" for t in negative)
    correct_abstentions = sum(results[t.task_id].abstained for t in negative)
    over_abstentions = sum(results[t.task_id].abstained for t in positive)
    delivered = [t for t in positive if results[t.task_id].status == "completed"]
    matrix: dict[str, dict[str, int]] = {}
    for expected in ("answer", "partial", "abstain"):
        counts = Counter(
            observed_behavior(results[t.task_id]) for t in tasks if t.expected_behavior == expected
        )
        matrix[expected] = {
            name: counts[name] for name in ("answer", "partial", "abstain", "failed")
        }

    def mean_for(selected: list[EvalTask], metric: str) -> float | None:
        values = [
            condition.per_task[t.task_id][metric]
            for t in selected
            if metric in condition.per_task[t.task_id]
        ]
        return sum(values) / len(values) if values else None

    support_rows = [condition.grade_data[t.task_id].get("claim_support", {}) for t in delivered]
    evaluated = sum(cast(int, row.get("evaluated_claims", 0)) for row in support_rows)
    supported = sum(cast(int, row.get("supported", 0)) for row in support_rows)
    return {
        "unanswerable_tasks": len(negative),
        "answerable_tasks": len(positive),
        "false_answers": false_answers,
        "correct_abstentions": correct_abstentions,
        "over_abstentions": over_abstentions,
        "over_abstention_rate": over_abstentions / len(positive) if positive else None,
        "answerable_delivered": len(delivered),
        "answerable_coverage": len(delivered) / len(positive) if positive else None,
        "execution_failures": sum(r.status == "failed" for r in results.values()),
        "confusion_matrix": matrix,
        "answerable_required_points": mean_for(positive, "required_points"),
        "delivered_answerable_claim_support": mean_for(delivered, "claim_support"),
        "supported_claims": supported,
        "evaluated_claims": evaluated,
        "delivered_answerable_support_micro": supported / evaluated if evaluated else None,
    }


async def main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    try:
        tasks = limited_tasks(load_suite(args.suite), args.limit)
        store = ChunkStore.load(settings.processed_dir)
        if args.concurrency < 1 or args.max_judge_calls < 1:
            raise ValueError("concurrency and judge-call budget must be positive")
    except (DatasetError, StoreError, ValueError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    retriever = build_retriever(args.arm, store, settings.index_dir)
    agents = {
        name: ResearchAgent(store, settings, retriever=retriever, answerability_gate=enabled)
        for name, enabled in (("gate_off", False), ("gate_on", True))
    }
    judge = MeteredJudge(build_client(settings), args.max_judge_calls)
    provenance = capture_provenance(
        store,
        dataset_name=args.suite,
        dataset_version=tasks[0].dataset_version,
        retrieval_arm=retriever.name,
        top_k=settings.retrieval_top_k,
        model_id=judge.model_id,
    )
    print(f"Experiment D: 2 x {len(tasks)} tasks, 1 trial; model={judge.model_id}", flush=True)
    print(
        f"Writer/repair calls <= {4 * len(tasks)}; answerability <= {len(tasks)}; "
        f"offline judge <= {args.max_judge_calls} calls (only if --claim-support).",
        flush=True,
    )
    conditions = {name: ConditionResult(name=name) for name in agents}
    outputs: dict[str, dict[str, ResearchResult]] = {name: {} for name in agents}
    details: dict[str, dict[str, object]] = {name: {} for name in agents}
    semaphore = asyncio.Semaphore(args.concurrency)

    async def run_pair(task: EvalTask, index: int) -> None:
        async with semaphore:
            # Alternate arm order by task to reduce systematic warm-up/time ordering.
            order = list(agents) if index % 2 == 0 else list(reversed(agents))
            for name in order:
                result = await agents[name].research(
                    task.question, allowed_document_ids=task.allowed_document_ids
                )
                grades: list[GradeResult] = []
                if args.claim_support and result.status != "failed":
                    grade = await claim_support(task, result, judge, live=settings.live_model)
                    # A stub/failed judge is not a semantic score of zero.
                    if grade.data.get("evaluated") is not False:
                        grades.append(grade)
                record(
                    conditions[name],
                    task,
                    result,
                    settings.retrieval_top_k,
                    elapsed_seconds=result.elapsed_seconds,
                    extra_grades=grades,
                )
                outputs[name][task.task_id] = result
                details[name][task.task_id] = {
                    "question": task.question,
                    "expected_behavior": task.expected_behavior,
                    "observed_behavior": observed_behavior(result),
                    "answerability": result.answerability.model_dump(mode="json")
                    if result.answerability
                    else None,
                    "failure": result.failure,
                    "failure_detail": result.failure_detail,
                    "evidence": [e.model_dump(mode="json") for e in result.evidence],
                    "trace": [asdict(event) for event in result.trace],
                }
            print(f"completed {task.task_id}", flush=True)

    await asyncio.gather(*(run_pair(task, index) for index, task in enumerate(tasks)))
    behaviors = {
        name: behavior_summary(tasks, outputs[name], condition)
        for name, condition in conditions.items()
    }
    path = write_conditions_artifact(
        args.out,
        "experiment-d",
        list(conditions.values()),
        tasks,
        {
            "retriever": retriever.name,
            "model_id": judge.model_id,
            "answerability_version": ANSWERABILITY_VERSION,
            "claim_support_version": CLAIM_SUPPORT_VERSION if args.claim_support else None,
            "claim_support_enabled": args.claim_support,
            "behavior": behaviors,
            "observations": details,
            "paired_contrasts": paired_summary(conditions["gate_off"], conditions["gate_on"]),
            "judge_usage": asdict(judge.usage),
            "concurrency": args.concurrency,
            "run_complete": True,
            "limitations": [
                "One trial on a development suite; no significance claim.",
                "Runtime and offline judge have independent prompts but the same model family.",
                "Partial is the runtime verdict; baseline completion is not semantic partial credit.",
            ],
        },
        provenance=provenance,
        output=args.output,
    )
    print(json.dumps(behaviors, indent=2))
    print(f"Artifact: {path}")
    return 0 if judge.usage.errors == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="hand")
    parser.add_argument("--arm", choices=list(ARM_NAMES), default="rrf_hybrid_rerank")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--claim-support", action="store_true")
    parser.add_argument("--max-judge-calls", type=int, default=600)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("experiments/runs"))
    parser.add_argument("--output", type=Path)
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
