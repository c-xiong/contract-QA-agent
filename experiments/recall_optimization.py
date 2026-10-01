"""Fixed-k ablations with candidate diagnostics and provenance.

Choose arms on hand-v1, then freeze the choice before running cuad_holdout.
Exact query/text reranker scores are memoized across arms, never labels. Runtime
durations therefore are diagnostic wall times, not serving-latency benchmarks.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TypedDict

import numpy as np

from app.config import get_settings
from app.ingestion.store import ChunkStore
from app.retrieval.factory import build_retriever
from app.retrieval.rerank import CrossEncoderReranker
from app.schemas.retrieval import RetrievalFilters
from evals.graders.retrieval import (
    document_aware_evidence_recall,
    evidence_complete_span_recall,
    evidence_fragment_recall,
    grade_retrieval,
)
from evals.loader import load_suite
from evals.provenance import capture_provenance
from evals.schema import EvalTask

NEW_GRADERS = (
    document_aware_evidence_recall,
    evidence_fragment_recall,
    evidence_complete_span_recall,
)


class TaskResult(TypedDict):
    category: str
    scores: dict[str, float]
    candidate_scores: dict[str, float]
    chunk_ids: list[str]
    candidate_chunk_ids: list[str]
    returned_estimated_tokens: int
    wall_seconds_with_cache: float


class ArmResult(TypedDict):
    means: dict[str, float]
    by_category: dict[str, dict[str, float]]
    per_task: dict[str, TaskResult]


def paired(
    left: dict[str, TaskResult], right: dict[str, TaskResult], metric: str, tasks: list[EvalTask]
) -> dict[str, object]:
    differences = [
        left[t.task_id]["scores"][metric] - right[t.task_id]["scores"][metric] for t in tasks
    ]
    # Resample source-document groups rather than pretending repeated clause
    # categories from the same contract are independent observations.
    groups: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for task, delta in zip(tasks, differences, strict=True):
        groups[tuple(sorted(task.allowed_document_ids or task.expected_document_ids))].append(delta)
    values = list(groups.values())
    rng = np.random.default_rng(20260926)
    boot = []
    for _ in range(2000):
        sample = [
            v for index in rng.integers(0, len(values), len(values)) for v in values[int(index)]
        ]
        boot.append(float(np.mean(sample)))
    return {
        "delta": float(np.mean(differences)),
        "wins": sum(d > 1e-9 for d in differences),
        "losses": sum(d < -1e-9 for d in differences),
        "ties": sum(abs(d) <= 1e-9 for d in differences),
        "document_group_bootstrap_95ci": np.quantile(boot, [0.025, 0.975]).tolist(),
        "bootstrap_note": "For multi-document development tasks, groups can overlap; CI is descriptive only.",
    }


async def run(args: argparse.Namespace) -> None:
    settings = get_settings()
    store = ChunkStore.load(settings.processed_dir)
    tasks = [t for t in load_suite(args.suite) if t.expected_behavior != "abstain"]
    payload = capture_provenance(
        store,
        dataset_name=args.suite,
        dataset_version=tasks[0].dataset_version,
        retrieval_arm="recall-optimization",
        top_k=5,
        model_id="local-minilm",
    )
    arms: dict[str, ArmResult] = {}
    comparisons: dict[str, dict[str, dict[str, object]]] = {}
    payload.update(
        task_count=len(tasks),
        trials=1,
        k=5,
        final_budget="five original chunks",
        respect_allowlist=True,
        rerank_depth=40,
        arms=arms,
        comparisons=comparisons,
        timing_note="Exact query/text scores memoized across arms; wall times are NOT serving latency.",
    )
    versions: dict[str, str] = {}
    cache_path = settings.index_dir / "recall_rerank_scores.json"
    cache: dict[str, float] = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for arm in args.arms:
        retriever = build_retriever(arm, store, settings.index_dir)
        if isinstance(retriever, CrossEncoderReranker):
            model = retriever._model
            original_predict = model.predict
            model_id = retriever.model_id

            def cached_predict(
                pairs: list[tuple[str, str]],
                *,
                show_progress_bar: bool = False,
                _predict: Callable[..., Iterable[float]] = original_predict,
                _model_id: str = model_id,
            ) -> list[float]:
                keys = [
                    hashlib.sha256(json.dumps([_model_id, q, t]).encode()).hexdigest()
                    for q, t in pairs
                ]
                missing = list(dict.fromkeys(key for key in keys if key not in cache))
                lookup = dict(zip(keys, pairs, strict=True))
                if missing:
                    scores = _predict(
                        [lookup[key] for key in missing], show_progress_bar=show_progress_bar
                    )
                    cache.update(
                        {key: float(score) for key, score in zip(missing, scores, strict=True)}
                    )
                return [cache[key] for key in keys]

            model.predict = cached_predict
        rows: dict[str, TaskResult] = {}
        print(f"ARM {arm} n={len(tasks)}", flush=True)
        for index, task in enumerate(tasks, 1):
            filters = (
                RetrievalFilters(document_ids=task.allowed_document_ids)
                if task.allowed_document_ids
                else None
            )
            started = time.monotonic()
            if isinstance(retriever, CrossEncoderReranker):
                candidates = await retriever.retrieve_candidates(task.question, filters)
                hits = await retriever.rerank_candidates(task.question, candidates, 5, filters)
            else:
                hits = await retriever.search(task.question, 5, filters)
                candidates = await retriever.search(task.question, 40, filters)
            elapsed = time.monotonic() - started
            grades = grade_retrieval(task, hits, 5) + [
                grader(task, hits, 5) for grader in NEW_GRADERS
            ]
            candidate_grades = [grader(task, candidates, len(candidates)) for grader in NEW_GRADERS]
            versions.update({g.name: g.version for g in grades})
            rows[task.task_id] = {
                "category": task.category,
                "scores": {g.name: g.score for g in grades},
                "candidate_scores": {g.name: g.score for g in candidate_grades},
                "chunk_ids": [hit.chunk_id for hit in hits],
                "candidate_chunk_ids": [hit.chunk_id for hit in candidates],
                "returned_estimated_tokens": sum(hit.chunk.token_count for hit in hits),
                "wall_seconds_with_cache": elapsed,
            }
            if index % 6 == 0 or index == len(tasks):
                print(
                    f"  {index}/{len(tasks)} probe={np.mean([r['scores']['document_aware_evidence_recall'] for r in rows.values()]):.3f} complete={np.mean([r['scores']['evidence_complete_span_recall'] for r in rows.values()]):.3f}",
                    flush=True,
                )
                cache_path.write_text(json.dumps(cache))
        metrics = list(next(iter(rows.values()))["scores"])
        means = {m: float(np.mean([r["scores"][m] for r in rows.values()])) for m in metrics}
        by_category = {
            category: {
                "n": sum(t.category == category for t in tasks),
                **{
                    m: float(
                        np.mean(
                            [r["scores"][m] for r in rows.values() if r["category"] == category]
                        )
                    )
                    for m in metrics
                },
            }
            for category in sorted({t.category for t in tasks})
        }
        arms[arm] = {"means": means, "by_category": by_category, "per_task": rows}
        payload["grader_versions"] = versions
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"RESULT {arm}: {means}", flush=True)
    for arm in args.arms[1:]:
        for baseline in args.arms[: args.arms.index(arm)]:
            comparisons[f"{arm} vs {baseline}"] = {
                m: paired(
                    arms[arm]["per_task"],
                    arms[baseline]["per_task"],
                    m,
                    tasks,
                )
                for m in versions
            }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Artifact: {args.output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="hand")
    parser.add_argument(
        "--arms",
        nargs="+",
        default=[
            "bm25",
            "rrf_hybrid_rerank",
            "coverage_rerank",
            "windowed_rerank",
            "windowed_coverage_rerank",
        ],
    )
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
