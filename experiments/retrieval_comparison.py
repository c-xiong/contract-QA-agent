"""Experiment A: retrieval comparison. See docs/eval-methodology.md

Compares BM25, dense, RRF hybrid, and RRF+cross-encoder rerank on the same annotated
tasks, at the same k. Retrieval calls no model, so this experiment is free, fully
deterministic, and re-runnable at will.

Reporting follows SPEC 15.7 and .claude/rules/evals.md, which are strict for a reason:
with n in the low tens, a two-task difference is noise and an aggregate percentage hides
the per-category effect that is the actual finding. Every table therefore carries its n,
every comparison carries paired win/loss/tie counts, and failures are printed for
inspection rather than summarised away.

    uv run python experiments/retrieval_comparison.py
    uv run python experiments/retrieval_comparison.py --k 10 --suite full
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from app.config import get_settings
from app.ingestion.store import ChunkStore, StoreError
from app.retrieval.factory import ARM_NAMES, build_retriever
from app.schemas.retrieval import RetrievalFilters
from evals.graders.retrieval import grade_retrieval
from evals.loader import DatasetError, load_suite
from evals.provenance import capture_provenance
from evals.schema import EvalTask


@dataclass
class ArmResult:
    """One arm's scores across the whole suite."""

    arm: str
    per_task: dict[str, dict[str, float]] = field(default_factory=dict)
    per_task_category: dict[str, str] = field(default_factory=dict)
    grader_versions: dict[str, str] = field(default_factory=dict)

    def mean(self, metric: str) -> float:
        values = [s[metric] for s in self.per_task.values() if metric in s]
        return sum(values) / len(values) if values else 0.0

    def mean_by_category(self, metric: str) -> dict[str, tuple[float, int]]:
        buckets: dict[str, list[float]] = defaultdict(list)
        for task_id, scores in self.per_task.items():
            if metric in scores:
                buckets[self.per_task_category[task_id]].append(scores[metric])
        return {c: (sum(v) / len(v), len(v)) for c, v in sorted(buckets.items())}


def paired_counts(
    left: ArmResult, right: ArmResult, metric: str
) -> tuple[int, int, int, list[str]]:
    """Win / loss / tie for `left` against `right`, plus the task ids left lost on.

    Paired comparison on the same tasks, which is the only honest way to compare arms at
    this sample size. An aggregate difference of 0.05 between two arms says nothing about
    whether they failed on the same tasks or different ones.
    """
    wins = losses = ties = 0
    lost_on: list[str] = []
    for task_id, scores in sorted(left.per_task.items()):
        if task_id not in right.per_task:
            continue
        a, b = scores.get(metric, 0.0), right.per_task[task_id].get(metric, 0.0)
        if a > b:
            wins += 1
        elif a < b:
            losses += 1
            lost_on.append(task_id)
        else:
            ties += 1
    return wins, losses, ties, lost_on


async def run_arm(
    arm: str,
    tasks: list[EvalTask],
    store: ChunkStore,
    index_dir: Path,
    k: int,
    *,
    respect_allowlist: bool,
) -> tuple[ArmResult, dict[str, list[str]]]:
    """Score one arm over the suite.

    DECISION: the primary condition honours each task's `allowed_document_ids`.
      This is how the system actually runs -- the API and the agent both pass the
      allowlist through to the retriever -- so it is the faithful measurement.
      It also fixes a flaw that the unfiltered condition exposed. A question like "Can a
      party walk away without the other side being at fault?" is answerable from roughly
      thirty of the forty contracts. Unfiltered, retrieving any of them is correct
      behaviour, but only one is the labelled document, so `document_recall` punishes
      correct retrieval. The metric was measuring question ambiguity, not retrieval.
      The unfiltered condition is still worth running (`--no-allowlist`) as a harder
      corpus-wide task, but it must be reported as a different question, not as the same
      number under worse conditions.
    """
    retriever = build_retriever(arm, store, index_dir)
    result = ArmResult(arm=retriever.name)
    misses: dict[str, list[str]] = {}

    for task in tasks:
        filters = None
        if respect_allowlist and task.allowed_document_ids:
            filters = RetrievalFilters(document_ids=task.allowed_document_ids)
        retrieved = await retriever.search(task.question, top_k=k, filters=filters)
        grades = grade_retrieval(task, retrieved, k)
        result.per_task[task.task_id] = {g.name: g.score for g in grades}
        result.grader_versions.update({g.name: g.version for g in grades})
        result.per_task_category[task.task_id] = task.category

        if any(g.name == "document_recall" and g.score < 1.0 for g in grades):
            misses[task.task_id] = [c.chunk.document_id for c in retrieved[:3]]

    return result, misses


async def main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    try:
        tasks = load_suite(args.suite)
        store = ChunkStore.load(settings.processed_dir)
    except (DatasetError, StoreError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    generated = tasks[0].dataset_version.startswith("provisional-generated")
    provenance = capture_provenance(
        store,
        dataset_name=args.suite,
        dataset_version=tasks[0].dataset_version,
        retrieval_arm="four-arm-comparison",
        top_k=args.k,
        model_id="none",
    )

    # DECISION: abstain tasks are excluded from the retrieval aggregate.
    #   They carry no expected documents, so every retrieval grader scores them 1.0 by
    #   definition (see evals/graders/retrieval.py). Including them inflates every arm
    #   equally and compresses the differences that Experiment A exists to measure.
    #   They are still counted and reported separately, so n stays honest.
    scored = [t for t in tasks if t.expected_behavior != "abstain"]
    abstain_count = len(tasks) - len(scored)

    print("=" * 78)
    print("EXPERIMENT A -- RETRIEVAL COMPARISON")
    print("=" * 78)
    print(f"corpus       : {len(store.documents)} documents, {len(store)} chunks")
    print(f"suite        : {args.suite} ({tasks[0].dataset_version})")
    print(
        f"tasks (n)    : {len(scored)} scored"
        + (f", {abstain_count} abstain excluded" if abstain_count else "")
    )
    print("trials       : 1  (retrieval is deterministic; repeats add nothing)")
    print(f"k            : {args.k}")
    print(
        "condition    : "
        + (
            "corpus-wide (allowlist ignored)"
            if args.no_allowlist
            else "allowlist honoured (as the system runs)"
        )
    )
    print(f"arms         : {', '.join(args.arms)}")
    if generated:
        print()
        print("!! QUESTIONS IN THIS SUITE ARE MODEL-GENERATED. Evidence labels are expert")
        print("!! annotation, but the questions were written by the same model family the")
        print("!! system uses. Treat every figure below as a pipeline check, not a")
        print("!! capability measurement. See CLAUDE.md rule 1.")
    print()

    results: dict[str, ArmResult] = {}
    all_misses: dict[str, dict[str, list[str]]] = {}
    for arm in args.arms:
        print(f"running {arm}...", flush=True)
        result, misses = await run_arm(
            arm,
            scored,
            store,
            settings.index_dir,
            args.k,
            respect_allowlist=not args.no_allowlist,
        )
        results[arm] = result
        all_misses[arm] = misses

    metrics = ("document_recall", "evidence_span_recall", "reciprocal_rank")

    print("\n" + "=" * 78)
    print(f"AGGREGATE  (n={len(scored)} scored tasks, 1 trial, k={args.k})")
    print("=" * 78)
    header = f"{'arm':<26}" + "".join(f"{m.replace('_', ' '):>22}" for m in metrics)
    print(header)
    for arm in args.arms:
        row = f"{results[arm].arm:<26}"
        row += "".join(f"{results[arm].mean(m):>22.3f}" for m in metrics)
        print(row)

    print("\n" + "=" * 78)
    print("PER CATEGORY -- document_recall  (mean, n)")
    print("=" * 78)
    categories = sorted({t.category for t in scored})
    print(f"{'category':<34}" + "".join(f"{a[:14]:>16}" for a in args.arms))
    for category in categories:
        row = f"{category:<34}"
        for arm in args.arms:
            by_cat = results[arm].mean_by_category("document_recall")
            if category in by_cat:
                mean, n = by_cat[category]
                row += f"{f'{mean:.2f} (n={n})':>16}"
            else:
                row += f"{'-':>16}"
        print(row)

    print("\n" + "=" * 78)
    print("PAIRED COMPARISONS -- document_recall, same tasks")
    print("=" * 78)
    baseline = args.arms[0]
    for arm in args.arms[1:]:
        wins, losses, ties, lost = paired_counts(results[arm], results[baseline], "document_recall")
        print(f"{arm} vs {baseline}:  win {wins}  loss {losses}  tie {ties}")
        if lost:
            print(f"    lost on: {', '.join(lost[:8])}")

    print("\n" + "=" * 78)
    print("FAILURE INSPECTION")
    print("=" * 78)
    best = max(args.arms, key=lambda a: results[a].mean("document_recall"))
    misses = all_misses[best]
    print(f"Best arm by document_recall: {results[best].arm}, {len(misses)} task(s) missed\n")
    task_by_id = {t.task_id: t for t in scored}
    for task_id, retrieved_docs in list(misses.items())[:5]:
        task = task_by_id[task_id]
        print(f"  {task_id}  [{task.category}]")
        print(f"    question : {task.question[:70]}")
        print(f"    expected : {task.expected_document_ids}")
        print(f"    got      : {retrieved_docs}")
        for other in args.arms:
            if other != best and task_id not in all_misses[other]:
                print(f"    NOTE     : {other} found it; this is a fusion/ranking loss, not a miss")
        print()

    print("=" * 78)
    print("LIMITATIONS")
    print("=" * 78)
    print(f"  - n = {len(scored)} scored tasks ({abstain_count} abstain tasks excluded:")
    print("    they have no expected documents and score 1.0 for every arm by definition).")
    print("    A difference of one or two tasks is noise. Report")
    print("    these as directional only; no significance claim is supported at this n.")
    print("  - Per-category n is smaller still (2 to 10). Category rows are indicative.")
    print("  - The dense and rerank arms use small general-purpose models with no legal")
    print("    training, so their numbers are a floor for the approach, not a ceiling.")
    if generated:
        print("  - Questions are model-generated. This is a pipeline check, not a")
        print("    capability measurement.")

    payload = {
        **provenance,
        "grader_versions": {k: v for r in results.values() for k, v in r.grader_versions.items()},
        "experiment": "A_retrieval_comparison",
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "suite": args.suite,
        "dataset_version": tasks[0].dataset_version,
        "questions_generated": generated,
        "task_count": len(scored),
        "abstain_excluded": abstain_count,
        "respect_allowlist": not args.no_allowlist,
        "trials": 1,
        "k": args.k,
        "corpus": {"documents": len(store.documents), "chunks": len(store)},
        "arms": {
            arm: {
                "name": results[arm].arm,
                "means": {m: results[arm].mean(m) for m in metrics},
                "by_category": {
                    c: {"mean": v[0], "n": v[1]}
                    for c, v in results[arm].mean_by_category("document_recall").items()
                },
                "per_task": results[arm].per_task,
                "span_recall_by_category": {
                    c: {"mean": v[0], "n": v[1]}
                    for c, v in results[arm].mean_by_category("evidence_span_recall").items()
                },
            }
            for arm in args.arms
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = payload["run_at"].replace(":", "").replace("-", "")
    path = args.output or args.out / f"experiment-a-{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nArtifact: {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="full")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--arms", nargs="+", default=list(ARM_NAMES))
    parser.add_argument(
        "--no-allowlist",
        action="store_true",
        help="ignore each task's allowed_document_ids and search the whole corpus",
    )
    parser.add_argument("--out", type=Path, default=Path("experiments/runs"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
