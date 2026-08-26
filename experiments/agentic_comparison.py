"""Experiment B: single-pass RAG versus agentic retrieval. See SPEC 16.2.

Two configurations of the same system, same corpus, same retriever, same writer:

  single_pass  one search, no refinement loop, no cross-reference resolution.
               This is what a conventional RAG pipeline does.
  agentic      the full loop: search, assess sufficiency, refine and search again
               within budget, then resolve cross-references.

Everything except the control policy is held constant, so a difference is attributable
to the loop rather than to retrieval quality or prompting.

    uv run python experiments/agentic_comparison.py --arm rrf_hybrid_rerank
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import asdict
from pathlib import Path

from _harness import ConditionResult, print_report, record, write_artifact

from app.agent.runner import ResearchAgent
from app.agent.state import Budget
from app.config import get_settings
from app.ingestion.store import ChunkStore, StoreError
from app.retrieval.factory import build_retriever
from evals.loader import DatasetError, load_suite

# DECISION: the single-pass arm is produced by narrowing the budget, not by a separate
# code path.
#   One search, no refinement iterations, and a cross-reference chunk budget of zero
#   reproduces conventional RAG exactly, while running the identical graph, writer, and
#   verifier. A separate pipeline would introduce differences nobody intended and make
#   any measured gap partly an artifact of having written two things.
SINGLE_PASS = Budget(
    max_searches=1,
    max_research_iterations=1,
    max_cross_reference_chunks=0,
    max_cross_reference_depth=0,
)
AGENTIC = Budget()


async def main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    try:
        tasks = load_suite(args.suite)
        store = ChunkStore.load(settings.processed_dir)
    except (DatasetError, StoreError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    retriever = build_retriever(args.arm, store, settings.index_dir)
    agent = ResearchAgent(store, settings, retriever=retriever)
    generated = tasks[0].dataset_version.startswith("provisional-generated")

    print("=" * 78)
    print("EXPERIMENT B -- SINGLE-PASS RAG versus AGENTIC RETRIEVAL")
    print("=" * 78)
    print(f"corpus    : {len(store.documents)} documents, {len(store)} chunks")
    print(f"retriever : {retriever.name}")
    print(f"model     : {agent.client.model_id}")
    print(f"suite     : {args.suite} ({tasks[0].dataset_version}), n={len(tasks)}")
    print("single    : searches<=1, no refinement, no cross-reference")
    print(
        f"agentic   : searches<={AGENTIC.max_searches}, iterations<={AGENTIC.max_research_iterations}, "
        f"xref depth {AGENTIC.max_cross_reference_depth} budget {AGENTIC.max_cross_reference_chunks}"
    )
    if agent.client.model_id == "stub":
        print()
        print("!! Running against the deterministic stub. Retrieval and evidence-assembly")
        print("!! differences are real; answer-quality differences are not measurable")
        print("!! until CRA_LIVE_MODEL=1.")
    print()

    baseline = ConditionResult(name="single_pass")
    variant = ConditionResult(name="agentic")

    for label, budget, condition in (
        ("single_pass", SINGLE_PASS, baseline),
        ("agentic", AGENTIC, variant),
    ):
        print(f"running {label}...", flush=True)
        for task in tasks:
            result = await agent.research(
                task.question,
                allowed_document_ids=task.allowed_document_ids,
                budget=budget,
            )
            record(condition, task, result, settings.retrieval_top_k)

    print_report(
        "EXPERIMENT B",
        baseline,
        variant,
        tasks,
        k=settings.retrieval_top_k,
        generated=generated,
        notes=[
            "Both conditions share retriever, writer, verifier, and corpus. Only the",
            "  control policy differs, so a gap is attributable to the loop.",
            "The refiner is a deterministic keyword widener, not a model rewrite. A gain",
            "  here is a floor for what refinement can contribute.",
            "Against the stub, answer-quality metrics measure evidence assembly only.",
        ],
    )

    path = write_artifact(
        args.out,
        "experiment-b",
        baseline,
        variant,
        tasks,
        {
            "retriever": retriever.name,
            "model_id": agent.client.model_id,
            "single_pass_budget": asdict(SINGLE_PASS),
            "agentic_budget": asdict(AGENTIC),
        },
    )
    print(f"\nArtifact: {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="full")
    parser.add_argument("--arm", default="rrf_hybrid_rerank")
    parser.add_argument("--out", type=Path, default=Path("experiments/runs"))
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
