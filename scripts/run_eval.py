"""Run an eval suite and write a result artifact.

uv run python scripts/run_eval.py --suite smoke
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from app.agent.runner import ResearchAgent
from app.config import get_settings
from app.ingestion.store import ChunkStore, StoreError
from app.retrieval.factory import ARM_NAMES, UnknownArmError, build_retriever
from evals.loader import DatasetError, load_suite
from evals.runner import run_suite, write_report


async def run(
    suite: str, out_dir: Path, *, arm: str = "bm25", with_claim_support: bool = False
) -> int:
    settings = get_settings()
    try:
        tasks = load_suite(suite)
    except DatasetError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    try:
        store = ChunkStore.load(settings.processed_dir)
        retriever = build_retriever(arm, store, settings.index_dir)
        agent = ResearchAgent(store, settings, retriever=retriever)
    except (StoreError, UnknownArmError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {len(tasks)} task(s) from suite {suite!r}\n")
    if with_claim_support and not settings.live_model:
        print(
            "NOTE: --claim-support needs CRA_LIVE_MODEL=1; it will report 'not evaluated'.",
            file=sys.stderr,
        )
    report = await run_suite(suite, tasks, agent, settings, with_claim_support=with_claim_support)
    print(report.summary())

    path = write_report(report, out_dir)
    print(f"\nArtifact: {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="smoke")
    parser.add_argument(
        "--arm",
        default="bm25",
        choices=list(ARM_NAMES),
        help="retrieval arm. The default is bm25 so a run needs no dense index; use the "
        "arm the service actually runs when reporting headline numbers.",
    )
    parser.add_argument("--out", type=Path, default=Path("evals/runs"))
    parser.add_argument(
        "--claim-support",
        action="store_true",
        help="also run the model-based claim-support grader (SPEC 13.4 layer 2). "
        "One model call per factual claim; roughly doubles the cost of a run.",
    )
    args = parser.parse_args()
    return asyncio.run(
        run(args.suite, args.out, arm=args.arm, with_claim_support=args.claim_support)
    )


if __name__ == "__main__":
    raise SystemExit(main())
