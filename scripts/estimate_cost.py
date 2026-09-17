"""Measure the token cost of an eval run. SPEC 6.10.

Every number this prints is measured against the ingested corpus, not assumed. It
runs the real retrieval path and the real prompt builder, then counts characters and
converts with the configured heuristic. No model is called, so running this costs
nothing.

    uv run python scripts/estimate_cost.py
    uv run python scripts/estimate_cost.py --tasks 40 --trials 3
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys

from app.agent.graph import to_evidence
from app.agent.prompts import WRITER_SYSTEM, build_writer_prompt
from app.agent.runner import load_agent
from app.config import get_settings
from app.ingestion.store import StoreError
from app.ingestion.text import estimate_tokens

# Probe queries spanning the retrieval character of the eight categories in scope.
# These are cost probes, not eval tasks: they are never scored and never graded, so
# they are not eval content under CLAUDE.md rule 1.
PROBE_QUERIES = [
    "governing law",
    "limitation of liability cap",
    "uncapped liability carve-outs",
    "termination for convenience notice",
    "notice period to terminate renewal",
    "non-compete exclusivity restriction",
    "change of control assignment",
    "insurance coverage requirements",
]

# Standard API rates verified 2026-09-16; the Sonnet 5 scheduled increase was cancelled.
# https://platform.claude.com/docs/en/about-claude/pricing
PRICING = {
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
    "claude-opus-5": {"input": 5.00, "output": 25.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}

# Measured from the Sprint 0 stub runs; a real answer is longer than a stub answer,
# so this is a deliberate over-estimate rather than an observation.
ASSUMED_OUTPUT_TOKENS = 400


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, default=40, help="tasks in a full suite")
    parser.add_argument("--trials", type=int, default=3, help="trials per task")
    parser.add_argument("--arms", type=int, default=4, help="variants compared per experiment")
    args = parser.parse_args()

    settings = get_settings()
    try:
        agent = load_agent(settings)
    except StoreError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"corpus: {len(agent.store.documents)} documents, {len(agent.store)} chunks")
    print(f"top_k : {settings.retrieval_top_k}")
    print(f"token estimate: len(text) / {settings.chars_per_token}\n")

    system_tokens = estimate_tokens(WRITER_SYSTEM, settings.chars_per_token)
    per_task: list[int] = []

    print("Measured prompt size per probe query:")
    for query in PROBE_QUERIES:
        results = await agent.retriever.search(query, top_k=settings.retrieval_top_k)
        evidence = to_evidence(results, max_chars=24_000)
        prompt = build_writer_prompt(query, evidence)
        tokens = system_tokens + estimate_tokens(prompt, settings.chars_per_token)
        per_task.append(tokens)
        print(f"  {tokens:6d} in   {len(evidence)} evidence items   {query}")

    mean_in = statistics.mean(per_task)
    median_in = statistics.median(per_task)
    max_in = max(per_task)

    print(f"\ninput tokens per task : mean {mean_in:.0f}, median {median_in:.0f}, max {max_in}")
    print(f"output tokens per task: {ASSUMED_OUTPUT_TOKENS} (assumed; stub answers are shorter)")

    run_in = mean_in * args.tasks * args.trials
    run_out = ASSUMED_OUTPUT_TOKENS * args.tasks * args.trials
    print(f"\nOne suite run: {args.tasks} tasks x {args.trials} trials")
    print(f"  input  {run_in:12,.0f} tokens")
    print(f"  output {run_out:12,.0f} tokens")

    print(f"\nCost per suite run, and per {args.arms}-arm experiment sweep:")
    print(f"  {'model':26s} {'$/run':>10s} {'$/sweep':>10s}")
    for name, rates in PRICING.items():
        run_cost = run_in / 1e6 * rates["input"] + run_out / 1e6 * rates["output"]
        print(f"  {name:26s} {run_cost:10.2f} {run_cost * args.arms:10.2f}")

    configured = settings.model_id
    configured_rates = PRICING.get(configured)
    if configured_rates:
        run_cost = (
            run_in / 1e6 * configured_rates["input"] + run_out / 1e6 * configured_rates["output"]
        )
        print(f"\nConfigured model is {configured}.")
        print(f"  A full {args.arms}-arm sweep costs about ${run_cost * args.arms:.2f}.")
        print(f"  Ten such sweeps over the project: about ${run_cost * args.arms * 10:.2f}.")

    print(
        "\nCaveats. Token counts are the len/chars_per_token heuristic, not the "
        "tokenizer.\nOutput length is assumed, not measured -- no live model has been "
        "run.\nRetrieval-only experiments (A) call no model and cost nothing."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
