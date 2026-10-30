"""Run the offline judges over a standalone V2 report; verdicts are sealed from reviewers.

    # calibration rounds (at most two) on the 20 calibration cases
    CRA_LIVE_MODEL=1 uv run python -m experiments.v2_judge reports/v2-standalone-hand-live.json \
        --phase calibration --round 1 --preflight evals/preflight/v2-hand-judge.json
    # one frozen pass on all 41 cases, after the judge prompts are frozen
    CRA_LIVE_MODEL=1 uv run python -m experiments.v2_judge reports/v2-standalone-hand-live.json \
        --phase frozen --preflight evals/preflight/v2-hand-judge.json

Offline, both judges return `unknown` without calling a model: a plumbing check only.

DECISION: verdicts go to a `.sealed.jsonl` file and this script prints only counts.
  Human labels must be produced and frozen before `scripts/v2_judge_agreement.py`
  unseals them. This is a procedural control, not cryptography: a reviewer who opens
  the sealed file has broken the protocol and must disclose it.
"""

# ruff: noqa: ASYNC240 -- sequential local artifact I/O between awaited runs; no concurrency

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.agent.llm import (
    MAX_OUTPUT_TOKENS,
    ClaudeCodeClient,
    ModelClient,
    StubClient,
    build_client,
)
from app.config import Settings
from app.evidence.claim_support import CLAIM_SUPPORT_SYSTEM, CLAIM_SUPPORT_VERSION, check_answer
from app.schemas.evidence import Evidence
from evals.budget import BudgetedClient, Ledger, load_preflight
from evals.graders.answer_rubric import (
    ANSWER_RUBRIC_AUTHOR_REVIEWED,
    ANSWER_RUBRIC_SYSTEM,
    ANSWER_RUBRIC_VERSION,
    judge_answer,
)
from evals.loader import load_suite

CLAIM_CEILING = Path("evals/datasets/hand/claim_ceiling-v1.json")
MAX_CALIBRATION_ROUNDS = 2
PROVIDER_LIMIT = "provider_limit"


def prompt_hashes() -> dict[str, str]:
    return {
        CLAIM_SUPPORT_VERSION: hashlib.sha256(CLAIM_SUPPORT_SYSTEM.encode()).hexdigest(),
        ANSWER_RUBRIC_VERSION: hashlib.sha256(ANSWER_RUBRIC_SYSTEM.encode()).hexdigest(),
    }


def sealed_path(report: Path, phase: str, round_: int | None) -> Path:
    suffix = f".judge-{phase}" + (f"-r{round_}" if round_ else "") + ".sealed.jsonl"
    return report.with_suffix(suffix)


async def judge_row(
    row: dict[str, Any], task: Any, client: ModelClient, *, live: bool, ceiling: int
) -> dict[str, Any]:
    result = row["result"]
    base = {
        "task_id": row["task_id"],
        "split": row["split"],
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if row["execution"] != "completed" or result["status"] in ("failed", "cancelled"):
        # Failures are never judged into correct abstentions; humans label them directly.
        return base | {"not_judged": "execution_failure", "status": result["status"]}
    evidence = [Evidence.model_validate(e) for e in result["evidence"]]
    claims = []
    if result["status"] != "abstained":
        verdicts = await check_answer(
            result["answer"], evidence, client, live=live, max_model_claims=ceiling
        )
        claims = [asdict(v) for v in verdicts]
    rubric = (
        await judge_answer(task, result["answer"], client)
        if live
        else {"verdict": "unknown", "reason": "offline: judge requires CRA_LIVE_MODEL=1"}
    )
    return base | {"status": result["status"], "claims": claims, "answer_rubric": rubric}


async def run(args: argparse.Namespace) -> Path:
    settings = Settings()
    report_bytes = args.report.read_bytes()
    report = json.loads(report_bytes)
    ceiling = json.loads(CLAIM_CEILING.read_text())["C"]
    if args.phase == "calibration":
        if args.round not in range(1, MAX_CALIBRATION_ROUNDS + 1):
            raise SystemExit(f"--round must be 1..{MAX_CALIBRATION_ROUNDS}")
        rows = [r for r in report["rows"] if r["split"] == "calibration"]
    else:
        if args.round is not None:
            raise SystemExit("The frozen pass has no rounds.")
        if not ANSWER_RUBRIC_AUTHOR_REVIEWED and settings.live_model:
            raise SystemExit(
                "The answer-rubric prompt is still provisional (see TODO(author) in "
                "evals/graders/answer_rubric.py); the frozen pass needs an author-reviewed judge."
            )
        rows = list(report["rows"])
    if not rows:
        raise SystemExit("No rows for this phase; is this a hand-v1 standalone report?")
    output = sealed_path(args.report, args.phase, args.round)
    if output.exists() and not args.resume:
        raise SystemExit(f"{output} exists; judge passes are append-only records.")

    client: ModelClient = StubClient()
    ledger = None
    if settings.live_model:
        if args.preflight is None:
            raise SystemExit("A live judge run requires --preflight (phase: judge).")
        preflight = load_preflight(args.preflight)
        if preflight.phase != "judge":
            raise SystemExit("This runner needs a judge-phase preflight.")
        if preflight.max_output_tokens_per_call < MAX_OUTPUT_TOKENS:
            raise SystemExit("max_output_tokens_per_call must cover the client's ceiling")
        # One ledger for every judge phase of this report: the ceiling is cumulative.
        ledger = Ledger(preflight, args.report.with_suffix(".judge-ledger.json"))
        client = BudgetedClient(build_client(settings), ledger)

    tasks = {t.task_id: t for t in load_suite(report["provenance"]["frozen_config"]["suite"])}
    header = {
        "record": "header",
        "phase": args.phase,
        "round": args.round,
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "claim_ceiling": ceiling,
        "prompt_sha256": prompt_hashes(),
        "judge_model": settings.model_id if settings.live_model else "stub",
        "judge_backend": settings.model_backend if settings.live_model else "stub",
        "answer_rubric_author_reviewed": ANSWER_RUBRIC_AUTHOR_REVIEWED,
        "live": settings.live_model,
    }
    inner = client.inner if isinstance(client, BudgetedClient) else client
    cli = inner if isinstance(inner, ClaudeCodeClient) else None
    done: set[str] = set()
    if output.exists():
        # Resume after a provider limit: same header, then append the cases not yet judged.
        # Records are never rewritten; readers keep the last record per task.
        lines = output.read_text().splitlines()
        if json.loads(lines[0]) != header:
            raise SystemExit("Judge configuration or source report changed; not resuming.")
        done = {
            r["task_id"]
            for r in map(json.loads, lines[1:])
            if r.get("not_judged") != PROVIDER_LIMIT
        }
    stopped = 0
    with output.open("a") as handle:
        if not done and output.stat().st_size == 0:
            handle.write(json.dumps(header) + "\n")
        for row in rows:
            if row["task_id"] in done:
                continue
            if cli and cli.limit_reached:
                stopped += 1
                handle.write(
                    json.dumps({"task_id": row["task_id"], "not_judged": PROVIDER_LIMIT}) + "\n"
                )
                continue
            if ledger and ledger.exhausted:
                stopped += 1
                handle.write(json.dumps({"task_id": row["task_id"], "not_judged": "budget"}) + "\n")
                continue
            verdict = await judge_row(
                row, tasks[row["task_id"]], client, live=settings.live_model, ceiling=ceiling
            )
            if cli and cli.limit_reached:
                # Calls in this row hit the limit: re-judge it on resume, never keep unknowns.
                verdict = {"task_id": row["task_id"], "not_judged": PROVIDER_LIMIT}
                stopped += 1
            handle.write(json.dumps(verdict) + "\n")
            handle.flush()
    latest = {r["task_id"]: r for r in map(json.loads, output.read_text().splitlines()[1:])}
    records = list(latest.values())
    outcomes = Counter(r.get("not_judged") or r["answer_rubric"]["verdict"] for r in records)
    print(
        f"Sealed {len(records)} verdicts to {output}: {dict(outcomes)}"
        + (
            f"; ledger {ledger.state.attempts} attempts, cost {ledger.state.cost:.4f}"
            if ledger
            else ""
        )
        + ". Do not open it before labels are frozen."
        + (
            f" {stopped} cases hit the provider limit; rerun with --resume later."
            if cli and cli.limit_reached
            else ""
        )
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", type=Path)
    parser.add_argument("--phase", choices=["calibration", "frozen"], required=True)
    parser.add_argument("--round", type=int)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument(
        "--resume", action="store_true", help="continue a pass stopped by a provider limit"
    )
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
