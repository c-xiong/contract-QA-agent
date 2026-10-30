"""Blind human review of a V1 report and a V2 report over the same cases.

    label    interactive terminal labelling of the blind sheet (resumable)
    sheet    one Markdown sheet showing each case's two answers as A/B in seeded order,
             a blank blind-labels file, and the A/B key (do not open it while labelling)
    unblind  validate complete blind labels and write each report's `.labels.jsonl`;
             then freeze each report with `scripts.v2_labels freeze`

    uv run python -m scripts.paired_review sheet reports/v1-standalone-hand-live.json \
        reports/v2-standalone-hand-live.json --output reports/v1v2-hand-live
    uv run python -m scripts.paired_review unblind reports/v1v2-hand-live

Each report's labels are bound to its exact bytes at `sheet` time, exactly as
`scripts.v2_labels sheet` does, so the existing freeze/judge/report steps apply unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, get_args

from pydantic import ValidationError
from scripts.v2_labels import locator
from scripts.v2_labels import paths as report_paths

from app.evidence.claim_support import extract_cited_claims
from evals.graders.v2 import FailureCategory
from evals.loader import load_suite
from evals.paired import MODES, BlindLabel, Mode, assign_slots, check_comparable, unblind
from evals.v2_labels import CompletenessError, Outcome, bind_source_report, sha256

DEFAULT_SEED = 20261006


def paths(prefix: Path) -> dict[str, Path]:
    return {
        name: Path(f"{prefix}.{suffix}")
        for name, suffix in (
            ("sheet", "review.md"),
            ("labels", "blind-labels.jsonl"),
            ("key", "blind-key.json"),
        )
    }


def answer_block(slot: str, row: dict[str, Any]) -> list[str]:
    """One answer with nothing that names the mode: no stop reason, trace or versions."""
    result = row["result"]
    evidence = [
        f"- {e['document_id']} "
        f"{locator(e['page_number'], (e['section_path'] or [None])[-1])}: "
        f"{' '.join(e['excerpt'].split())[:400]}"
        for e in result["evidence"]
    ]
    return [
        f"### Answer {slot}",
        "",
        f"**Run status:** {result['status']} (execution: {row['execution']})",
        "",
        "> " + (result["answer"] or "(empty)").replace("\n", "\n> "),
        "",
        "**Evidence the system admitted:**",
        *(evidence or ["- (none)"]),
        "",
    ]


def sheet(v1_path: Path, v2_path: Path, prefix: Path, seed: int) -> None:
    p = paths(prefix)
    for existing in (
        p["labels"],
        p["key"],
        *(report_paths(r)["labels"] for r in (v1_path, v2_path)),
    ):
        if existing.exists():
            raise SystemExit(f"{existing} exists; labels are never regenerated over reviews.")
    raw = {"v1": v1_path.read_bytes(), "v2": v2_path.read_bytes()}
    reports = {mode: json.loads(data) for mode, data in raw.items()}
    try:
        check_comparable(reports["v1"], reports["v2"])
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    rows = {m: {r["task_id"]: r for r in reports[m]["rows"]} for m in ("v1", "v2")}
    slots = assign_slots(list(rows["v1"]), seed)
    suite = reports["v1"]["provenance"]["frozen_config"]["suite"]
    tasks = {t.task_id: t for t in load_suite(suite)}
    lines = [
        f"# Blind paired review: {prefix.name}",
        "",
        "Each case shows two answers, A and B, from two system variants in a random order.",
        f"Label both in `{p['labels'].name}` (one line per case and slot) before any judge",
        f"verdict is unsealed. Do not open `{p['key'].name}` or the source reports until",
        "`unblind` has run. Label failures as `execution_failure`, never as correct",
        "abstentions. `required_points`/`forbidden_claims` lists follow the order below.",
        "",
    ]
    labels: list[dict[str, Any]] = []
    for task_id in sorted(slots):
        task = tasks[task_id]
        lines += [
            f"## {task_id} ({task.category})",
            "",
            f"**Question:** {task.question}",
            f"**Expected behaviour:** {task.expected_behavior}",
            "",
            "**Required points:** "
            + ("; ".join(f"[{i}] {x}" for i, x in enumerate(task.required_points)) or "none"),
            "**Forbidden claims:** "
            + ("; ".join(f"[{i}] {x}" for i, x in enumerate(task.forbidden_claims)) or "none"),
            "",
            "**Reference evidence:**",
            *[
                f"- `{e.document_id}` {locator(e.page_number, e.section_id)}: {e.span_text}"
                for e in task.expected_evidence
            ],
            "",
        ]
        for slot in ("A", "B"):
            mode: Mode = slots[task_id][slot]
            lines += answer_block(slot, rows[mode][task_id])
            labels.append(BlindLabel(task_id=task_id, slot=slot).model_dump())
    hashes = {mode: hashlib.sha256(data).hexdigest() for mode, data in raw.items()}
    for mode, report in (("v1", v1_path), ("v2", v2_path)):
        bind_source_report(report_paths(report)["labels"], hashes[mode])
    p["key"].write_text(
        json.dumps(
            {
                "seed": seed,
                "reports": {"v1": str(v1_path), "v2": str(v2_path)},
                "report_sha256": hashes,
                "slots": slots,
            },
            indent=2,
        )
        + "\n"
    )
    p["labels"].write_text("".join(json.dumps(label) + "\n" for label in labels))
    p["sheet"].write_text("\n".join(lines) + "\n")
    print(f"Wrote {p['sheet']} and {p['labels']} ({len(labels)} blank labels).")


def unblind_labels(prefix: Path) -> None:
    p = paths(prefix)
    key = json.loads(p["key"].read_text())
    blind = [
        BlindLabel.model_validate_json(line)
        for line in p["labels"].read_text().splitlines()
        if line.strip()
    ]
    try:
        by_mode = unblind(blind, key["slots"])
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    for mode in ("v1", "v2"):
        report = Path(key["reports"][mode])
        if sha256(report) != key["report_sha256"][mode]:
            raise SystemExit(f"{report} changed since the blind sheet was created")
        if report_paths(report)["labels"].exists():
            raise SystemExit(f"{report_paths(report)['labels']} exists; not overwriting labels")
    for mode in MODES:
        target = report_paths(Path(key["reports"][mode]))["labels"]
        target.write_text("".join(label.model_dump_json() + "\n" for label in by_mode[mode]))
        print(f"Wrote {target}; freeze it with `scripts.v2_labels freeze`.")


class Quit(Exception):
    """The reviewer asked to stop; progress is already saved."""


def ask(prompt: str, valid: list[str] | None = None, default: str | None = None) -> str:
    hint = f" [{default}]" if default is not None else ""
    while True:
        value = input(f"{prompt}{hint}: ").strip()
        if value == "q":
            raise Quit
        if not value and default is not None:
            value = default
        if valid is None or value in valid:
            return value
        print(f"  choose one of: {', '.join(valid)} (q to save and quit)")


def choose(prompt: str, options: tuple[str, ...], default: str | None = None) -> str:
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")
    numbers = [str(i) for i in range(1, len(options) + 1)]
    index = default and str(options.index(default) + 1)
    return options[int(ask(prompt, numbers, index or None)) - 1]


def count(prompt: str, default: int, maximum: int) -> int:
    valid = [str(i) for i in range(maximum + 1)]
    return int(ask(f"{prompt} (0-{maximum})", valid, str(min(default, maximum))))


def show(task: Any, slot: str, row: dict[str, Any], done: int, total: int) -> list[Any]:
    wrap = textwrap.TextWrapper(width=96, initial_indent="    ", subsequent_indent="    ")
    result = row["result"]
    print("\n" + "=" * 96)
    print(f"[{done + 1}/{total}] {task.task_id} ({task.category}) - answer {slot}")
    print(f"Question: {task.question}")
    print(f"Expected behaviour: {task.expected_behavior}")
    for i, point in enumerate(task.required_points):
        print(f"  required [{i}] {point}")
    for i, claim in enumerate(task.forbidden_claims):
        print(f"  forbidden [{i}] {claim}")
    print("Reference evidence:")
    for e in task.expected_evidence:
        print(wrap.fill(f"{e.document_id} {locator(e.page_number, e.section_id)}: {e.span_text}"))
    print(f"Run status: {result['status']} (execution: {row['execution']})")
    print("Answer:")
    for line in (result["answer"] or "(empty)").splitlines() or ["(empty)"]:
        print(wrap.fill(line) if line.strip() else "")
    claims = extract_cited_claims(result["answer"] or "") if result["status"] == "completed" else []
    if claims:
        print("Claims (sentence units, as the judge splits them; * = has a citation):")
        for i, claim in enumerate(claims, 1):
            print(wrap.fill(f"{i}{'*' if claim.citations else ' '} {claim.text}"))
    print(f"Admitted evidence: {len(result['evidence'])} items (type e to list)")
    return claims


def label_one(
    task: Any, slot: str, row: dict[str, Any], reviewer: str, done: int, total: int
) -> dict[str, Any]:
    claims = show(task, slot, row, done, total)
    failed = row["result"]["status"] in ("failed", "cancelled") or row["execution"] != "completed"
    outcomes: tuple[str, ...] = get_args(Outcome)
    default = "execution_failure" if failed else None
    while True:
        print("Outcome:")
        try:
            outcome = choose("outcome", outcomes, default)
        except Quit:
            raise
        values: dict[str, Any] = {
            "task_id": task.task_id,
            "slot": slot,
            "reviewer": reviewer,
            "reviewer_type": "author",
            "labeled_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "outcome": outcome,
        }
        if outcome == "execution_failure":
            values |= {"task_success": False, "primary_failure": "infrastructure_failure"}
        else:
            values["required_points"] = [
                {"p": "present", "a": "absent", "c": "contradicted"}[
                    ask(f"required [{i}] {point[:60]} present/absent/contradicted", ["p", "a", "c"])
                ]
                for i, point in enumerate(task.required_points)
            ] or None
            values["forbidden_claims"] = [
                {"m": "made", "n": "not_made"}[
                    ask(f"forbidden [{i}] {claim[:60]} made/not made", ["m", "n"], "n")
                ]
                for i, claim in enumerate(task.forbidden_claims)
            ] or None
            if claims:
                inspected = count("claims inspected", len(claims), len(claims) + 10)
                supported = count("  supported by the admitted evidence", inspected, inspected)
                contradicted = count("  contradicted by it", 0, inspected - supported)
                values |= {
                    "claims_inspected": inspected,
                    "claims_supported": supported,
                    "claims_contradicted": contradicted,
                    "claims_insufficient": inspected - supported - contradicted,
                }
            success = ask("task success? y/n", ["y", "n"]) == "y"
            values["task_success"] = success
            if not success:
                print("Primary failure:")
                values["primary_failure"] = choose("primary failure", get_args(FailureCategory))
                kinds: tuple[str, ...] = get_args(CompletenessError)
                picked = ask(
                    "completeness errors, comma numbers or blank: "
                    + ", ".join(f"{i}={k}" for i, k in enumerate(kinds, 1)),
                    default="",
                )
                values["completeness_errors"] = [
                    kinds[int(x) - 1] for x in picked.split(",") if x.strip().isdigit()
                ]
        values["notes"] = ask("notes (optional)", default="")
        try:
            return BlindLabel.model_validate(values).model_dump(mode="json")
        except ValidationError as exc:
            print(f"  not valid, please redo this answer: {exc.errors()[0]['msg']}")


def label(prefix: Path, reviewer: str) -> None:
    """Interactive labelling of a blind sheet; saves after every answer."""
    p = paths(prefix)
    key = json.loads(p["key"].read_text())  # used only to fetch answers; never printed
    rows = {
        mode: {r["task_id"]: r for r in json.loads(Path(path).read_text())["rows"]}
        for mode, path in key["reports"].items()
    }
    suite = json.loads(Path(key["reports"]["v1"]).read_text())["provenance"]["frozen_config"]
    tasks = {t.task_id: t for t in load_suite(suite["suite"])}
    labels = [json.loads(line) for line in p["labels"].read_text().splitlines() if line.strip()]
    total = len(labels)
    print("Labels save after every answer; type q at any prompt to stop and resume later.")
    try:
        for i, entry in enumerate(labels):
            if entry.get("task_success") is not None:
                continue
            done = sum(e.get("task_success") is not None for e in labels)
            task_id, slot = entry["task_id"], entry["slot"]
            row = rows[key["slots"][task_id][slot]][task_id]
            labels[i] = label_one(tasks[task_id], slot, row, reviewer, done, total)
            temporary = p["labels"].with_suffix(".tmp")
            temporary.write_text("".join(json.dumps(x) + "\n" for x in labels))
            os.replace(temporary, p["labels"])
    except (Quit, KeyboardInterrupt, EOFError):
        pass
    done = sum(e.get("task_success") is not None for e in labels)
    print(f"\n{done}/{total} answers labelled in {p['labels']}.")
    if done == total:
        print(f"Complete. Next: uv run python -m scripts.paired_review unblind {prefix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("sheet")
    make.add_argument("v1_report", type=Path)
    make.add_argument("v2_report", type=Path)
    make.add_argument("--output", type=Path, required=True, help="path prefix for sheet files")
    make.add_argument("--seed", type=int, default=DEFAULT_SEED)
    split = sub.add_parser("unblind")
    split.add_argument("prefix", type=Path)
    interactive = sub.add_parser("label", help="label the blind sheet interactively")
    interactive.add_argument("prefix", type=Path)
    interactive.add_argument("--reviewer", required=True, help="your name, recorded per label")
    args = parser.parse_args()
    if args.command == "sheet":
        sheet(args.v1_report, args.v2_report, args.output, args.seed)
    elif args.command == "label":
        label(args.prefix, args.reviewer)
    else:
        unblind_labels(args.prefix)


if __name__ == "__main__":
    main()
