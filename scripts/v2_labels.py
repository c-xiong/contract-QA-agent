"""Human labeling workflow for a standalone V2 report.

    sheet     write a blank labels file and a Markdown review sheet (no judge verdicts)
    freeze    validate complete labels and record their hash; required before unsealing
    agreement unseal judge verdicts and compare them with frozen labels
    report    final report: human-label quality, deterministic metrics, judge agreement

    uv run python -m scripts.v2_labels sheet reports/v2-standalone-hand-live.json
    uv run python -m scripts.v2_labels freeze reports/v2-standalone-hand-live.json
    uv run python -m scripts.v2_labels agreement reports/v2-standalone-hand-live.json \
        --sealed reports/v2-standalone-hand-live.judge-frozen.sealed.jsonl
    uv run python -m scripts.v2_labels report reports/v2-standalone-hand-live.json \
        --sealed reports/v2-standalone-hand-live.judge-frozen.sealed.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.loader import load_suite
from evals.v2_labels import (
    HumanLabel,
    agreement,
    apply_adjudications,
    bind_source_report,
    check_frozen,
    freeze,
    load_labels,
    load_sealed,
    quality,
    sha256,
)

SPLIT = Path("evals/datasets/hand/judge_split-v1.json")


def paths(report: Path) -> dict[str, Path]:
    return {
        "labels": report.with_suffix(".labels.jsonl"),
        "sheet": report.with_suffix(".review.md"),
        "adjudications": report.with_suffix(".adjudications.jsonl"),
        "unseal_log": report.with_suffix(".unseal-log.jsonl"),
        "final": report.with_suffix(".final.json"),
    }


def locator(page: int | None, section: str | None) -> str:
    parts = ([f"p. {page}"] if page is not None else []) + ([f"§{section}"] if section else [])
    return ", ".join(parts) or "(no locator)"


def sheet(report_path: Path) -> None:
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    p = paths(report_path)
    if p["labels"].exists():
        raise SystemExit(f"{p['labels']} exists; labels are never regenerated over reviews.")
    tasks = {t.task_id: t for t in load_suite(report["provenance"]["frozen_config"]["suite"])}
    labels, lines = [], [f"# Review sheet: {report_path.name}", ""]
    lines += [
        "Label every case in the `.labels.jsonl` file before any judge verdict is unsealed.",
        "Do not open `*.sealed.jsonl`. Label failures as `execution_failure`, never as",
        "correct abstentions. `required_points`/`forbidden_claims` lists follow the order below.",
        f"Source report SHA-256: `{report_sha256}`. Keep the source report unchanged.",
        "",
    ]
    for row in report["rows"]:
        task, result = tasks[row["task_id"]], row["result"]
        labels.append(HumanLabel(task_id=row["task_id"]).model_dump())
        lines += [
            f"## {row['task_id']} ({row['category']}, split: {row['split']})",
            "",
            f"**Question:** {task.question}",
            f"**Expected behaviour:** {task.expected_behavior}",
            "",
            "**Required points:** "
            + ("; ".join(f"[{i}] {p}" for i, p in enumerate(task.required_points)) or "none"),
            "**Forbidden claims:** "
            + ("; ".join(f"[{i}] {c}" for i, c in enumerate(task.forbidden_claims)) or "none"),
            "",
            "**Reference evidence:**",
            *[
                f"- `{e.document_id}` {locator(e.page_number, e.section_id)}: {e.span_text}"
                for e in task.expected_evidence
            ],
            "",
            f"**Run status:** {result['status']} (stop: {result['stop_reason']}, "
            f"execution: {row['execution']})",
            "",
            "**Delivered answer:**",
            "",
            "> " + (result["answer"] or "(empty)").replace("\n", "\n> "),
            "",
            "**Evidence the system admitted:**",
            *[
                f"- `{e['evidence_id']}` {e['document_id']} "
                f"{locator(e['page_number'], (e['section_path'] or [None])[-1])}: {' '.join(e['excerpt'].split())[:400]}"
                for e in result["evidence"]
            ],
            "",
        ]
    bind_source_report(p["labels"], report_sha256)
    p["labels"].write_text("".join(json.dumps(label) + "\n" for label in labels))
    p["sheet"].write_text("\n".join(lines) + "\n")
    print(f"Wrote {p['labels']} ({len(labels)} blank labels) and {p['sheet']}.")


def unseal(report_path: Path, sealed: Path, purpose: str) -> tuple[dict[str, Any], Any]:
    p = paths(report_path)
    record = check_frozen(p["labels"], sha256(report_path))
    header, verdicts = load_sealed(sealed, record["report_sha256"])
    with p["unseal_log"].open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "sealed": str(sealed),
                    "purpose": purpose,
                    "unsealed_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "labels_frozen_at": record["frozen_at"],
                    "labels_sha256": record["labels_sha256"],
                    "report_sha256": record["report_sha256"],
                }
            )
            + "\n"
        )
    return header, verdicts


def split_ids() -> dict[str, list[str]]:
    manifest = json.loads(SPLIT.read_text())
    return {"calibration": manifest["calibration"], "audit": manifest["audit"]}


def compare(report_path: Path, sealed: Path) -> dict[str, Any]:
    p = paths(report_path)
    header, verdicts = unseal(report_path, sealed, "agreement")
    labels = apply_adjudications(load_labels(p["labels"]), p["adjudications"])
    ids = split_ids()
    present = set(verdicts)
    result = {
        "judge": header,
        "calibration_in_sample": agreement(
            labels, verdicts, [t for t in ids["calibration"] if t in present]
        ),
    }
    if header["phase"] == "frozen":
        result["audit_primary"] = agreement(labels, verdicts, ids["audit"])
    return result


def final_report(report_path: Path, sealed: Path | None) -> None:
    p = paths(report_path)
    report_bytes = report_path.read_bytes()
    frozen = check_frozen(p["labels"], hashlib.sha256(report_bytes).hexdigest())
    report = json.loads(report_bytes)
    labels = apply_adjudications(load_labels(p["labels"]), p["adjudications"])
    output: dict[str, Any] = {
        "provenance": report["provenance"],
        "labels": frozen | {"adjudications": p["adjudications"].exists()},
        "answer_quality_human_labels": quality(labels, report),
        "deterministic_diagnostics": report["summary"],
        "unexecuted": report["unexecuted"],
        "application_budget": report["budget_ledger"],
        "judge_agreement": compare(report_path, sealed) if sealed else "not run",
        "limitations": [
            "hand-v1 is a 41-case development regression set, not an unseen test set.",
            "Labels are "
            + (
                "author-reviewed"
                if set(frozen["reviewer_types"]) == {"author"}
                else "from mixed/independent reviewers"
            )
            + "; see reviewer_types.",
            "Judge agreement on calibration cases is in-sample; the 21-case audit is primary.",
            "The author saw historical hand-v1 outputs; the judge audit is not blind.",
            "Runtime and judge models share a model family; separate prompts do not imply "
            "independent errors.",
            "Single trial; no confidence intervals are claimed.",
        ],
        "reporting_eligible": report["provenance"]["reporting_eligible"],
    }
    p["final"].write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {p['final']}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["sheet", "freeze", "agreement", "report"])
    parser.add_argument("report", type=Path)
    parser.add_argument("--sealed", type=Path)
    args = parser.parse_args()
    if args.command == "sheet":
        sheet(args.report)
    elif args.command == "freeze":
        record = freeze(paths(args.report)["labels"], args.report)
        print(f"Froze {record['task_count']} labels at {record['frozen_at']}.")
    elif args.command == "agreement":
        if args.sealed is None:
            raise SystemExit("--sealed is required")
        print(json.dumps(compare(args.report, args.sealed), indent=2))
    else:
        final_report(args.report, args.sealed)


if __name__ == "__main__":
    main()
