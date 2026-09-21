"""Aggregate completed human labels; never infer semantic labels from runtime output."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evals.graders.v2 import ManualLabel, ratio


def score(report: dict[str, Any], labels: list[ManualLabel]) -> dict[str, Any]:
    indexed = {(label.task_id, label.mode): label for label in labels}
    if len(indexed) != len(labels):
        raise ValueError("Duplicate manual labels")
    expected = {(row["task_id"], row["mode"]) for row in report["rows"]}
    if set(indexed) != expected:
        raise ValueError("Labels must cover exactly every run, including failures")
    for row in report["rows"]:
        label = indexed[row["task_id"], row["mode"]]
        if label.task_success is None:
            raise ValueError(f"Unreviewed task {label.task_id}/{label.mode}")
        if row["diagnostics"]["status"] in ("failed", "cancelled") and label.task_success:
            raise ValueError("Infrastructure failures cannot be scored as successful abstentions")
    summary: dict[str, Any] = {}
    for mode in ("v1", "v2"):
        mode_rows = [r for r in report["rows"] if r["mode"] == mode]
        summary[mode] = {}
        for category in ["all", *sorted({r["category"] for r in mode_rows})]:
            subset = [
                indexed[r["task_id"], mode]
                for r in mode_rows
                if category == "all" or r["category"] == category
            ]
            semantic = [label for label in subset if label.inspected_citations is not None]
            useful = [label for label in subset if label.inspected_calls is not None]
            summary[mode][category] = {
                "task_success": ratio(
                    sum(label.task_success is True for label in subset), len(subset)
                ),
                "semantic_citation_support": ratio(
                    sum(label.supported_citations or 0 for label in semantic),
                    sum(label.inspected_citations or 0 for label in semantic),
                ),
                "tool_usefulness": ratio(
                    sum(label.useful_calls or 0 for label in useful),
                    sum(label.inspected_calls or 0 for label in useful),
                ),
                "primary_failures": dict(
                    Counter(label.primary_failure for label in subset if label.primary_failure)
                ),
            }
    pairs = [
        (indexed[t, "v1"].task_success, indexed[t, "v2"].task_success)
        for t in sorted({t for t, _ in expected})
    ]
    return {
        "reporting_eligible": report["provenance"]["reporting_eligible"],
        "summary": summary,
        "paired_success": {
            "wins": sum(a is False and b is True for a, b in pairs),
            "losses": sum(a is True and b is False for a, b in pairs),
            "ties": sum(a == b for a, b in pairs),
            "n": len(pairs),
        },
        "warning": "Provisional or stub runs remain pipeline checks even after labels are supplied.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    labels = [
        ManualLabel.model_validate_json(line)
        for line in args.labels.read_text().splitlines()
        if line.strip()
    ]
    args.output.write_text(json.dumps(score(report, labels), indent=2) + "\n")


if __name__ == "__main__":
    main()
