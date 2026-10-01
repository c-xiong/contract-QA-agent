"""Derive the declared per-answer claim ceiling C from existing live hand-v1 answers.

Counts use the same deterministic `extract_cited_claims` the `claim_support@2` grader
uses, so no model is called. The result is written once and cited by the preflight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from app.evidence.claim_support import CLAIM_SUPPORT_VERSION, extract_cited_claims

SOURCES = (Path("results/eval-hand-bm25.json"), Path("results/eval-hand-rerank.json"))
OUTPUT = Path("evals/datasets/hand/claim_ceiling-v1.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    counts: list[int] = []
    sources = []
    for path in SOURCES:
        report = json.loads(path.read_text())
        answers = [run["answer"] for run in report["runs"]]
        counts += [len(extract_cited_claims(answer)) for answer in answers]
        sources.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "model_id": report["model_id"],
                "suite": report["suite"],
                "answers": len(answers),
            }
        )
    ordered = sorted(counts)
    ceiling = ordered[-1]
    manifest = {
        "grader_version": CLAIM_SUPPORT_VERSION,
        "rule": "C = maximum cited-claim count over all historical live hand-v1 answers, "
        "counted by extract_cited_claims. Answers above C are not truncated; claims beyond C "
        "route to human review and are counted as such.",
        "C": ceiling,
        "answers": len(counts),
        "distribution": dict(sorted(Counter(counts).items())),
        "median": ordered[len(ordered) // 2],
        "sources": sources,
        "limitation": "Historical answers came from the V1 writer; V2 answers may be longer.",
    }
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"C={ceiling} over {len(counts)} answers: {manifest['distribution']}")


if __name__ == "__main__":
    main()
