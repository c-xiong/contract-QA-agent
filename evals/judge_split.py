"""Frozen calibration/audit assignment of hand-v1 cases for evaluating the LLM judge.

DECISION: the split is made from task metadata only, before any live V2 answer or judge
  verdict exists, and checked in. Stratifying by `category` keeps every category on both
  sides where possible. Alternative rejected: choosing calibration cases after seeing
  outputs "for coverage", which would let knowledge of judge behaviour shape the audit.

This split evaluates the judge. It does not make hand-v1 a held-out QA test set, and it
cannot undo the author's earlier exposure to historical hand-v1 outputs.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evals.schema import EvalTask

SPLIT_VERSION = "judge-split-v1"
SEED = 20260928
CALIBRATION_SIZE = 20
RULE = (
    "Group by category; shuffle each group's sorted task_ids with random.Random(seed). "
    "Even groups split evenly; odd groups give the extra case to audit. Then move cases "
    "from audit to calibration by one seeded draw over the sorted audit ids until "
    "calibration has 20 cases."
)


def split(tasks: list[EvalTask], seed: int = SEED) -> dict[str, list[str]]:
    rng = random.Random(seed)
    groups: dict[str, list[str]] = defaultdict(list)
    for task in tasks:
        groups[task.category].append(task.task_id)
    calibration: list[str] = []
    audit: list[str] = []
    for category in sorted(groups):
        ids = sorted(groups[category])
        rng.shuffle(ids)
        half = len(ids) // 2
        calibration += ids[:half]
        audit += ids[half:]
    if len(calibration) > CALIBRATION_SIZE:
        raise ValueError("stratified floor already exceeds the calibration size")
    while len(calibration) < CALIBRATION_SIZE:
        moved = rng.choice(sorted(audit))
        audit.remove(moved)
        calibration.append(moved)
    return {"calibration": sorted(calibration), "audit": sorted(audit)}


def build_manifest(tasks: list[EvalTask], dataset_path: Path, seed: int = SEED) -> dict[str, Any]:
    assignment = split(tasks, seed)
    by_id = {t.task_id: t for t in tasks}

    def counts(ids: list[str], field: str) -> dict[str, int]:
        return dict(sorted(Counter(getattr(by_id[i], field) for i in ids).items()))

    return {
        "split_version": SPLIT_VERSION,
        "dataset_version": tasks[0].dataset_version,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "task_count": len(tasks),
        "seed": seed,
        "rule": RULE,
        "made_before_live_v2_outputs": True,
        "limitations": [
            "Evaluates the judge only; hand-v1 remains a development regression set.",
            "The author inspected historical hand-v1 outputs before this split; audit "
            "results are not a held-out or blind validation of the judge.",
        ],
        "calibration": assignment["calibration"],
        "audit": assignment["audit"],
        "balance": {
            side: {
                "n": len(assignment[side]),
                "category": counts(assignment[side], "category"),
                "expected_behavior": counts(assignment[side], "expected_behavior"),
            }
            for side in ("calibration", "audit")
        },
    }
