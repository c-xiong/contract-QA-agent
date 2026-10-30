"""Paired V1/V2 comparison over two standalone reports and their frozen human labels.

DECISION: the comparison pairs two independent standalone runs (`--mode v1`, `--mode v2`)
  rather than re-introducing an interleaved paired runner. Each arm keeps its own ledger,
  resume, label freeze and judge record; pairing happens only after both label sets are
  frozen. Alternative rejected: one runner executing both modes per case, which would
  give the arms a shared budget and a shared failure/resume history.

DECISION: human labels are collected blind to mode where the output allows it. One
  review sheet shows each case's two answers as A/B in a seeded order; the key is kept in
  a separate file. Style, V2's fixed abstention wording, evidence lists and run status can
  still reveal the mode; hiding answer text would stop the reviewer judging it. The
  blinding is therefore partial and is disclosed as such.

DECISION: the headline paired outcome is human task success, reported as V2 win / loss /
  tie with its discordant pairs. An exact sign test and a paired bootstrap interval are
  included as descriptive context only: one trial, 41 development cases, author labels.
"""

from __future__ import annotations

import random
from collections import Counter
from math import comb
from statistics import median
from typing import Any, Literal

from evals.graders.v2 import ratio
from evals.v2_labels import HumanLabel, quality

Mode = Literal["v1", "v2"]
Slot = Literal["A", "B"]
MODES: tuple[Mode, Mode] = ("v1", "v2")
# Only these frozen-config keys may differ between the two arms.
TREATMENT_KEYS = frozenset({"agent_mode", "planner_version"})
BOOTSTRAP_RESAMPLES = 10_000
COST_KEYS = ("model_calls", "input_tokens", "output_tokens", "elapsed_seconds", "tool_calls")
PAIRED_DIAGNOSTICS = (
    "evidence_recall_at_5",
    "trajectory_evidence_recall",
    "answer_abstain_decision",
    "structural_gate_pass",
)


class BlindLabel(HumanLabel):
    """A human label written against slot A or B of a blind sheet, before unblinding."""

    slot: Slot


def check_comparable(v1: dict[str, Any], v2: dict[str, Any]) -> dict[str, Any]:
    """Refuse pairs that differ in anything other than the agent mode, or are incomplete."""
    c1, c2 = v1["provenance"]["frozen_config"], v2["provenance"]["frozen_config"]
    if c1.get("agent_mode") != "v1" or c2.get("agent_mode") != "v2":
        raise ValueError("Expected a --mode v1 report and a --mode v2 report, in that order")
    differing = sorted(k for k in c1.keys() | c2.keys() if c1.get(k) != c2.get(k))
    confounded = [k for k in differing if k not in TREATMENT_KEYS]
    if confounded:
        raise ValueError(f"Arms differ outside the treatment: {', '.join(confounded)}")
    ids1 = [r["task_id"] for r in v1["rows"]]
    ids2 = [r["task_id"] for r in v2["rows"]]
    if sorted(ids1) != sorted(ids2) or len(set(ids1)) != len(ids1):
        raise ValueError("The two reports must contain the same cases exactly once")
    for name, report in (("v1", v1), ("v2", v2)):
        if report["unexecuted"]:
            raise ValueError(f"The {name} run has unexecuted cases; resume it before pairing")
    return {
        "differing_keys": differing,
        "shared_config": {k: c1[k] for k in sorted(c1) if k not in differing},
        "v1_config": {k: c1.get(k) for k in differing},
        "v2_config": {k: c2.get(k) for k in differing},
        "claude_code_version": {
            "v1": v1["provenance"].get("claude_code_version"),
            "v2": v2["provenance"].get("claude_code_version"),
        },
    }


def assign_slots(task_ids: list[str], seed: int) -> dict[str, dict[Slot, Mode]]:
    rng = random.Random(seed)
    slots: dict[str, dict[Slot, Mode]] = {}
    for task_id in sorted(task_ids):
        first: Mode = "v1" if rng.random() < 0.5 else "v2"
        slots[task_id] = {"A": first, "B": "v2" if first == "v1" else "v1"}
    return slots


def unblind(
    blind: list[BlindLabel], slots: dict[str, dict[Slot, Mode]]
) -> dict[Mode, list[HumanLabel]]:
    """Split complete blind labels into one ordered label list per mode."""
    keys = [(label.task_id, label.slot) for label in blind]
    expected = [(task_id, slot) for task_id in sorted(slots) for slot in ("A", "B")]
    if sorted(keys) != expected or len(set(keys)) != len(keys):
        raise ValueError("Blind labels must cover every case's A and B answer exactly once")
    incomplete = [f"{label.task_id}/{label.slot}" for label in blind if not label.complete]
    if incomplete:
        raise ValueError(f"Unlabelled answers: {', '.join(incomplete)}")
    by_mode: dict[Mode, dict[str, HumanLabel]] = {"v1": {}, "v2": {}}
    for label in blind:
        values = label.model_dump(exclude={"slot"})
        by_mode[slots[label.task_id][label.slot]][label.task_id] = HumanLabel.model_validate(values)
    return {mode: [labels[t] for t in sorted(labels)] for mode, labels in by_mode.items()}


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided sign test on discordant pairs; 1.0 when there are none."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail: float = sum(comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def bootstrap_difference(
    pairs: list[tuple[bool, bool]], *, seed: int, resamples: int = BOOTSTRAP_RESAMPLES
) -> dict[str, Any]:
    """Percentile interval for the paired success-rate difference (V2 minus V1)."""
    if not pairs:
        return {"difference": None, "interval_95": None, "resamples": 0, "seed": seed}
    diffs = [int(b) - int(a) for a, b in pairs]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(resamples))
    low, high = means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]
    return {
        "difference": sum(diffs) / n,
        "interval_95": [low, high],
        "resamples": resamples,
        "seed": seed,
    }


def wins_losses(pairs: list[tuple[Any, Any]]) -> dict[str, int]:
    """V2 win/loss/tie over pairs whose two values are both known (higher is better)."""
    known = [(a, b) for a, b in pairs if a is not None and b is not None]
    return {
        "v2_wins": sum(b > a for a, b in known),
        "v2_losses": sum(b < a for a, b in known),
        "ties": sum(a == b for a, b in known),
        "n": len(known),
        "excluded_unknown": len(pairs) - len(known),
    }


def claim_rate(label: HumanLabel) -> float | None:
    if not label.claims_inspected:
        return None
    return (label.claims_supported or 0) / label.claims_inspected


def _label_view(label: HumanLabel) -> dict[str, Any]:
    return {
        "outcome": label.outcome,
        "task_success": label.task_success,
        "primary_failure": label.primary_failure,
        "notes": label.notes,
    }


def _cost(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows)}
    for key in COST_KEYS:
        values = [r["diagnostics"][key] for r in rows if r["diagnostics"][key] is not None]
        out[f"median_{key}"] = median(values) if values else None
        out[f"{key}_unknown"] = len(rows) - len(values)
    return out


def compare(
    v1: dict[str, Any],
    v2: dict[str, Any],
    labels: dict[Mode, list[HumanLabel]],
    *,
    seed: int,
) -> dict[str, Any]:
    """Paired comparison from frozen human labels and deterministic diagnostics."""
    comparability = check_comparable(v1, v2)
    reports = {"v1": v1, "v2": v2}
    rows = {m: {r["task_id"]: r for r in reports[m]["rows"]} for m in MODES}
    by_id = {m: {label.task_id: label for label in labels[m]} for m in MODES}
    task_ids = sorted(rows["v1"])
    for mode in MODES:
        if sorted(by_id[mode]) != task_ids:
            raise ValueError(f"{mode} labels do not cover the paired cases exactly")
    categories = {t: rows["v1"][t]["category"] for t in task_ids}

    def success_block(ids: list[str]) -> dict[str, Any]:
        pairs = [(by_id["v1"][t].task_success, by_id["v2"][t].task_success) for t in ids]
        counts = wins_losses(pairs)
        both = sum(bool(a) and bool(b) for a, b in pairs)
        return counts | {
            "both_succeed": both,
            "both_fail": counts["ties"] - both,
            "v1_success": ratio(sum(bool(a) for a, _ in pairs), len(pairs)),
            "v2_success": ratio(sum(bool(b) for _, b in pairs), len(pairs)),
        }

    overall = success_block(task_ids)
    success_pairs = [
        (bool(by_id["v1"][t].task_success), bool(by_id["v2"][t].task_success)) for t in task_ids
    ]
    overall["sign_test_p_two_sided"] = sign_test(overall["v2_wins"], overall["v2_losses"])
    overall["bootstrap"] = bootstrap_difference(success_pairs, seed=seed)

    transitions = Counter(f"{by_id['v1'][t].outcome} -> {by_id['v2'][t].outcome}" for t in task_ids)

    def case(t: str) -> dict[str, Any]:
        return {
            "task_id": t,
            "category": categories[t],
            "split": rows["v2"][t].get("split"),
            "v1": _label_view(by_id["v1"][t]),
            "v2": _label_view(by_id["v2"][t]),
        }

    wins = [
        case(t) for t in task_ids if by_id["v2"][t].task_success and not by_id["v1"][t].task_success
    ]
    losses = [
        case(t) for t in task_ids if by_id["v1"][t].task_success and not by_id["v2"][t].task_success
    ]
    both_fail = [
        case(t)
        for t in task_ids
        if not by_id["v1"][t].task_success and not by_id["v2"][t].task_success
    ]

    diagnostics = {
        key: wins_losses(
            [
                (rows["v1"][t]["diagnostics"][key], rows["v2"][t]["diagnostics"][key])
                for t in task_ids
            ]
        )
        for key in PAIRED_DIAGNOSTICS
    }
    cost_differences: dict[str, Any] = {}
    for key in COST_KEYS:
        deltas = [
            rows["v2"][t]["diagnostics"][key] - rows["v1"][t]["diagnostics"][key]
            for t in task_ids
            if rows["v1"][t]["diagnostics"][key] is not None
            and rows["v2"][t]["diagnostics"][key] is not None
        ]
        cost_differences[key] = {
            "median_v2_minus_v1": median(deltas) if deltas else None,
            "n": len(deltas),
        }

    return {
        "task_count": len(task_ids),
        "trial_count": 1,
        "comparability": comparability,
        "reporting_eligible": bool(
            v1["provenance"].get("reporting_eligible")
            and v2["provenance"].get("reporting_eligible")
        ),
        "human_quality": {m: quality(labels[m], reports[m]) for m in MODES},
        "paired_task_success": {
            "all": overall,
            **{
                f"category:{c}": success_block([t for t in task_ids if categories[t] == c])
                for c in sorted(set(categories.values()))
            },
        },
        "outcome_transitions": dict(sorted(transitions.items())),
        "paired_claim_support_rate": wins_losses(
            [(claim_rate(by_id["v1"][t]), claim_rate(by_id["v2"][t])) for t in task_ids]
        ),
        "paired_diagnostics": diagnostics,
        "primary_failures": {
            m: dict(Counter(x.primary_failure for x in labels[m] if x.primary_failure))
            for m in MODES
        },
        "cost": {
            "v1": _cost(list(rows["v1"].values())),
            "v2": _cost(list(rows["v2"].values())),
            "paired_differences": cost_differences,
        },
        "cases": {"v2_wins": wins, "v2_losses": losses, "both_fail": both_fail},
        "budgets": {m: reports[m]["budget_ledger"] for m in MODES},
    }
