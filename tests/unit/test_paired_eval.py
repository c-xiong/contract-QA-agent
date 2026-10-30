"""Paired V1/V2 evaluation: V1 standalone runs, blind review, and the paired report."""

# ruff: noqa: ASYNC240 -- small local artifact reads inside an async test

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evals.paired import (
    BlindLabel,
    assign_slots,
    bootstrap_difference,
    check_comparable,
    compare,
    sign_test,
    unblind,
)
from evals.v2_labels import HumanLabel
from tests.unit.test_v2_eval_pipeline import corpus, label, runner_args

# --- fixtures ----------------------------------------------------------------------


def config(mode: str, **overrides: Any) -> dict[str, Any]:
    return {
        "suite": "hand",
        "agent_mode": mode,
        "planner_version": "v2-planner-1" if mode == "v2" else None,
        "retrieval_arm": "bm25",
        "model_id": "stub",
        "live_model": False,
        "dataset_sha256": "d",
        "index_hash": "i",
    } | overrides


def diagnostics(**overrides: Any) -> dict[str, Any]:
    return {
        "answerable": True,
        "evidence_recall_at_5": 0.5,
        "trajectory_evidence_recall": 0.5,
        "answer_abstain_decision": True,
        "structural_gate_pass": True,
        "model_calls": 1,
        "input_tokens": 100,
        "output_tokens": 10,
        "elapsed_seconds": 1.0,
        "tool_calls": None,
    } | overrides


def report(mode: str, ids: list[str], **config_overrides: Any) -> dict[str, Any]:
    return {
        "provenance": {"frozen_config": config(mode, **config_overrides)},
        "unexecuted": [],
        "budget_ledger": "offline: no provider calls",
        "rows": [
            {
                "task_id": t,
                "category": "lookup" if t != "c" else "crossref",
                "split": "audit",
                "diagnostics": diagnostics(tool_calls=3 if mode == "v2" else None),
            }
            for t in ids
        ],
    }


# --- comparability -------------------------------------------------------------------


def test_comparable_reports_differ_only_in_the_treatment() -> None:
    result = check_comparable(report("v1", ["a", "b"]), report("v2", ["b", "a"]))
    assert result["differing_keys"] == ["agent_mode", "planner_version"]
    assert result["shared_config"]["retrieval_arm"] == "bm25"


@pytest.mark.parametrize(
    "v1,v2,message",
    [
        (report("v2", ["a"]), report("v1", ["a"]), "in that order"),
        (report("v1", ["a"]), report("v2", ["a"], retrieval_arm="dense"), "retrieval_arm"),
        (report("v1", ["a"]), report("v2", ["a"], model_id="other"), "model_id"),
        (report("v1", ["a"]), report("v2", ["a", "b"]), "same cases"),
        (
            report("v1", ["a"]) | {"unexecuted": [{"task_id": "b"}]},
            report("v2", ["a"]),
            "unexecuted",
        ),
    ],
)
def test_incomparable_reports_are_refused(
    v1: dict[str, Any], v2: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        check_comparable(v1, v2)


# --- blinding ------------------------------------------------------------------------


def test_slots_are_seeded_and_cover_both_modes() -> None:
    ids = [f"t{i}" for i in range(40)]
    slots = assign_slots(ids, seed=7)
    assert slots == assign_slots(list(reversed(ids)), seed=7)
    assert all(sorted(s.values()) == ["v1", "v2"] for s in slots.values())
    first = [s["A"] for s in slots.values()]
    assert 0 < first.count("v1") < len(ids)  # order is actually randomised


def blind(task_id: str, slot: str, success: bool, **overrides: Any) -> BlindLabel:
    return BlindLabel.model_validate(
        label(task_id, success).model_dump() | {"slot": slot} | overrides
    )


def test_unblind_routes_each_slot_to_its_mode() -> None:
    slots = {"a": {"A": "v2", "B": "v1"}}
    by_mode = unblind([blind("a", "A", True), blind("a", "B", False)], slots)  # type: ignore[arg-type]
    assert by_mode["v2"][0].task_success is True and by_mode["v1"][0].task_success is False
    assert isinstance(by_mode["v1"][0], HumanLabel) and not isinstance(by_mode["v1"][0], BlindLabel)


@pytest.mark.parametrize(
    "labels,message",
    [
        ([], "exactly once"),
        ([BlindLabel(task_id="a", slot="A"), BlindLabel(task_id="a", slot="B")], "Unlabelled"),
    ],
)
def test_unblind_refuses_incomplete_labels(labels: list[BlindLabel], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        unblind(labels, {"a": {"A": "v1", "B": "v2"}})
    with pytest.raises(ValueError, match="exactly once"):
        unblind([blind("a", "A", True)] * 2, {"a": {"A": "v1", "B": "v2"}})


# --- statistics ----------------------------------------------------------------------


def test_sign_test_is_exact_and_two_sided() -> None:
    assert sign_test(0, 0) == 1.0
    assert sign_test(5, 0) == pytest.approx(2 / 32)
    assert sign_test(0, 5) == sign_test(5, 0)
    assert sign_test(3, 3) == 1.0


def test_bootstrap_is_seeded_and_brackets_the_difference() -> None:
    pairs = [(False, True)] * 6 + [(True, False)] * 2 + [(True, True)] * 10
    first = bootstrap_difference(pairs, seed=1, resamples=2000)
    assert first == bootstrap_difference(pairs, seed=1, resamples=2000)
    low, high = first["interval_95"]
    assert first["difference"] == pytest.approx(4 / 18) and low <= first["difference"] <= high
    assert bootstrap_difference([], seed=1)["difference"] is None


# --- paired comparison ---------------------------------------------------------------


def test_compare_counts_paired_outcomes_and_keeps_failures() -> None:
    ids = ["a", "b", "c", "d"]
    v1_labels = [
        label("a", False),
        label("b", True),
        label("c", False),
        label(
            "d",
            False,
            outcome="execution_failure",
            primary_failure="infrastructure_failure",
            claims_inspected=None,
            claims_supported=None,
            claims_contradicted=None,
            claims_insufficient=None,
        ),
    ]
    v2_labels = [label("a", True), label("b", False), label("c", True), label("d", True)]
    result = compare(
        report("v1", ids), report("v2", ids), {"v1": v1_labels, "v2": v2_labels}, seed=1
    )
    paired = result["paired_task_success"]["all"]
    assert (paired["v2_wins"], paired["v2_losses"], paired["ties"]) == (3, 1, 0)
    assert paired["v1_success"] == {"count": 1, "denominator": 4, "rate": 0.25}
    assert result["paired_task_success"]["category:crossref"]["v2_wins"] == 1
    assert result["outcome_transitions"]["execution_failure -> correct_answer"] == 1
    # The V1 execution failure has no claim counts: excluded from the paired claim rate, visible.
    assert result["paired_claim_support_rate"]["excluded_unknown"] == 1
    assert [c["task_id"] for c in result["cases"]["v2_losses"]] == ["b"]
    assert result["primary_failures"]["v1"]["infrastructure_failure"] == 1
    # V1 tool calls are null, never zero, so no paired tool-call difference exists.
    assert result["cost"]["v1"]["median_tool_calls"] is None
    assert result["cost"]["paired_differences"]["tool_calls"]["n"] == 0
    assert result["reporting_eligible"] is False


def test_compare_requires_labels_for_every_paired_case() -> None:
    with pytest.raises(ValueError, match="v2 labels"):
        compare(
            report("v1", ["a"]),
            report("v2", ["a"]),
            {"v1": [label("a", True)], "v2": []},
            seed=1,
        )


# --- end to end, offline --------------------------------------------------------------


def fill(prefix: Path) -> None:
    """Synthetic labels: slot A succeeds unless its run failed. A wiring check only."""
    key = json.loads(Path(f"{prefix}.blind-key.json").read_text())
    status = {
        (row["task_id"], mode): row["result"]["status"]
        for mode, path in key["reports"].items()
        for row in json.loads(Path(path).read_text())["rows"]
    }
    path = Path(f"{prefix}.blind-labels.jsonl")
    filled = []
    for line in path.read_text().splitlines():
        task_id, slot = json.loads(line)["task_id"], json.loads(line)["slot"]
        failed = status[task_id, key["slots"][task_id][slot]] in ("failed", "cancelled")
        if failed:
            values = label(
                task_id,
                False,
                outcome="execution_failure",
                primary_failure="infrastructure_failure",
            )
        else:
            values = label(task_id, slot == "A")
        filled.append(values.model_dump(mode="json") | {"slot": slot})
    path.write_text("".join(json.dumps(r) + "\n" for r in filled))


async def test_offline_paired_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments import v2_standalone
    from scripts import compare_v1_v2, paired_review
    from scripts.v2_labels import freeze

    corpus(tmp_path, monkeypatch)
    out = {m: tmp_path / m / "r.json" for m in ("v1", "v2")}
    for mode, path in out.items():
        await v2_standalone.run(runner_args(path, mode=mode))
    v1 = json.loads(out["v1"].read_text())
    assert v1["provenance"]["frozen_config"]["agent_mode"] == "v1"
    assert v1["provenance"]["frozen_config"]["planner_version"] is None
    assert {r["mode"] for r in v1["rows"]} == {"v1"}
    assert all(r["diagnostics"]["tool_calls"] is None for r in v1["rows"])

    prefix = tmp_path / "pair"
    paired_review.sheet(out["v1"], out["v2"], prefix, seed=3)
    sheet = Path(f"{prefix}.review.md").read_text()
    # Mode names, stop reasons and versions are hidden; answer text (including V2's fixed
    # abstention wording) is shown because the reviewer must judge it, and is disclosed.
    for leak in ("v1", "v2", "stop:", "sufficient_evidence", "planner"):
        assert leak not in sheet.lower().replace("pair", "")
    with pytest.raises(SystemExit, match="exists"):
        paired_review.sheet(out["v1"], out["v2"], prefix, seed=3)
    with pytest.raises(SystemExit, match="Unlabelled"):
        paired_review.unblind_labels(prefix)

    fill(prefix)
    paired_review.unblind_labels(prefix)
    with pytest.raises(SystemExit, match="v1: Labels are not frozen"):
        compare_v1_v2.run(out["v1"], out["v2"], prefix, {"v1": None, "v2": None}, seed=1)
    for path in out.values():
        freeze(path.with_suffix(".labels.jsonl"), path)

    target = compare_v1_v2.run(out["v1"], out["v2"], prefix, {"v1": None, "v2": None}, seed=1)
    result = json.loads(target.read_text())
    v2_labels = out["v2"].with_suffix(".labels.jsonl").read_text().splitlines()
    v2_success = sum(json.loads(line)["task_success"] for line in v2_labels)
    assert result["comparison"]["paired_task_success"]["all"]["v2_success"]["count"] == v2_success
    assert result["limitations"][0].startswith("NOT reporting-eligible")
    assert "Reporting-eligible: **no**" in target.with_suffix(".md").read_text()


async def test_interactive_labeller_saves_valid_blind_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments import v2_standalone
    from scripts import paired_review

    corpus(tmp_path, monkeypatch)
    out = {m: tmp_path / m / "r.json" for m in ("v1", "v2")}
    for mode, path in out.items():
        await v2_standalone.run(runner_args(path, mode=mode, limit=1))
    prefix = tmp_path / "pair"
    paired_review.sheet(out["v1"], out["v2"], prefix, seed=3)
    labels_path = Path(f"{prefix}.blind-labels.jsonl")
    before = labels_path.read_text()

    monkeypatch.setattr("builtins.input", lambda prompt: "q")
    paired_review.label(prefix, "tester")
    assert labels_path.read_text() == before  # quitting saves nothing partial

    def respond(prompt: str) -> str:
        if prompt.endswith("]: "):
            return ""  # accept the offered default
        if "present/absent" in prompt:
            return "p"
        if "task success" in prompt:
            return "y"
        return "1"  # first listed outcome

    monkeypatch.setattr("builtins.input", respond)
    paired_review.label(prefix, "tester")
    labels = [BlindLabel.model_validate_json(x) for x in labels_path.read_text().splitlines()]
    assert len(labels) == 2 and all(x.complete and x.reviewer == "tester" for x in labels)
    paired_review.unblind_labels(prefix)
    assert out["v1"].with_suffix(".labels.jsonl").exists()
