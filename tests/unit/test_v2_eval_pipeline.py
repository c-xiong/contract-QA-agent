"""Standalone V2 evaluation infrastructure: split, budget ledger, judges, labels, runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from app.agent.graph import build_graph
from app.agent.llm import ModelError, ModelResponse
from app.agent.state import initial_state
from app.agent.tool_registry import ToolRegistry
from app.evidence.claim_support import check_answer
from app.schemas.evidence import Evidence
from evals.budget import BudgetedClient, BudgetExceeded, Ledger, Preflight
from evals.graders.answer_rubric import ANSWER_RUBRIC_SYSTEM, judge_answer, rubric_prompt
from evals.judge_split import CALIBRATION_SIZE, build_manifest, split
from evals.loader import load_suite, suite_path
from evals.schema import EvalTask
from evals.v2_labels import (
    Adjudication,
    HumanLabel,
    agreement,
    apply_adjudications,
    bind_source_report,
    check_frozen,
    freeze,
    quality,
    sha256,
)
from tests.conftest import make_chunk, make_document
from tests.unit.test_v2 import ScriptedModel, action, registry

__all__ = ["registry"]  # re-exported pytest fixture


# --- frozen split and claim ceiling ------------------------------------------


def test_checked_in_split_is_reproducible_and_stratified() -> None:
    tasks = load_suite("hand")
    manifest = build_manifest(tasks, suite_path("hand"))
    checked_in = json.loads(Path("evals/datasets/hand/judge_split-v1.json").read_text())
    assert manifest == checked_in
    calibration, audit = set(manifest["calibration"]), set(manifest["audit"])
    assert len(calibration) == CALIBRATION_SIZE and len(audit) == len(tasks) - CALIBRATION_SIZE
    assert not calibration & audit and calibration | audit == {t.task_id for t in tasks}
    for category in {t.category for t in tasks}:
        ids = {t.task_id for t in tasks if t.category == category}
        assert ids & audit, category  # every category is audited
    assert split(tasks, seed=1) != split(tasks)  # the seed matters and is recorded


def test_claim_ceiling_matches_historical_answers() -> None:
    from app.evidence.claim_support import extract_cited_claims

    manifest = json.loads(Path("evals/datasets/hand/claim_ceiling-v1.json").read_text())
    counts = []
    for source in manifest["sources"]:
        report = json.loads(Path(source["path"]).read_text())
        counts += [len(extract_cited_claims(r["answer"])) for r in report["runs"]]
    assert manifest["C"] == max(counts) and manifest["answers"] == len(counts)


# --- budget ledger -------------------------------------------------------------


def preflight(**overrides: Any) -> Preflight:
    values: dict[str, Any] = {
        "experiment_id": "test",
        "phase": "application",
        "model_id": "scripted-v2-fixture",
        "input_price_per_million_tokens": 3.0,
        "output_price_per_million_tokens": 15.0,
        "max_attempts": 10,
        "max_input_tokens_per_call": 50_000,
        "max_output_tokens_per_call": 2048,
        "max_total_input_tokens": 1_000_000,
        "max_total_output_tokens": 100_000,
        "max_cost": 10.0,
    }
    return Preflight.model_validate(values | overrides)


class Fixed:
    model_id = "scripted-v2-fixture"

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = 0

    async def complete(self, system: str, user: str) -> ModelResponse:
        self.calls += 1
        if self.fail:
            raise ModelError("provider timeout")
        return ModelResponse("ok", input_tokens=100, output_tokens=10, model_id=self.model_id)


async def test_ledger_settles_reported_usage(tmp_path: Path) -> None:
    ledger = Ledger(preflight(), tmp_path / "ledger.json")
    client = BudgetedClient(Fixed(), ledger)
    await client.complete("system", "user prompt")
    s = ledger.state
    assert s.attempts == 1 and s.settled_input_tokens == 100 and s.settled_output_tokens == 10
    assert s.reserved_input_tokens == 0 and s.reserved_cost == pytest.approx(0)
    assert s.settled_cost == pytest.approx((100 * 3 + 10 * 15) / 1_000_000)


async def test_unknown_usage_keeps_full_reservation(tmp_path: Path) -> None:
    ledger = Ledger(preflight(), tmp_path / "ledger.json")
    with pytest.raises(ModelError):
        await BudgetedClient(Fixed(fail=True), ledger).complete("s", "x" * 400)
    s = ledger.state
    assert s.unknown_usage_attempts == 1
    assert s.settled_output_tokens == 2048 and s.settled_input_tokens == 201  # ceil(401 / 2)


async def test_caps_refuse_before_sending_and_persist(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    inner = Fixed()
    ledger = Ledger(preflight(max_attempts=2), path)
    client = BudgetedClient(inner, ledger)
    await client.complete("s", "u" * 200)
    await client.complete("s", "u" * 200)
    with pytest.raises(BudgetExceeded):
        await client.complete("s", "u" * 200)
    assert inner.calls == 2 and ledger.exhausted and ledger.state.exhausted_reason == "attempt_cap"
    # Resuming reloads cumulative totals; the budget cannot be reset by restarting.
    resumed = Ledger(preflight(max_attempts=2), path)
    assert resumed.state.attempts == 2 and resumed.exhausted
    with pytest.raises(ValueError, match="different preflight"):
        Ledger(preflight(max_attempts=99), path)


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"max_cost": 0.01}, "cost_cap"),
        ({"max_total_output_tokens": 1000}, "output_token_cap"),
        ({"max_total_input_tokens": 1}, "input_token_cap"),
    ],
)
async def test_token_and_currency_caps(
    tmp_path: Path, overrides: dict[str, Any], reason: str
) -> None:
    ledger = Ledger(preflight(**overrides), tmp_path / "ledger.json")
    with pytest.raises(BudgetExceeded):
        await BudgetedClient(Fixed(), ledger).complete("s", "u" * 10)
    assert ledger.state.exhausted_reason == reason and ledger.state.attempts == 0


async def test_oversized_prompt_is_refused_without_ending_experiment(tmp_path: Path) -> None:
    ledger = Ledger(preflight(max_input_tokens_per_call=10), tmp_path / "ledger.json")
    with pytest.raises(BudgetExceeded):
        await BudgetedClient(Fixed(), ledger).complete("s", "x" * 100)
    assert not ledger.exhausted and ledger.state.refused == 1


def test_preflight_must_price_the_client_model(tmp_path: Path) -> None:
    ledger = Ledger(preflight(model_id="other-model"), tmp_path / "ledger.json")
    with pytest.raises(ValueError, match="preflight prices"):
        BudgetedClient(Fixed(), ledger)


async def test_budget_refusal_is_a_failure_not_an_abstention(
    registry: ToolRegistry, tmp_path: Path
) -> None:
    ledger = Ledger(preflight(max_attempts=1), tmp_path / "ledger.json")
    model = ScriptedModel([action("document_search", query="notice"), action("draft")])
    graph = build_graph(
        registry.retriever,
        registry.verifier,
        BudgetedClient(model, ledger),
        registry.store,
        agent_mode="v2",
        runs_dir=tmp_path / "runs",
    )
    state = initial_state("When is notice due?", allowed_document_ids=["doc-900"])
    final = await graph.ainvoke(state, config={"recursion_limit": 100})
    assert final["status"] == "failed" and final["stop_reason"] == "model_unavailable"
    assert ledger.exhausted


# --- judges ----------------------------------------------------------------------


class Scripted:
    model_id = "judge-fixture"

    def __init__(self, texts: list[str]):
        self.texts = iter(texts)
        self.prompts: list[tuple[str, str]] = []

    async def complete(self, system: str, user: str) -> ModelResponse:
        self.prompts.append((system, user))
        return ModelResponse(next(self.texts))


def evidence(section: str, text: str) -> Evidence:
    chunk = make_chunk(f"doc-900-c{section}", section_path=[section], text=text)
    return Evidence(
        evidence_id=f"e-{section}",
        topic="t",
        document_id=chunk.document_id,
        document_title=chunk.document_title,
        section_path=chunk.section_path,
        section_title=chunk.section_title,
        page_number=chunk.page_number,
        excerpt=chunk.text,
        source_chunk_id=chunk.chunk_id,
    )


async def test_claim_ceiling_routes_excess_claims_to_humans() -> None:
    pool = [evidence(str(i), f"Clause {i} says fact {i}.") for i in range(1, 4)]
    answer = " ".join(f"Fact {i} applies [doc-900, p. 1, §{i}]." for i in range(1, 4))
    supported = json.dumps({"verdict": "supported", "quote": "fact 1", "reason": "r"})
    client = Scripted([supported, supported])
    verdicts = await check_answer(answer, pool, client, live=True, max_model_claims=2)
    assert len(client.prompts) == 2 and len(verdicts) == 3
    assert verdicts[2].verdict == "unknown" and "human review" in verdicts[2].reason


def task(**overrides: Any) -> EvalTask:
    values: dict[str, Any] = {
        "task_id": "t-1",
        "category": "single_document_fact_lookup",
        "question": "How much notice?",
        "allowed_document_ids": ["doc-900"],
        "expected_document_ids": ["doc-900"],
        "expected_evidence": [
            {
                "document_id": "doc-900",
                "section_id": "1",
                "page_number": 1,
                "span_text": "30 days notice",
                "source": "manual",
            }
        ],
        "required_points": ["30 days", "written notice"],
        "forbidden_claims": ["60 days"],
        "expected_behavior": "answer",
        "dataset_version": "fixture",
    }
    return EvalTask.model_validate(values | overrides)


def rubric_json(**overrides: Any) -> str:
    values = {
        "required_points": [
            {"index": 0, "verdict": "present"},
            {"index": 1, "verdict": "absent"},
        ],
        "forbidden_claims": [{"index": 0, "verdict": "not_made"}],
        "behavior": "answered",
        "conclusion_correct": True,
        "task_success": False,
        "reason": "missing written notice",
    }
    return "```json\n" + json.dumps(values | overrides) + "\n```"


async def test_answer_judge_grades_every_criterion() -> None:
    client = Scripted([rubric_json()])
    verdict = await judge_answer(task(), "30 days. [doc-900, p. 1, §1]", client)
    assert verdict["verdict"] == "graded" and verdict["task_success"] is False
    system, user = client.prompts[0]
    assert system == ANSWER_RUBRIC_SYSTEM
    assert "30 days notice" in user and "[1] " not in user  # rubric numbering is 0-based text
    # The judge never sees runtime verification results or system labels.
    for hidden in ("verification", "verified", "V2", "stop_reason"):
        assert hidden not in rubric_prompt(task(), "answer")


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        rubric_json(required_points=[{"index": 0, "verdict": "present"}]),
        rubric_json(behavior="maybe"),
    ],
)
async def test_incomplete_judge_output_is_unknown(response: str) -> None:
    verdict = await judge_answer(task(), "answer", Scripted([response]))
    assert verdict["verdict"] == "unknown"


async def test_judge_error_is_unknown_not_pass() -> None:
    class Down:
        model_id = "down"

        async def complete(self, system: str, user: str) -> ModelResponse:
            raise ModelError("secret provider payload")

    verdict = await judge_answer(task(), "answer", Down())
    assert verdict["verdict"] == "unknown" and "secret" not in verdict["reason"]


# --- human labels ----------------------------------------------------------------


def label(task_id: str, success: bool, **overrides: Any) -> HumanLabel:
    values: dict[str, Any] = {
        "task_id": task_id,
        "reviewer": "author",
        "reviewer_type": "author",
        "labeled_at": "2026-09-28T00:00:00+00:00",
        "outcome": "correct_answer" if success else "partial_answer",
        "task_success": success,
        "required_points": ["present", "present" if success else "absent"],
        "forbidden_claims": ["not_made"],
        "claims_inspected": 2,
        "claims_supported": 2 if success else 1,
        "claims_contradicted": 0,
        "claims_insufficient": 0 if success else 1,
        "primary_failure": None if success else "incomplete_evidence",
    }
    return HumanLabel.model_validate(values | overrides)


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"reviewer": None}, "needs reviewer"),
        ({"primary_failure": None}, "primary failure"),
        ({"claims_supported": 5}, "sum"),
        ({"outcome": "execution_failure", "task_success": True, "primary_failure": None}, "never"),
    ],
)
def test_label_validation(overrides: dict[str, Any], message: str) -> None:
    success = overrides.get("task_success", False)
    with pytest.raises(ValueError, match=message):
        label("t-1", success, **overrides)


def report(statuses: dict[str, str]) -> dict[str, Any]:
    return {
        "rows": [
            {
                "task_id": t,
                "category": "lookup",
                "result": {"status": s},
                "diagnostics": {"answerable": True},
            }
            for t, s in statuses.items()
        ]
    }


def write_labels(path: Path, labels: list[HumanLabel]) -> None:
    path.write_text("".join(label.model_dump_json() + "\n" for label in labels))


def test_freeze_requires_complete_consistent_labels(tmp_path: Path) -> None:
    path = tmp_path / "r.labels.jsonl"
    report_path = tmp_path / "r.json"
    report_path.write_text(json.dumps(report({"a": "completed", "b": "completed"})))
    bind_source_report(path, sha256(report_path))
    write_labels(path, [label("a", True), HumanLabel(task_id="b")])
    with pytest.raises(ValueError, match="Unlabelled"):
        freeze(path, report_path)
    failed_path = tmp_path / "failed.labels.jsonl"
    failed_report = tmp_path / "failed.json"
    failed_report.write_text(json.dumps(report({"a": "completed", "b": "failed"})))
    bind_source_report(failed_path, sha256(failed_report))
    write_labels(failed_path, [label("a", True), label("b", True)])
    with pytest.raises(ValueError, match="execution failure"):
        freeze(failed_path, failed_report)
    write_labels(path, [label("a", True)])
    with pytest.raises(ValueError, match="every report row"):
        freeze(path, report_path)
    write_labels(path, [label("a", True), label("b", True)])
    record = freeze(path, report_path)
    assert record["reviewer_types"] == {"author": 2}
    assert record["report_sha256"] == sha256(report_path)
    assert check_frozen(path, sha256(report_path))["labels_sha256"] == record["labels_sha256"]
    write_labels(path, [label("a", True), label("b", False)])
    with pytest.raises(ValueError, match="changed after freezing"):
        check_frozen(path, sha256(report_path))
    with pytest.raises(ValueError, match="exists"):
        freeze(path, report_path)


def test_sheet_binding_rejects_report_mutation_before_freeze(tmp_path: Path) -> None:
    from scripts.v2_labels import paths, sheet

    task = load_suite("hand")[0]
    report_path = tmp_path / "r.json"
    source = {
        "provenance": {"frozen_config": {"suite": "hand"}},
        "rows": [
            {
                "task_id": task.task_id,
                "category": task.category,
                "split": "calibration",
                "execution": "completed",
                "result": {
                    "status": "completed",
                    "stop_reason": "verified",
                    "answer": "Original answer",
                    "evidence": [],
                },
            }
        ],
    }
    report_path.write_text(json.dumps(source))
    original_hash = sha256(report_path)
    sheet(report_path)
    labels_path = paths(report_path)["labels"]
    assert json.loads(labels_path.with_suffix(".source.json").read_text()) == {
        "report_sha256": original_hash
    }
    assert original_hash in paths(report_path)["sheet"].read_text()
    write_labels(labels_path, [label(task.task_id, True)])
    source["rows"][0]["result"]["answer"] = "Changed answer"
    report_path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="changed since the review sheet"):
        freeze(labels_path, report_path)
    assert not labels_path.with_suffix(".freeze.json").exists()


def frozen_review(tmp_path: Path) -> tuple[Path, Path, Path]:
    report_path = tmp_path / "r.json"
    source = report({"a": "completed"}) | {
        "provenance": {"reporting_eligible": True},
        "summary": {},
        "unexecuted": [],
        "budget_ledger": {},
    }
    source["rows"][0]["result"]["answer"] = "Original answer"
    report_path.write_text(json.dumps(source))
    labels_path = report_path.with_suffix(".labels.jsonl")
    bind_source_report(labels_path, sha256(report_path))
    write_labels(labels_path, [label("a", True)])
    freeze(labels_path, report_path)
    sealed = tmp_path / "r.sealed.jsonl"
    header = {"record": "header", "phase": "calibration", "report_sha256": sha256(report_path)}
    sealed.write_text(json.dumps(header) + '\n{"task_id": "a", "not_judged": "budget"}\n')
    return report_path, labels_path, sealed


@pytest.mark.parametrize("boundary", ["unseal", "final_report"])
def test_frozen_review_rejects_mutated_report(tmp_path: Path, boundary: str) -> None:
    from scripts.v2_labels import final_report, paths, unseal

    report_path, _, sealed = frozen_review(tmp_path)
    source = json.loads(report_path.read_text())
    source["rows"][0]["result"]["answer"] = "Different run's answer"
    report_path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="does not match the report used to freeze"):
        if boundary == "unseal":
            unseal(report_path, sealed, "agreement")
        else:
            final_report(report_path, None)
    assert not paths(report_path)["unseal_log"].exists()
    assert not paths(report_path)["final"].exists()


@pytest.mark.parametrize("boundary", ["unseal", "final_report"])
@pytest.mark.parametrize("legacy", [False, True])
def test_reporting_rejects_wrong_run_or_hashless_verdicts(
    tmp_path: Path, boundary: str, legacy: bool
) -> None:
    from scripts.v2_labels import final_report, paths, unseal

    report_path, _, sealed = frozen_review(tmp_path)
    # The other run contains the same task IDs, so task identity cannot detect the mix-up.
    other_report = tmp_path / "other.json"
    source = json.loads(report_path.read_text())
    source["rows"][0]["result"]["answer"] = "Other run's answer"
    other_report.write_text(json.dumps(source))
    header = {"record": "header", "phase": "calibration"}
    if not legacy:
        header["report_sha256"] = sha256(other_report)
    # A malformed verdict body proves header validation happens before verdicts are read.
    sealed.write_text(json.dumps(header) + "\nnot JSON\n")
    with pytest.raises(
        ValueError, match="hash is missing" if legacy else "different source report"
    ):
        if boundary == "unseal":
            unseal(report_path, sealed, "agreement")
        else:
            final_report(report_path, sealed)
    assert not paths(report_path)["unseal_log"].exists()
    assert not paths(report_path)["final"].exists()


def test_reporting_accepts_matching_report_hashes(tmp_path: Path) -> None:
    from scripts.v2_labels import final_report, paths, unseal

    report_path, _, sealed = frozen_review(tmp_path)
    header, verdicts = unseal(report_path, sealed, "agreement")
    assert header["report_sha256"] == sha256(report_path)
    assert set(verdicts) == {"a"}
    logged = json.loads(paths(report_path)["unseal_log"].read_text())
    assert logged["report_sha256"] == sha256(report_path)
    final_report(report_path, None)
    final = json.loads(paths(report_path)["final"].read_text())
    assert final["labels"]["report_sha256"] == sha256(report_path)
    assert final["answer_quality_human_labels"]["all"]["task_success"]["count"] == 1


def test_legacy_hashless_labels_are_rejected(tmp_path: Path) -> None:
    report_path, labels_path, _ = frozen_review(tmp_path)
    record_path = labels_path.with_suffix(".freeze.json")
    record = json.loads(record_path.read_text())
    del record["report_sha256"]
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Unsupported legacy frozen labels"):
        check_frozen(labels_path, sha256(report_path))
    unbound = tmp_path / "legacy.labels.jsonl"
    write_labels(unbound, [label("a", True)])
    with pytest.raises(ValueError, match="no source-report binding"):
        freeze(unbound, report_path)


def test_adjudication_is_explicit(tmp_path: Path) -> None:
    labels = [label("a", True)]
    path = tmp_path / "adj.jsonl"
    record = Adjudication(
        task_id="a",
        field="notes",
        old="",
        new="re-read clause",
        reason="typo",
        adjudicator="author",
        adjudicated_at="2026-09-29",
    )
    path.write_text(record.model_dump_json() + "\n")
    assert apply_adjudications(labels, path)[0].notes == "re-read clause"
    path.write_text(record.model_copy(update={"old": "wrong"}).model_dump_json() + "\n")
    with pytest.raises(ValueError, match="does not match"):
        apply_adjudications(labels, path)


def test_agreement_counts_false_passes_and_keeps_unknowns_visible() -> None:
    labels = [label("a", True), label("b", False), label("c", True), label("d", False)]

    def graded(success: bool, second: str) -> dict[str, Any]:
        return {
            "claims": [{"verdict": "supported"}, {"verdict": "unknown"}],
            "answer_rubric": {
                "verdict": "graded",
                "task_success": success,
                "required_points": [
                    {"index": 1, "verdict": second},
                    {"index": 0, "verdict": "present"},
                ],
                "forbidden_claims": [{"index": 0, "verdict": "not_made"}],
            },
        }

    verdicts = {
        "a": graded(True, "present"),
        "b": graded(True, "present"),  # false pass
        "c": {"claims": [], "answer_rubric": {"verdict": "unknown"}},
        "d": {"not_judged": "execution_failure"},
    }
    result = agreement(labels, verdicts, ["a", "b", "c", "d"])
    assert result["graded_by_judge"] == 2
    assert result["false_pass"] == {"count": 1, "denominator": 2, "rate": 0.5}
    assert result["judge_unknown"] == ["c"] and result["not_judged"] == ["d"]
    assert result["required_point_agreement"]["denominator"] == 4
    assert result["disagreements_to_review"] == [
        {"task_id": "b", "fields": ["task_success", "required_points"]}
    ]
    assert result["claim_support_totals"]["judge_graded"] == 2


def test_quality_uses_human_labels_with_denominators() -> None:
    labels = [
        label("a", True),
        label("b", False, outcome="false_abstention"),
        label("c", False, outcome="execution_failure", primary_failure="infrastructure_failure"),
    ]
    result = quality(labels, report({"a": "completed", "b": "abstained", "c": "failed"}))["all"]
    assert result["task_success"] == {"count": 1, "denominator": 3, "rate": 1 / 3}
    assert result["false_abstention"]["denominator"] == 3
    assert result["execution_failure"]["count"] == 1
    assert result["claims_supported"] == {"count": 4, "denominator": 6, "rate": 4 / 6}


# --- standalone runner -------------------------------------------------------------


def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.ingestion.store import ChunkStore

    docs = [make_document("doc-033")]
    chunks = [make_chunk("doc-033-c1", document_id="doc-033", text="Notice 30 days before.")]
    ChunkStore(docs, chunks).save(tmp_path / "data" / "processed")
    monkeypatch.setenv("CRA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CRA_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("CRA_LIVE_MODEL", "0")


def runner_args(output: Path, **overrides: Any) -> argparse.Namespace:
    values = {
        "mode": "v2",
        "suite": "hand",
        "arm": "bm25",
        "limit": 2,
        "output": str(output),
        "preflight": None,
        "resume": False,
    }
    return argparse.Namespace(**(values | overrides))


async def test_standalone_runner_is_incremental_and_resume_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments import v2_standalone

    corpus(tmp_path, monkeypatch)
    output = tmp_path / "out" / "r.json"
    await v2_standalone.run(runner_args(output))
    saved = json.loads(output.read_text())
    assert len(saved["rows"]) == 2 and saved["unexecuted"] == []
    assert saved["budget_ledger"] == "offline: no provider calls"
    assert saved["provenance"]["reporting_eligible"] is False
    assert {r["split"] for r in saved["rows"]} <= {"calibration", "audit"}
    assert saved["summary"]["all"]["answer_abstain_decision"]["denominator"] == 2
    with pytest.raises(SystemExit, match="--resume"):
        await v2_standalone.run(runner_args(output))
    # Resuming with the same frozen config re-runs nothing that already completed.
    await v2_standalone.run(runner_args(output, resume=True))
    lines = output.with_suffix(".predictions.jsonl").read_text().splitlines()
    assert len(lines) == 2
    with pytest.raises(SystemExit, match="Frozen configuration changed"):
        await v2_standalone.run(runner_args(output, resume=True, limit=3))


async def test_live_runner_requires_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments import v2_standalone

    corpus(tmp_path, monkeypatch)
    monkeypatch.setenv("CRA_LIVE_MODEL", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-used")
    with pytest.raises(SystemExit, match="--preflight"):
        await v2_standalone.run(runner_args(tmp_path / "live" / "r.json"))


@pytest.mark.parametrize(
    "name", ["v2-hand-application", "v2-hand-judge", "v1-hand-application", "v1-hand-judge"]
)
def test_unfilled_preflight_templates_cannot_be_used(name: str) -> None:
    from pydantic import ValidationError

    from evals.budget import load_preflight

    path = Path(f"evals/preflight/{name}.template.json")
    with pytest.raises(ValidationError, match="price"):
        load_preflight(path)
    ceiling = json.loads(Path("evals/datasets/hand/claim_ceiling-v1.json").read_text())["C"]
    calibration_rounds = 2 if name.startswith("v2") else 0  # V1 gets only the frozen pass
    expected = 41 * 12 if "application" in name else (calibration_rounds * 20 + 41) * (ceiling + 1)
    assert json.loads(path.read_text())["max_attempts"] == expected
