"""Counterexamples for attribution, provenance and answerability denominators."""

from dataclasses import replace

from experiments._harness import ConditionResult, record
from experiments.agentic_comparison import factorial_budgets, factorial_effects
from experiments.answerability_comparison import behavior_summary

from app.agent.runner import ResearchResult
from app.ingestion.store import ChunkStore
from evals.provenance import chunks_sha256
from evals.schema import EvalTask


def test_factorial_changes_only_the_named_limits() -> None:
    budgets = factorial_budgets(4)
    assert len(budgets) == 4
    assert budgets["single_no_xref"].max_searches == 1
    assert budgets["loop_no_xref"].max_searches > 1
    assert budgets["loop_no_xref"].max_cross_reference_chunks == 0
    assert budgets["single_xref"].max_cross_reference_chunks > 0
    assert len({b.max_evidence_chars for b in budgets.values()}) == 1
    assert {b.max_chunks_per_query for b in budgets.values()} == {4}


def test_factorial_attributes_interaction_separately() -> None:
    conditions = {name: ConditionResult(name=name) for name in factorial_budgets(5)}
    for name, value in zip(conditions, (0.0, 0.2, 0.3, 0.9), strict=True):
        conditions[name].per_task["t"] = {"required_points": value}
    effect = factorial_effects(conditions)["required_points"]
    import pytest

    assert effect["means"] == pytest.approx({"loop": 0.5, "xref": 0.4, "interaction": 0.4})


def test_corpus_hash_detects_content_and_ignores_order(store: ChunkStore) -> None:
    reversed_store = ChunkStore(documents=store.documents, chunks=list(reversed(store.chunks)))
    assert chunks_sha256(store) == chunks_sha256(reversed_store)
    chunks = list(store.chunks)
    chunks[0] = chunks[0].model_copy(update={"text": chunks[0].text + " changed"})
    changed = ChunkStore(documents=store.documents, chunks=chunks)
    assert chunks_sha256(store) != chunks_sha256(changed)


def test_execution_failure_is_not_correct_abstention_or_false_answer() -> None:
    negative = EvalTask(
        task_id="n",
        category="missing",
        question="Missing?",
        expected_behavior="abstain",
        dataset_version="fixture",
    )
    positive = EvalTask(
        task_id="p",
        category="present",
        question="Present?",
        expected_document_ids=["doc-900"],
        expected_behavior="answer",
        dataset_version="fixture",
    )
    result = ResearchResult(
        question="Missing?",
        answer="API failed",
        status="failed",
        citations=[],
        citation_errors=[],
        retrieved=[],
        evidence=[],
        trace=[],
        input_tokens=0,
        output_tokens=0,
        failure="internal_error",
    )
    outputs = {"n": result, "p": replace(result, status="abstained", failure=None)}
    condition = ConditionResult("fixture")
    for task in (negative, positive):
        record(condition, task, outputs[task.task_id], 5)
    summary = behavior_summary([negative, positive], outputs, condition)
    assert summary["false_answers"] == 0
    assert summary["correct_abstentions"] == 0
    assert summary["execution_failures"] == 1
    assert summary["over_abstentions"] == 1
    assert summary["over_abstention_rate"] == 1.0
