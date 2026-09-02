"""Result rendering remains compatible across grader versions."""

from __future__ import annotations

from scripts.summarize_results import render_eval


def test_historical_claim_support_v1_artifact_remains_readable() -> None:
    lines = render_eval(
        {
            "suite": "historical",
            "dataset_version": "hand-v1",
            "task_count": 1,
            "model_id": "historical-model",
            "retriever": "bm25",
            "top_k": 5,
            "grader_versions": {"claim_support": "claim_support@1"},
            "means": {"claim_support": 0.807},
        }
    )
    rendered = "\n".join(lines)
    assert "`claim_support@1`" in rendered
    assert "0.807" in rendered


def test_current_artifact_renders_claim_support_macro_micro_and_coverage() -> None:
    lines = render_eval(
        {
            "suite": "current",
            "dataset_version": "fixture-v1",
            "task_count": 2,
            "model_id": "fixture",
            "retriever": "bm25",
            "top_k": 5,
            "grader_versions": {
                "claim_support": "claim_support@2",
                "citation_coverage": "citation_coverage@1",
            },
            "means": {"claim_support": 0.75, "citation_coverage": 0.5},
            "claim_support_aggregate": {
                "macro": 0.75,
                "micro": 0.8,
                "supported_claims": 4,
                "evaluated_claims": 5,
            },
            "citation_coverage_aggregate": {
                "macro": 0.5,
                "micro": 0.6,
                "cited_claims": 3,
                "factual_claims": 5,
            },
        }
    )
    rendered = "\n".join(lines)
    assert "macro **0.750**, micro **0.800**" in rendered
    assert "Citation coverage: macro **0.500**, micro **0.600**" in rendered
