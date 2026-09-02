"""Citation-ablation metric definitions."""

from __future__ import annotations

import pytest

from app.evidence.citation_verifier import VerificationResult
from app.schemas.evidence import Citation, CitationError
from evals.citation_metrics import CitationAblationObservation, summarize_citation_ablation

VALID = Citation(document_id="doc-900", page_number=1, section_id="8.1", raw="[valid]")


def verified() -> VerificationResult:
    return VerificationResult(citations=[VALID])


def invalid() -> VerificationResult:
    return VerificationResult(
        citations=[VALID],
        errors=[
            CitationError(
                code="not_in_evidence",
                raw="[valid]",
                citation=VALID,
                detail="not selected",
            )
        ],
    )


class TestCitationAblationMetrics:
    def test_metrics_keep_raw_exposure_gate_and_repair_semantics_separate(self) -> None:
        rows = [
            CitationAblationObservation(
                ungated_answer="The cap is annual [doc-900, p. 1, §8.1].",
                ungated_audit=invalid(),
                gated_answer="The cap is annual [doc-900, p. 1, §8.1].",
                gated_audit=verified(),
                gated_delivered=True,
                repair_triggered=True,
            ),
            CitationAblationObservation(
                ungated_answer="The cap is annual.",
                ungated_audit=VerificationResult(uncited=True),
                gated_answer="I could not return a safely cited answer.",
                gated_audit=VerificationResult(uncited=True),
                gated_delivered=False,
                repair_triggered=True,
            ),
            CitationAblationObservation(
                ungated_answer="The cap is annual [doc-900, p. 1, §8.1].",
                ungated_audit=verified(),
                gated_answer="The cap is annual [doc-900, p. 1, §8.1].",
                gated_audit=verified(),
                gated_delivered=True,
                repair_triggered=False,
            ),
        ]

        metrics = summarize_citation_ablation(rows)

        assert metrics["raw_invalid_citation_rate"] == 0.5
        assert metrics["raw_invalid_citations"] == 1
        assert metrics["raw_citation_attempts"] == 2
        assert metrics["unsafe_answer_exposure_rate"] == pytest.approx(2 / 3)
        assert metrics["post_gate_citation_validity"] == 1.0
        assert metrics["citation_coverage"]["gate_off"] == pytest.approx(2 / 3)
        assert metrics["citation_coverage"]["gate_on"] == 1.0
        assert metrics["repair_trigger_rate"] == pytest.approx(2 / 3)
        assert metrics["repair_success_rate"] == 0.5

    def test_malformed_attempt_is_in_raw_invalid_denominator(self) -> None:
        malformed = VerificationResult(
            errors=[
                CitationError(
                    code="unparseable",
                    raw="[doc-900, page nope]",
                    detail="malformed",
                )
            ]
        )
        metrics = summarize_citation_ablation(
            [
                CitationAblationObservation(
                    ungated_answer="The cap is annual [doc-900, page nope].",
                    ungated_audit=malformed,
                    gated_answer="",
                    gated_audit=VerificationResult(),
                    gated_delivered=False,
                    repair_triggered=False,
                )
            ]
        )
        assert metrics["raw_invalid_citation_rate"] == 1.0
        assert metrics["raw_citation_attempts"] == 1
        assert metrics["unsafe_answer_exposure_rate"] == 1.0
        assert metrics["post_gate_citation_validity"] is None
        assert metrics["repair_success_rate"] is None
