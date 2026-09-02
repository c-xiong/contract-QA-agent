"""Answer-level graders."""

from __future__ import annotations

from app.agent.runner import ResearchResult
from app.schemas.evidence import Citation, CitationError
from evals.graders.answer import (
    abstention_correctness,
    citation_coverage,
    citation_validity,
    forbidden_claims,
    grade_answer,
    post_gate_citation_validity,
    required_points,
)
from evals.schema import EvalTask

HYPOTHESIS = (
    "Receiving Party shall not reverse engineer any objects which embody Confidential Information."
)


def task(**overrides: object) -> EvalTask:
    payload: dict[str, object] = {
        "task_id": "t-001",
        "category": "single_document_fact_lookup",
        "question": "Which jurisdiction's law applies?",
        "expected_behavior": "answer",
        "dataset_version": "test-v1",
        "expected_document_ids": ["doc-900"],
        "expected_evidence": [
            {"document_id": "doc-900", "page_number": 1, "span_text": "x", "source": "cuad"}
        ],
    }
    payload.update(overrides)
    return EvalTask.model_validate(payload)


def result(
    answer: str = "The agreement is governed by New York law. [doc-900, p. 1]",
    status: str = "completed",
    citations: int = 1,
    errors: int = 0,
) -> ResearchResult:
    return ResearchResult(
        question="q",
        answer=answer,
        status=status,
        citations=[Citation(document_id="doc-900", page_number=1, raw="[doc-900, p. 1]")]
        * citations,
        citation_errors=[
            CitationError(code="not_in_evidence", raw="[doc-901, p. 1]", detail="never retrieved")
        ]
        * errors,
        retrieved=[],
        evidence=[],
        trace=[],
        input_tokens=0,
        output_tokens=0,
    )


class TestAbstention:
    def test_correct_answer(self) -> None:
        assert abstention_correctness(task(), result()).passed

    def test_correct_abstention(self) -> None:
        abstain_task = task(
            expected_behavior="abstain", expected_document_ids=[], expected_evidence=[]
        )
        assert abstention_correctness(abstain_task, result(status="abstained")).passed

    def test_failing_to_abstain_is_a_hallucination(self) -> None:
        abstain_task = task(
            expected_behavior="abstain", expected_document_ids=[], expected_evidence=[]
        )
        grade = abstention_correctness(abstain_task, result())
        assert not grade.passed
        assert "FAILED TO ABSTAIN" in grade.detail

    def test_over_abstaining_is_a_different_failure(self) -> None:
        """A system tuned to abstain constantly scores perfectly on unanswerable tasks
        and is useless; both directions must be visible."""
        grade = abstention_correctness(task(), result(status="abstained"))
        assert not grade.passed
        assert "over-abstained" in grade.detail


class TestForbiddenClaims:
    def test_asserting_a_forbidden_claim_fails(self) -> None:
        """The ContractNLI Contradiction case: on-topic evidence exists and is
        retrievable, so every deterministic citation check passes. Only this catches it."""
        grade = forbidden_claims(
            task(forbidden_claims=[HYPOTHESIS]),
            result(answer=f"Yes. {HYPOTHESIS} [doc-900, p. 1]"),
        )
        assert not grade.passed
        assert grade.data["asserted"] == [HYPOTHESIS]

    def test_not_asserting_it_passes(self) -> None:
        grade = forbidden_claims(
            task(forbidden_claims=[HYPOTHESIS]),
            result(answer="The agreement is silent on reverse engineering. [doc-900, p. 1]"),
        )
        assert grade.passed

    def test_abstention_cannot_assert_anything(self) -> None:
        grade = forbidden_claims(
            task(forbidden_claims=[HYPOTHESIS]), result(status="abstained", answer="")
        )
        assert grade.passed
        assert "abstained" in grade.detail

    def test_no_declared_claims_passes_trivially(self) -> None:
        assert forbidden_claims(task(), result()).passed

    def test_matching_survives_whitespace_and_case(self) -> None:
        noisy = HYPOTHESIS.upper().replace(" ", "  ")
        grade = forbidden_claims(task(forbidden_claims=[HYPOTHESIS]), result(answer=noisy))
        assert not grade.passed


class TestRequiredPoints:
    def test_present(self) -> None:
        grade = required_points(task(required_points=["New York"]), result())
        assert grade.score == 1.0

    def test_partially_present(self) -> None:
        grade = required_points(task(required_points=["New York", "arbitration"]), result())
        assert grade.score == 0.5
        assert grade.data["missing"] == ["arbitration"]

    def test_abstention_scores_zero(self) -> None:
        grade = required_points(
            task(required_points=["New York"]), result(status="abstained", answer="")
        )
        assert grade.score == 0.0


class TestCitationValidity:
    def test_all_verified(self) -> None:
        assert citation_validity(task(), result()).score == 1.0

    def test_mixed(self) -> None:
        assert citation_validity(task(), result(citations=1, errors=1)).score == 0.5

    def test_uncited_answer_scores_zero(self) -> None:
        grade = citation_validity(task(), result(answer="The cap is $1M.", citations=0))
        assert grade.score == 0.0
        assert "no citations" in grade.detail

    def test_abstention_has_nothing_to_validate(self) -> None:
        """The gate worked and the answer did not; the decomposition must show both."""
        grade = citation_validity(task(), result(status="abstained", answer="", citations=0))
        assert grade.passed

    def test_new_runs_use_explicit_post_gate_name(self) -> None:
        grade = post_gate_citation_validity(task(), result())
        assert grade.name == "post_gate_citation_validity"
        assert grade.version == "post_gate_citation_validity@1"

    def test_rejected_parsed_citation_is_not_also_counted_as_valid(self) -> None:
        citation = Citation(
            document_id="doc-901",
            page_number=1,
            raw="[doc-901, p. 1]",
        )
        rejected = result(citations=0)
        rejected = ResearchResult(
            question=rejected.question,
            answer="Unsupported [doc-901, p. 1].",
            status="completed",
            citations=[citation],
            citation_errors=[
                CitationError(
                    code="not_in_evidence",
                    raw=citation.raw,
                    citation=citation,
                    detail="never retrieved",
                )
            ],
            retrieved=[],
            evidence=[],
            trace=[],
            input_tokens=0,
            output_tokens=0,
        )
        grade = post_gate_citation_validity(task(), rejected)
        assert grade.score == 0.0
        assert grade.data["valid_citations"] == 0
        assert grade.data["citation_attempts"] == 1


class TestCitationCoverage:
    def test_counts_factual_claims_not_raw_citation_occurrences(self) -> None:
        grade = citation_coverage(
            task(),
            result(
                answer=(
                    "New York law governs [doc-900, p. 1]. The agreement also requires arbitration."
                )
            ),
        )
        assert grade.version == "citation_coverage@1"
        assert grade.score == 0.5
        assert grade.data == {
            "claims": 2,
            "cited_claims": 1,
            "applicable": True,
            "malformed_claims": 0,
        }

    def test_malformed_attempt_does_not_count_as_coverage(self) -> None:
        grade = citation_coverage(
            task(),
            result(answer="New York law governs [doc-900, page nope].", citations=0),
        )
        assert grade.score == 0.0
        assert grade.data["malformed_claims"] == 1

    def test_abstention_has_no_factual_claim_denominator(self) -> None:
        grade = citation_coverage(task(), result(status="abstained", answer="", citations=0))
        assert grade.score == 1.0
        assert grade.data["applicable"] is False


class TestGradeAnswer:
    def test_runs_every_grader_and_versions_each(self) -> None:
        grades = grade_answer(task(), result())
        assert {g.name for g in grades} == {
            "abstention_correctness",
            "forbidden_claims",
            "required_points",
            "post_gate_citation_validity",
            "citation_coverage",
        }
        assert all(g.version for g in grades)
