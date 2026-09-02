"""Citation-scoped claim extraction and semantic grading."""

from __future__ import annotations

import json

from app.agent.llm import ModelResponse
from app.agent.runner import ResearchResult
from app.evidence.claim_support import (
    CLAIM_SUPPORT_READABLE_VERSIONS,
    CLAIM_SUPPORT_VERSION,
    LEGACY_CLAIM_SUPPORT_VERSION,
    check_answer,
    extract_cited_claims,
)
from app.schemas.evidence import Evidence
from evals.graders.claim_support import claim_support
from evals.schema import EvalTask


def evidence(
    evidence_id: str,
    page: int | None,
    section: str | None,
    excerpt: str,
    *,
    document_id: str = "doc-900",
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        topic="test",
        document_id=document_id,
        document_title="Test Agreement",
        section_path=[section] if section is not None else [],
        page_number=page,
        excerpt=excerpt,
        source_chunk_id=f"{document_id}-{evidence_id}",
    )


class FakeSemanticJudge:
    """Queued semantic verdicts that record exactly which excerpts they received."""

    model_id = "fake-semantic-judge"

    def __init__(self, *responses: tuple[str, str]) -> None:
        self.responses = list(responses)
        self.users: list[str] = []

    async def complete(self, system: str, user: str) -> ModelResponse:
        self.users.append(user)
        verdict, quote = self.responses.pop(0)
        return ModelResponse(
            text=json.dumps({"verdict": verdict, "quote": quote, "reason": "fixture"}),
            model_id=self.model_id,
        )


class TestClaimExtraction:
    def test_one_claim_retains_one_citation(self) -> None:
        claims = extract_cited_claims(
            "The agreement is governed by Delaware law [doc-900, p. 3, §14.2]."
        )
        assert len(claims) == 1
        assert [citation.render() for citation in claims[0].citations] == ["[doc-900, p. 3, §14.2]"]

    def test_one_claim_retains_two_citations(self) -> None:
        claims = extract_cited_claims(
            "The cap has two exceptions [doc-900, p. 1, §8.1] [doc-900, p. 2, §8.3]."
        )
        assert [citation.section_id for citation in claims[0].citations] == ["8.1", "8.3"]

    def test_consecutive_sentences_keep_different_citations(self) -> None:
        claims = extract_cited_claims(
            "The cap is annual [doc-900, p. 1, §8.1]. "
            "Gross negligence is excluded [doc-900, p. 2, §8.3]."
        )
        assert len(claims) == 2
        assert [claim.citations[0].section_id for claim in claims] == ["8.1", "8.3"]

    def test_markdown_bullets_are_independent_claims(self) -> None:
        claims = extract_cited_claims(
            "- The cap is annual [doc-900, p. 1, §8.1].\n"
            "- Gross negligence is excluded [doc-900, p. 2, §8.3]."
        )
        assert [claim.text for claim in claims] == [
            "The cap is annual.",
            "Gross negligence is excluded.",
        ]

    def test_citation_only_sentence_is_attached_not_counted(self) -> None:
        claims = extract_cited_claims(
            "The agreement is governed by Delaware law. [doc-900, p. 3, §14.2]"
        )
        assert len(claims) == 1
        assert claims[0].citations[0].section_id == "14.2"

    def test_normalizes_safe_compound_citations_before_extraction(self) -> None:
        claims = extract_cited_claims(
            "The cap has exceptions [doc-900, p. 1, §8.1; doc-900, p. 2, §8.3]."
        )
        assert [citation.section_id for citation in claims[0].citations] == ["8.1", "8.3"]

    def test_repeated_citations_do_not_double_count(self) -> None:
        claims = extract_cited_claims(
            "The cap is annual [doc-900, p. 1, §8.1] [doc-900, page 1, section 8.1]."
        )
        assert len(claims[0].citations) == 1

    def test_uncited_and_malformed_are_distinct(self) -> None:
        claims = extract_cited_claims(
            "The cap is annual.\nThe carve-out applies [doc-900, page nope, §8.3]."
        )
        assert claims[0].citations == () and claims[0].malformed_citations == ()
        assert claims[1].citations == ()
        assert claims[1].malformed_citations == ("[doc-900, page nope, §8.3]",)


class TestCitationScopedVerdicts:
    async def test_each_sentence_receives_only_its_cited_evidence(self) -> None:
        first = evidence("e1", 1, "8.1", "The cap is limited to annual fees.")
        second = evidence("e2", 2, "8.3", "Gross negligence is excluded from the cap.")
        judge = FakeSemanticJudge(
            ("supported", "annual fees"),
            ("supported", "Gross negligence"),
        )
        verdicts = await check_answer(
            "The cap is annual [doc-900, p. 1, §8.1]. "
            "Gross negligence is excluded [doc-900, p. 2, §8.3].",
            [first, second],
            judge,
            live=True,
        )
        assert [verdict.verdict for verdict in verdicts] == ["supported", "supported"]
        assert "Gross negligence" not in judge.users[0]
        assert "annual fees" not in judge.users[1]

    async def test_two_citations_use_union_of_only_those_locations(self) -> None:
        first = evidence("e1", 1, "8.1", "The cap is limited to annual fees.")
        second = evidence("e2", 2, "8.3", "Gross negligence is excluded from the cap.")
        elsewhere = evidence("e3", 3, "14.2", "This Agreement is governed by Delaware.")
        judge = FakeSemanticJudge(("supported", "annual fees"))
        verdicts = await check_answer(
            "The cap has exceptions [doc-900, p. 1, §8.1] [doc-900, p. 2, §8.3].",
            [first, second, elsewhere],
            judge,
            live=True,
        )
        assert verdicts[0].evidence_ids == ("e1", "e2")
        assert "governed by Delaware" not in judge.users[0]

    async def test_support_elsewhere_never_rescues_wrong_cited_location(self) -> None:
        correct_elsewhere = evidence("e1", 1, "8.1", "The liability cap is one million dollars.")
        cited_wrong = evidence("e2", 3, "14.2", "This Agreement is governed by Delaware.")
        judge = FakeSemanticJudge(("not_addressed", ""))
        verdicts = await check_answer(
            "The liability cap is one million dollars [doc-900, p. 3, §14.2].",
            [correct_elsewhere, cited_wrong],
            judge,
            live=True,
        )
        assert verdicts[0].verdict == "not_addressed"
        assert verdicts[0].evidence_ids == ("e2",)
        assert "one million dollars" not in judge.users[0].split("Excerpts:\n", 1)[1]

    async def test_uncited_malformed_and_missing_location_do_not_call_judge(self) -> None:
        judge = FakeSemanticJudge()
        verdicts = await check_answer(
            "The cap is annual.\n"
            "The carve-out applies [doc-900, page nope, §8.3].\n"
            "Delaware law governs [doc-900, p. 99, §14.2].",
            [evidence("e1", 1, "8.1", "The cap is limited to annual fees.")],
            judge,
            live=True,
        )
        assert [verdict.verdict for verdict in verdicts] == [
            "uncited",
            "malformed_citation",
            "citation_not_in_evidence",
        ]
        assert judge.users == []

    async def test_semantic_judge_distinguishes_all_three_content_verdicts(self) -> None:
        items = [
            evidence("e1", 1, "1", "Alpha text."),
            evidence("e2", 2, "2", "Beta text."),
            evidence("e3", 3, "3", "Gamma text."),
        ]
        judge = FakeSemanticJudge(
            ("supported", "Alpha text"),
            ("contradicted", "Beta text"),
            ("not_addressed", ""),
        )
        verdicts = await check_answer(
            "Alpha applies [doc-900, p. 1, §1]. "
            "Beta does not apply [doc-900, p. 2, §2]. "
            "Gamma has a cap [doc-900, p. 3, §3].",
            items,
            judge,
            live=True,
        )
        assert [verdict.verdict for verdict in verdicts] == [
            "supported",
            "contradicted",
            "not_addressed",
        ]

    async def test_normalized_section_location_matches_case_insensitively(self) -> None:
        judge = FakeSemanticJudge(("supported", "Confidential Information"))
        verdicts = await check_answer(
            "The clause defines Confidential Information [doc-901, §S37].",
            [
                evidence(
                    "e1",
                    None,
                    "s37",
                    "Confidential Information means non-public information.",
                    document_id="doc-901",
                )
            ],
            judge,
            live=True,
        )
        assert verdicts[0].verdict == "supported"


def eval_task() -> EvalTask:
    return EvalTask.model_validate(
        {
            "task_id": "fixture",
            "category": "unanswerable",
            "question": "Missing?",
            "expected_behavior": "abstain",
            "dataset_version": "fixture-v1",
            "expected_document_ids": [],
            "expected_evidence": [],
        }
    )


class TestGraderVersioning:
    async def test_abstention_has_no_factual_claim_and_needs_no_judge(self) -> None:
        judge = FakeSemanticJudge()
        result = ResearchResult(
            question="Missing?",
            answer="I found no relevant passages.",
            status="abstained",
            citations=[],
            citation_errors=[],
            retrieved=[],
            evidence=[],
            trace=[],
            input_tokens=0,
            output_tokens=0,
        )
        grade = await claim_support(eval_task(), result, judge, live=True)
        assert grade.version == "claim_support@2"
        assert grade.score == 1.0
        assert grade.data["claims"] == 0
        assert judge.users == []

    def test_claim_support_v1_remains_a_declared_readable_version(self) -> None:
        assert CLAIM_SUPPORT_VERSION == "claim_support@2"
        assert LEGACY_CLAIM_SUPPORT_VERSION == "claim_support@1"
        assert {"claim_support@1", "claim_support@2"} == CLAIM_SUPPORT_READABLE_VERSIONS
