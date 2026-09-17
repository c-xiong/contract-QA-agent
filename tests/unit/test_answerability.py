"""Runtime routing requires real quoted evidence, not just a plausible JSON verdict."""

from __future__ import annotations

import json

import pytest

from app.agent.answerability import AnswerabilityError, validate_answerability
from app.agent.graph import to_evidence
from app.schemas.evidence import Evidence
from tests.conftest import make_chunk, make_retrieved


@pytest.fixture
def evidence() -> list[Evidence]:
    return to_evidence(
        [make_retrieved(make_chunk(text="The Supplier shall not disclose\n confidential data."))],
        max_chars=1000,
    )


def verdict(**changes: object) -> str:
    payload: dict[str, object] = {
        "verdict": "answerable",
        "supported_aspects": ["Disclosure is prohibited"],
        "missing_aspects": [],
        "evidence_ids": ["e01"],
        "quotes": [{"evidence_id": "e01", "quote": "shall not disclose confidential data"}],
        "reason": "The clause prohibits disclosure.",
    }
    payload.update(changes)
    return json.dumps(payload)


def test_a_verbatim_negative_clause_is_answerable(evidence: list[Evidence]) -> None:
    decision = validate_answerability(verdict(), evidence)
    assert decision.verdict == "answerable"
    assert decision.version == "answerability@1"


@pytest.mark.parametrize(
    "changes",
    [
        {"quotes": [{"evidence_id": "e01", "quote": "shall disclose confidential data"}]},
        {"quotes": [{"evidence_id": "e01", "quote": "shall NOT disclose confidential data"}]},
        {"quotes": [{"evidence_id": "e01", "quote": " "}]},
        {"quotes": []},
        {
            "evidence_ids": ["not-retrieved"],
            "quotes": [{"evidence_id": "not-retrieved", "quote": "shall not disclose"}],
        },
        {"evidence_ids": ["e02"]},
        {"evidence_ids": ["e01", "e01"]},
        {"supported_aspects": []},
        {"supported_aspects": [" "]},
        {"missing_aspects": ["Insurance coverage"]},
        {"verdict": "partial"},
        {"verdict": "unanswerable"},
        {"verdict": "probably"},
    ],
)
def test_invalid_support_never_authorizes_an_answer(
    evidence: list[Evidence], changes: dict[str, object]
) -> None:
    with pytest.raises(AnswerabilityError):
        validate_answerability(verdict(**changes), evidence)


def test_quote_cannot_be_borrowed_from_another_evidence_item(evidence: list[Evidence]) -> None:
    other = evidence[0].model_copy(
        update={"evidence_id": "e02", "excerpt": "Insurance is required."}
    )
    with pytest.raises(AnswerabilityError, match="quote not found"):
        validate_answerability(
            verdict(quotes=[{"evidence_id": "e01", "quote": "Insurance is required."}]),
            [*evidence, other],
        )


def test_partial_requires_supported_and_missing_aspects(evidence: list[Evidence]) -> None:
    decision = validate_answerability(
        verdict(verdict="partial", missing_aspects=["Insurance coverage"]), evidence
    )
    assert decision.verdict == "partial"
    assert decision.missing_aspects == ["Insurance coverage"]


def test_unanswerable_does_not_require_fabricating_a_support_quote(
    evidence: list[Evidence],
) -> None:
    decision = validate_answerability(
        verdict(
            verdict="unanswerable",
            supported_aspects=[],
            evidence_ids=[],
            quotes=[],
            missing_aspects=["Required insurance amount"],
        ),
        evidence,
    )
    assert decision.verdict == "unanswerable"


@pytest.mark.parametrize("text", ["not json", "{}", "[]", '{"verdict": "answerable"}'])
def test_malformed_model_output_is_not_an_abstention(text: str, evidence: list[Evidence]) -> None:
    with pytest.raises(AnswerabilityError):
        validate_answerability(text, evidence)
