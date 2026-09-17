"""Model-based claim-support grader. SPEC 13.4 layer 2, SPEC 15.5.

DECISION-BEARING MODULE. See docs/decisions.md.

Layer 1 (`citation_verifier`) proves a citation points somewhere real that was actually
retrieved. It cannot prove the cited text *supports* the claim attached to it, because
that is semantic and layer 1 is deliberately not.

This grader closes that gap, and it is the only grader in the project that calls a
model. It therefore:

- costs money, so it is gated behind CRA_LIVE_MODEL like everything else;
- returns `unknown` rather than a guess when it cannot run, because reporting
  "supported" from a stub would be worse than reporting nothing;
- uses one structured rubric per dimension and requires a verbatim quote, which is
  verified in code (`app/evidence/claim_support`), never taken on trust.
"""

from __future__ import annotations

from app.agent.llm import ModelClient
from app.agent.runner import ResearchResult
from app.evidence.claim_support import CLAIM_SUPPORT_VERSION, check_answer
from evals.graders.retrieval import GradeResult
from evals.schema import EvalTask


async def claim_support(
    task: EvalTask,
    result: ResearchResult,
    client: ModelClient,
    *,
    live: bool,
) -> GradeResult:
    """Fraction of the answer's factual claims the cited evidence actually supports.

    DECISION: an abstention scores 1.0, with a note.
      An abstention asserts nothing, so there is no unsupported claim in it. Scoring it
      0.0 would conflate "said nothing" with "said something wrong" -- the exact
      distinction the abstention category exists to measure.

    DECISION: `not_addressed` counts against the score, not just `contradicted`.
      A claim whose cited evidence is merely on-topic is unsupported. Counting only
      outright contradictions would pass the most common real failure: an answer that
      drifts past what the evidence says while citing something adjacent.
    """
    if result.abstained:
        return GradeResult(
            name="claim_support",
            version=CLAIM_SUPPORT_VERSION,
            score=1.0,
            passed=True,
            detail="abstained; no claims asserted",
            data={
                "abstained": True,
                "claims": 0,
                "evaluated_claims": 0,
                "supported": 0,
            },
        )

    verdicts = await check_answer(result.answer, result.evidence, client, live=live)

    if not verdicts:
        return GradeResult(
            name="claim_support",
            version=CLAIM_SUPPORT_VERSION,
            score=1.0,
            passed=True,
            detail="no checkable factual claims in the answer",
            data={"claims": 0, "evaluated_claims": 0, "supported": 0},
        )

    evaluated = [verdict for verdict in verdicts if verdict.verdict != "unknown"]
    if not evaluated:
        return GradeResult(
            name="claim_support",
            version=CLAIM_SUPPORT_VERSION,
            score=0.0,
            passed=False,
            detail="not evaluated (requires CRA_LIVE_MODEL=1)",
            data={
                "evaluated": False,
                "claims": len(verdicts),
                "evaluated_claims": 0,
                "supported": 0,
            },
        )

    supported = sum(1 for verdict in evaluated if verdict.verdict == "supported")
    problems = [verdict for verdict in evaluated if verdict.is_problem]
    score = supported / len(evaluated)
    counts = {
        status: sum(1 for verdict in verdicts if verdict.verdict == status)
        for status in (
            "supported",
            "contradicted",
            "not_addressed",
            "uncited",
            "malformed_citation",
            "citation_not_in_evidence",
            "unknown",
        )
    }

    return GradeResult(
        name="claim_support",
        version=CLAIM_SUPPORT_VERSION,
        score=score,
        passed=supported == len(verdicts),
        detail=f"{supported}/{len(evaluated)} evaluated claims supported by their citations",
        data={
            "evaluated": True,
            "claims": len(verdicts),
            "evaluated_claims": len(evaluated),
            "supported": supported,
            **counts,
            "unsupported_examples": [
                {
                    "claim": verdict.claim[:160],
                    "verdict": verdict.verdict,
                    "reason": verdict.reason[:160],
                    "citations": list(verdict.citations),
                    "evidence_ids": list(verdict.evidence_ids),
                }
                for verdict in problems[:3]
            ],
        },
    )
