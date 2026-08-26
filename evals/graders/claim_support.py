"""Model-based claim-support grader. SPEC 13.4 layer 2, SPEC 15.5.

DECISION-BEARING MODULE. See docs/review-questions.md.

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
            data={"abstained": True},
        )

    verdicts = await check_answer(result.answer, result.evidence, client, live=live)

    if not verdicts:
        return GradeResult(
            name="claim_support",
            version=CLAIM_SUPPORT_VERSION,
            score=1.0,
            passed=True,
            detail="no checkable factual claims in the answer",
        )

    if all(v.verdict == "unknown" for v in verdicts):
        return GradeResult(
            name="claim_support",
            version=CLAIM_SUPPORT_VERSION,
            score=0.0,
            passed=False,
            detail="not evaluated (requires CRA_LIVE_MODEL=1)",
            data={"evaluated": False},
        )

    supported = sum(1 for v in verdicts if v.verdict == "supported")
    problems = [v for v in verdicts if v.is_problem]
    score = supported / len(verdicts)

    return GradeResult(
        name="claim_support",
        version=CLAIM_SUPPORT_VERSION,
        score=score,
        passed=not problems,
        detail=f"{supported}/{len(verdicts)} claims supported by the cited evidence",
        data={
            "evaluated": True,
            "claims": len(verdicts),
            "contradicted": sum(1 for v in verdicts if v.verdict == "contradicted"),
            "not_addressed": sum(1 for v in verdicts if v.verdict == "not_addressed"),
            "unsupported_examples": [
                {"claim": v.claim[:160], "verdict": v.verdict, "reason": v.reason[:160]}
                for v in problems[:3]
            ],
        },
    )
