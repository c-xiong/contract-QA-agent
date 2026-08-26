"""Claim support checking. See docs/SPEC.md section 13.4, layer 2.

DECISION-BEARING MODULE. See docs/review-questions.md.

Layer 1 (`citation_verifier`) proves a citation points somewhere real that was actually
retrieved. It cannot prove the cited text SUPPORTS the claim attached to it, because
that is a semantic question and layer 1 is deliberately not semantic.

ContractNLI `Contradiction` cases are the natural test bed and the reason this layer
exists: the evidence is real, retrievable, and topically on point, and it contradicts
the claim. Every deterministic check passes. Only a reader of both catches it.

The check is model-based, so it costs money and is gated behind the same
`CRA_LIVE_MODEL` flag as everything else. With the flag off it returns `unknown` rather
than a guess -- reporting "supported" from a stub would be worse than reporting nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from app.agent.llm import ModelClient, ModelError
from app.evidence.citation_parser import extract_citations
from app.schemas.evidence import Evidence

Support = Literal["supported", "contradicted", "not_addressed", "unknown"]

# DECISION: a structured rubric with one dimension, not an overall quality score.
#   .claude/rules/evals.md requires model-based graders use separate structured rubrics
#   per dimension and never one call producing an opaque overall score. This asks
#   exactly one question -- does this excerpt support this sentence -- and requires a
#   verbatim quote as justification.
#   The quote requirement is the load-bearing part. A grader that answers "supported"
#   with no quote can rationalise anything; one that must point at the words has to find
#   them, and when it cannot, it says not_addressed.
CLAIM_SUPPORT_SYSTEM = """You check whether contract excerpts support a specific claim.

You will be given one claim and the excerpts that were cited for it.

Answer with JSON only, no other text:
{"verdict": "supported" | "contradicted" | "not_addressed", "quote": "<verbatim quote \
from the excerpts, or empty string>", "reason": "<one sentence>"}

Definitions:
- "supported": the excerpts state the claim, or state something that plainly entails it.
- "contradicted": the excerpts state something incompatible with the claim.
- "not_addressed": the excerpts are about the topic but neither state nor contradict it.

Rules:
- The quote must appear VERBATIM in the excerpts. If you cannot find one, the verdict is
  "not_addressed" and the quote is "".
- Judge only what the excerpts say. Do not use outside knowledge about contracts.
- Being on the same topic is not support."""

CLAIM_SUPPORT_VERSION = "claim_support@1"


@dataclass(frozen=True, slots=True)
class ClaimVerdict:
    """One claim's support verdict."""

    claim: str
    verdict: Support
    quote: str
    reason: str
    quote_found_verbatim: bool

    @property
    def is_problem(self) -> bool:
        """True when a claim was asserted that the evidence does not carry."""
        return self.verdict in ("contradicted", "not_addressed")


def split_claims(answer: str) -> list[str]:
    """Split an answer into individually checkable factual sentences.

    DECISION: sentence-level splitting, done deterministically.
      A whole answer gets one blended verdict where a single unsupported sentence hides
      behind three supported ones. Splitting first means each claim is checked on its
      own and the failure is localised to the sentence.
      Rejected: asking a model to extract claims. It is better at finding the real
      propositions, and it adds a model call whose output varies run to run, ahead of a
      grader whose whole job is to be reproducible.
      Consequence: sentences carrying two propositions ("The cap is $1M and it does not
      apply to fraud") get one verdict for both. A partially supported sentence of that
      shape is graded on its weaker half, which errs toward reporting a problem.
    """
    import re

    stripped = answer.strip()
    if not stripped:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", stripped)
    claims: list[str] = []
    for sentence in sentences:
        text = sentence.strip()
        if len(text) < 20:
            continue
        # A sentence that is only a citation carries no claim.
        citations, _ = extract_citations(text)
        without = text
        for citation in citations:
            without = without.replace(citation.raw, "")
        if len(without.strip(" .,;:")) < 20:
            continue
        claims.append(text)
    return claims


def _excerpt_block(evidence: list[Evidence]) -> str:
    return "\n\n---\n\n".join(e.excerpt for e in evidence)


async def check_claim(
    claim: str,
    evidence: list[Evidence],
    client: ModelClient,
    *,
    live: bool,
) -> ClaimVerdict:
    """Check one claim against the evidence it was cited for."""
    if not live:
        return ClaimVerdict(
            claim=claim,
            verdict="unknown",
            quote="",
            reason="claim support requires CRA_LIVE_MODEL=1; not evaluated",
            quote_found_verbatim=False,
        )
    if not evidence:
        return ClaimVerdict(
            claim=claim,
            verdict="not_addressed",
            quote="",
            reason="no evidence was cited for this claim",
            quote_found_verbatim=False,
        )

    excerpts = _excerpt_block(evidence)
    user = f"Claim:\n{claim}\n\nExcerpts:\n{excerpts}"

    try:
        response = await client.complete(CLAIM_SUPPORT_SYSTEM, user)
    except ModelError as exc:
        return ClaimVerdict(claim, "unknown", "", f"grader call failed: {exc}", False)

    try:
        payload = json.loads(_extract_json(response.text))
        verdict = payload.get("verdict")
        if verdict not in ("supported", "contradicted", "not_addressed"):
            raise ValueError(f"unexpected verdict {verdict!r}")
        quote = str(payload.get("quote") or "")
        reason = str(payload.get("reason") or "")
    except (json.JSONDecodeError, ValueError, AttributeError) as exc:
        # A grader that cannot be parsed returns unknown. Defaulting to "supported"
        # would let a parse failure silently pass unsupported claims.
        return ClaimVerdict(claim, "unknown", "", f"unparseable grader output: {exc}", False)

    # DECISION: the quote is verified against the excerpts in code.
    #   The rubric asks for a verbatim quote; this checks that it actually is one. A
    #   model that fabricates its justification is exactly the failure this layer exists
    #   to catch, and trusting its self-report would reproduce the problem one level up.
    normalized_excerpts = " ".join(excerpts.split()).casefold()
    normalized_quote = " ".join(quote.split()).casefold()
    found = bool(normalized_quote) and normalized_quote in normalized_excerpts

    if verdict == "supported" and not found:
        return ClaimVerdict(
            claim=claim,
            verdict="not_addressed",
            quote=quote,
            reason=f"grader claimed support but its quote is not in the excerpts: {reason}",
            quote_found_verbatim=False,
        )

    return ClaimVerdict(claim, verdict, quote, reason, found)


def _extract_json(text: str) -> str:
    """Pull the first JSON object out of a response that may carry prose around it."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return text
    return text[start : end + 1]


async def check_answer(
    answer: str,
    evidence: list[Evidence],
    client: ModelClient,
    *,
    live: bool,
) -> list[ClaimVerdict]:
    """Check every factual claim in an answer."""
    return [await check_claim(c, evidence, client, live=live) for c in split_claims(answer)]
