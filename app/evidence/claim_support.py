"""Citation-scoped claim support checking.

DECISION-BEARING MODULE. See docs/review-questions.md.

Layer 1 (``citation_verifier``) establishes that citation locators are valid and were
present in the selected evidence. This module supplies layer 2: deterministic claim
extraction and location resolution followed by a semantic judgement over *only* the
evidence named by each claim's own citations.

DECISION: a claim is judged against the evidence its own citations name, never against
  the whole selected-evidence pool.
  The pool-scoped predecessor (`claim_support@1`) asked "is this claim true somewhere in
  what we retrieved", which scores an answer as supported even when its citation points
  at an unrelated clause. That is the one failure mode citations exist to prevent, so the
  metric could not detect the defect it was named after.
  *Rejected:* keeping pool scope and adding a separate "citation relevance" metric, which
  leaves the headline number still reading as own-citation support; and document-level
  scoping, which passes any citation landing in a long agreement.
  *If reversed:* scores rise across the board and stop being comparable to any published
  citation-grounding number.

DECISION: locations resolve exactly -- document, page, and section must all match.
  Fuzzy resolution (nearest section, same page, same document) would silently repair the
  errors this grader exists to count.
  *Rejected:* falling back to the document when a section is absent, which would make an
  imprecise citation indistinguishable from a precise one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Literal

from app.agent.llm import ModelClient, ModelError
from app.evidence.citation_parser import (
    extract_citations,
    normalize_compound_citations,
    strip_citations,
)
from app.schemas.evidence import Citation, Evidence

Support = Literal[
    "supported",
    "contradicted",
    "not_addressed",
    "uncited",
    "malformed_citation",
    "citation_not_in_evidence",
    "unknown",
]
type EvidenceLocation = tuple[str, int | None, str | None]

CLAIM_SUPPORT_VERSION = "claim_support@2"
LEGACY_CLAIM_SUPPORT_VERSION = "claim_support@1"
CLAIM_SUPPORT_READABLE_VERSIONS = frozenset({LEGACY_CLAIM_SUPPORT_VERSION, CLAIM_SUPPORT_VERSION})


@dataclass(frozen=True, slots=True)
class CitedClaim:
    """A deterministically extracted factual claim and its attached citation attempts."""

    text: str
    citations: tuple[Citation, ...] = ()
    malformed_citations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ClaimVerdict:
    """One claim's support verdict and the exact evidence scope used to reach it."""

    claim: str
    verdict: Support
    quote: str
    reason: str
    quote_found_verbatim: bool
    citations: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()

    @property
    def is_problem(self) -> bool:
        """True unless the claim was affirmatively supported."""
        return self.verdict != "supported"


CLAIM_SUPPORT_SYSTEM = """You check whether contract excerpts support a specific claim.

You will be given one claim and the excerpts cited by that claim.

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

_BULLET_PREFIX = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_BRACKETED_SPAN = re.compile(r"\[[^\[\]]{0,120}\]")


def citation_location(citation: Citation) -> EvidenceLocation:
    """Return the normalized, exact location named by a citation."""
    section = citation.section_id.casefold() if citation.section_id is not None else None
    return (citation.document_id.casefold(), citation.page_number, section)


def evidence_location(evidence: Evidence) -> EvidenceLocation:
    """Return the normalized, exact location of one selected evidence item."""
    section = evidence.section_id.casefold() if evidence.section_id is not None else None
    return (evidence.document_id.casefold(), evidence.page_number, section)


def _deduplicate_citations(citations: list[Citation]) -> tuple[Citation, ...]:
    unique: dict[EvidenceLocation, Citation] = {}
    for citation in citations:
        unique.setdefault(citation_location(citation), citation)
    return tuple(unique.values())


def _deduplicate_strings(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _claim_prose(text: str, malformed: list[str]) -> str:
    prose = strip_citations(text)
    for span in malformed:
        prose = prose.replace(span, "")
    prose = _BULLET_PREFIX.sub("", prose).strip()
    return re.sub(r"\s+([.,;:!?])", r"\1", prose)


def _is_factual_prose(text: str) -> bool:
    """Apply a deterministic floor that excludes headings and bare yes/no text."""
    compact = text.strip(" \t.,;:!?()")
    return len(re.sub(r"\W", "", compact)) >= 5 and bool(re.search(r"[A-Za-z0-9]", compact))


def _split_sentences(line: str) -> list[str]:
    """Split sentences without treating ``p.`` inside a citation as punctuation."""
    spans: list[str] = []

    def shield(match: re.Match[str]) -> str:
        spans.append(match.group(0))
        return f"\ufdd0{len(spans) - 1}\ufdd1"

    shielded = _BRACKETED_SPAN.sub(shield, line)
    sentences = _SENTENCE_BOUNDARY.split(shielded)
    return [
        re.sub(
            r"\ufdd0(\d+)\ufdd1",
            lambda match: spans[int(match.group(1))],
            sentence,
        )
        for sentence in sentences
    ]


def extract_cited_claims(answer: str) -> list[CitedClaim]:
    """Split prose into claims while preserving citations attached to each claim.

    Compound citations are normalized first. Markdown bullets are split line-by-line,
    then sentences are split deterministically. A citation-only sentence is not a
    factual claim; when it immediately follows prose on the same line, its locators are
    attached to that preceding claim.

    DECISION: the claim unit is a sentence, split deterministically in code.
      A model-segmented claim unit would move the denominator of every support score
      into the model being graded: a run that emits fewer, vaguer claims would score
      better without answering better. A sentence is coarse -- a compound sentence can
      carry two propositions and is judged as one -- but it is stable across runs and
      reproducible from the answer text alone.
      *Rejected:* proposition-level extraction by a model (unstable denominator), and
      paragraph-level claims (too coarse to locate which part failed).
    """
    normalized = normalize_compound_citations(answer).strip()
    if not normalized:
        return []

    claims: list[CitedClaim] = []
    for raw_line in normalized.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        line_claim_indexes: list[int] = []
        for sentence in _split_sentences(line):
            text = sentence.strip()
            if not text:
                continue
            citations, malformed = extract_citations(text)
            prose = _claim_prose(text, malformed)

            if not _is_factual_prose(prose):
                if line_claim_indexes and (citations or malformed):
                    previous_index = line_claim_indexes[-1]
                    previous = claims[previous_index]
                    claims[previous_index] = replace(
                        previous,
                        citations=_deduplicate_citations([*previous.citations, *citations]),
                        malformed_citations=_deduplicate_strings(
                            [*previous.malformed_citations, *malformed]
                        ),
                    )
                continue

            claims.append(
                CitedClaim(
                    text=prose,
                    citations=_deduplicate_citations(citations),
                    malformed_citations=_deduplicate_strings(malformed),
                )
            )
            line_claim_indexes.append(len(claims) - 1)

    return claims


def split_claims(answer: str) -> list[str]:
    """Compatibility view of deterministic claim extraction."""
    return [claim.text for claim in extract_cited_claims(answer)]


def build_evidence_lookup(evidence: list[Evidence]) -> dict[EvidenceLocation, list[Evidence]]:
    """Index selected evidence by exact normalized citation location."""
    lookup: dict[EvidenceLocation, list[Evidence]] = {}
    for item in evidence:
        lookup.setdefault(evidence_location(item), []).append(item)
    return lookup


def evidence_for_claim(
    claim: CitedClaim,
    lookup: dict[EvidenceLocation, list[Evidence]],
) -> tuple[list[Evidence], tuple[Citation, ...]]:
    """Resolve a claim's citations without consulting any uncited location."""
    resolved: list[Evidence] = []
    missing: list[Citation] = []
    seen_ids: set[str] = set()
    for citation in claim.citations:
        matches = lookup.get(citation_location(citation), [])
        if not matches:
            missing.append(citation)
            continue
        for item in matches:
            if item.evidence_id not in seen_ids:
                seen_ids.add(item.evidence_id)
                resolved.append(item)
    return resolved, tuple(missing)


def _excerpt_block(evidence: list[Evidence]) -> str:
    return "\n\n---\n\n".join(e.excerpt for e in evidence)


def _deterministic_verdict(claim: CitedClaim, verdict: Support, reason: str) -> ClaimVerdict:
    return ClaimVerdict(
        claim=claim.text,
        verdict=verdict,
        quote="",
        reason=reason,
        quote_found_verbatim=False,
        citations=tuple(citation.render() for citation in claim.citations),
    )


async def check_claim(
    claim: CitedClaim,
    evidence: list[Evidence],
    client: ModelClient,
    *,
    live: bool,
) -> ClaimVerdict:
    """Semantically check one claim against its already citation-scoped evidence."""
    citation_renders = tuple(citation.render() for citation in claim.citations)
    evidence_ids = tuple(item.evidence_id for item in evidence)
    if not live:
        return ClaimVerdict(
            claim=claim.text,
            verdict="unknown",
            quote="",
            reason="claim support requires CRA_LIVE_MODEL=1; not evaluated",
            quote_found_verbatim=False,
            citations=citation_renders,
            evidence_ids=evidence_ids,
        )

    excerpts = _excerpt_block(evidence)
    user = f"Claim:\n{claim.text}\n\nExcerpts:\n{excerpts}"
    try:
        response = await client.complete(CLAIM_SUPPORT_SYSTEM, user)
    except ModelError as exc:
        return ClaimVerdict(
            claim.text,
            "unknown",
            "",
            f"grader call failed: {exc}",
            False,
            citation_renders,
            evidence_ids,
        )

    try:
        payload = json.loads(_extract_json(response.text))
        verdict = payload.get("verdict")
        if verdict not in ("supported", "contradicted", "not_addressed"):
            raise ValueError(f"unexpected verdict {verdict!r}")
        quote = str(payload.get("quote") or "")
        reason = str(payload.get("reason") or "")
    except (json.JSONDecodeError, ValueError, AttributeError) as exc:
        return ClaimVerdict(
            claim.text,
            "unknown",
            "",
            f"unparseable grader output: {exc}",
            False,
            citation_renders,
            evidence_ids,
        )

    # DECISION: a "supported" verdict survives only if its quote is found verbatim in the
    #   cited excerpts, checked here in Python rather than requested in the prompt.
    #   The prompt already demands a verbatim quote; the prompt is not the enforcement.
    #   A grader that paraphrases, or quotes from memory of the contract genre rather than
    #   from the excerpt, is exactly how a support metric inflates. Downgrading to
    #   "not_addressed" rather than "unknown" keeps the failure inside the score instead of
    #   excusing it from the denominator.
    #   Rejected: trusting the verdict (unverifiable), and re-asking the model to confirm
    #   its own quote (same model, same error, twice the cost).
    normalized_excerpts = " ".join(excerpts.split()).casefold()
    normalized_quote = " ".join(quote.split()).casefold()
    found = bool(normalized_quote) and normalized_quote in normalized_excerpts
    if verdict == "supported" and not found:
        return ClaimVerdict(
            claim.text,
            "not_addressed",
            quote,
            f"grader claimed support but its quote is not in the excerpts: {reason}",
            False,
            citation_renders,
            evidence_ids,
        )

    return ClaimVerdict(
        claim.text,
        verdict,
        quote,
        reason,
        found,
        citation_renders,
        evidence_ids,
    )


def _extract_json(text: str) -> str:
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
    """Check every factual claim using only evidence at its cited locations.

    DECISION: the three citation defects below are decided in code, before any model call.
      A factual claim with no citation, a malformed citation, and a citation naming a
      location that is not in the selected evidence are all determinable exactly. Sending
      them to a model would spend money to re-derive a fact already in hand, and would let
      a lenient judge grade an uncited sentence as supported because the pool happens to
      contain the answer -- the pool-scoped bug in a new place.
      *Rejected:* folding these into "not_addressed", which merges "the writer never cited
      this" with "the cited clause does not say it". They fail for different reasons and a
      failure taxonomy that cannot tell them apart cannot direct a fix.

    DECISION: an unavailable or unparseable judge yields "unknown", never "supported".
      "unknown" is not a passing verdict downstream, so a grader outage lowers the score
      and shows up as a problem, rather than quietly certifying every claim.
    """
    lookup = build_evidence_lookup(evidence)
    verdicts: list[ClaimVerdict] = []
    for claim in extract_cited_claims(answer):
        if claim.malformed_citations:
            verdicts.append(
                _deterministic_verdict(
                    claim,
                    "malformed_citation",
                    "claim contains a malformed citation attempt",
                )
            )
            continue
        if not claim.citations:
            verdicts.append(
                _deterministic_verdict(claim, "uncited", "factual claim has no citation")
            )
            continue

        cited_evidence, missing = evidence_for_claim(claim, lookup)
        if missing:
            rendered = ", ".join(citation.render() for citation in missing)
            verdicts.append(
                _deterministic_verdict(
                    claim,
                    "citation_not_in_evidence",
                    f"citation location is absent from selected evidence: {rendered}",
                )
            )
            continue
        verdicts.append(await check_claim(claim, cited_evidence, client, live=live))
    return verdicts
