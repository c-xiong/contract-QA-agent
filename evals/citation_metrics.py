"""Exact aggregation semantics for citation-gate ablations.

DECISION-BEARING MODULE. See docs/decisions.md.

DECISION: the ungated arm is scored by an independent audit of the writer's raw output,
  never by the runtime verifier's own report.
  The ungated arm has no verifier running, so asking the pipeline for its citation
  validity returns 1.0 by construction -- every parsed citation is approved because
  nothing checks any of them. The published `citation_validity = 1.0` for that arm was
  therefore a tautology, not a measurement, and made the gate look like it changed
  nothing. Auditing the raw answer separately gives the arm the failure rate it actually
  had, which is the only number that makes the gate's benefit visible.
  *Rejected:* running the verifier inside the ungated arm "just for reporting", which
  makes the arm no longer ungated and contaminates its latency and token counts.

DECISION: post-gate validity counts only answers the gated arm actually delivered.
  An answer the gate suppressed was never exposed to a caller, so counting its citations
  as valid would credit the gate for output it withheld. Suppression is reported instead
  as gate-induced abstention, where it can be weighed against over-abstention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from app.evidence.citation_verifier import VerificationResult
from app.evidence.claim_support import extract_cited_claims


@dataclass(frozen=True, slots=True)
class CitationAblationObservation:
    """Audited raw and final output for one paired ablation task."""

    ungated_answer: str
    ungated_audit: VerificationResult
    gated_answer: str
    gated_audit: VerificationResult
    gated_delivered: bool
    repair_triggered: bool


class CitationCoverageMetrics(TypedDict):
    gate_off: float
    gate_on: float
    gate_off_cited_claims: int
    gate_off_factual_claims: int
    gate_on_cited_claims: int
    gate_on_factual_claims: int


class CitationAblationMetrics(TypedDict):
    raw_invalid_citation_rate: float
    raw_invalid_citations: int
    raw_citation_attempts: int
    unsafe_answer_exposure_rate: float
    unsafe_answer_exposures: int
    tasks: int
    post_gate_citation_validity: float | None
    post_gate_valid_citations: int
    post_gate_citation_attempts: int
    citation_coverage: CitationCoverageMetrics
    repair_trigger_rate: float
    repair_triggers: int
    repair_success_rate: float | None
    repair_successes: int


def citation_attempts(result: VerificationResult) -> int:
    """Count parseable citations plus malformed citation attempts exactly once."""
    malformed = sum(error.code == "unparseable" for error in result.errors)
    return len(result.citations) + malformed


def _coverage_counts(answer: str, *, delivered: bool) -> tuple[int, int]:
    if not delivered:
        return 0, 0
    claims = extract_cited_claims(answer)
    return sum(bool(claim.citations) for claim in claims), len(claims)


def summarize_citation_ablation(
    observations: list[CitationAblationObservation],
) -> CitationAblationMetrics:
    """Compute independently audited citation-ablation metrics.

    The ungated verifier's self-report is deliberately absent. Raw validity comes from
    the independent auditor; final validity only includes answers actually delivered by
    the gated arm.
    """
    task_count = len(observations)
    raw_attempts = sum(citation_attempts(row.ungated_audit) for row in observations)
    raw_invalid = sum(len(row.ungated_audit.errors) for row in observations)

    unsafe = 0
    ungated_cited = ungated_claims = 0
    gated_cited = gated_claims = 0
    final_attempts = final_valid = 0
    repair_triggers = repair_successes = 0

    for row in observations:
        cited, claims = _coverage_counts(row.ungated_answer, delivered=True)
        ungated_cited += cited
        ungated_claims += claims
        missing_claim_citation = cited < claims
        if row.ungated_audit.errors or missing_claim_citation:
            unsafe += 1

        cited, claims = _coverage_counts(row.gated_answer, delivered=row.gated_delivered)
        gated_cited += cited
        gated_claims += claims

        if row.gated_delivered:
            final_attempts += citation_attempts(row.gated_audit)
            final_valid += len(row.gated_audit.valid_citations)

        if row.repair_triggered:
            repair_triggers += 1
            if row.gated_delivered and row.gated_audit.ok:
                repair_successes += 1

    return {
        "raw_invalid_citation_rate": raw_invalid / raw_attempts if raw_attempts else 0.0,
        "raw_invalid_citations": raw_invalid,
        "raw_citation_attempts": raw_attempts,
        "unsafe_answer_exposure_rate": unsafe / task_count if task_count else 0.0,
        "unsafe_answer_exposures": unsafe,
        "tasks": task_count,
        "post_gate_citation_validity": (final_valid / final_attempts if final_attempts else None),
        "post_gate_valid_citations": final_valid,
        "post_gate_citation_attempts": final_attempts,
        "citation_coverage": {
            "gate_off": ungated_cited / ungated_claims if ungated_claims else 1.0,
            "gate_on": gated_cited / gated_claims if gated_claims else 1.0,
            "gate_off_cited_claims": ungated_cited,
            "gate_off_factual_claims": ungated_claims,
            "gate_on_cited_claims": gated_cited,
            "gate_on_factual_claims": gated_claims,
        },
        "repair_trigger_rate": repair_triggers / task_count if task_count else 0.0,
        "repair_triggers": repair_triggers,
        "repair_success_rate": (repair_successes / repair_triggers if repair_triggers else None),
        "repair_successes": repair_successes,
    }
