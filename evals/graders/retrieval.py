"""Retrieval graders. See docs/SPEC.md section 15.4.

AUTHOR-OWNED (CLAUDE.md rule 2, .claude/rules/evals.md). Every `# DECISION:` is a
choice to defend, and the thresholds are yours to set.

Graders are pure functions over (task, result). No I/O, no model calls, no hidden
state -- so a grader can be unit tested against hand-constructed fixtures, and a
result can be re-graded from a stored artifact without re-running the agent.

Every grader carries a version. When a grader changes, old numbers do not silently
become comparable to new ones; `grader_versions` is recorded with each result so a
regression can be traced to a grader change rather than a system change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.ingestion.text import normalize_for_matching
from app.schemas.retrieval import RetrievedChunk
from evals.schema import EvalTask

# Bump when the computation changes in a way that makes old scores incomparable.
DOCUMENT_RECALL_VERSION = "document_recall@1"
EVIDENCE_RECALL_VERSION = "evidence_span_recall@1"
MRR_VERSION = "mrr@1"

# DECISION: a span counts as retrieved when a normalized 60-character window of it
# appears in a retrieved chunk.
#   TODO(author): confirm 60. It is long enough that a match is not coincidence and
#   short enough to survive the chunker splitting a long clause across two chunks.
#   The measured clause-span match rate (91.5%, see docs/eval-methodology.md) was
#   computed on whole spans; a shorter probe is strictly more forgiving.
#   Rejected: whole-span containment, which scores 0 whenever a clause straddles a
#   chunk boundary -- penalizing the grader's own chunking rather than retrieval.
#   Rejected: token-overlap ratio (e.g. Jaccard > 0.5), which is more robust but
#   introduces a second threshold to defend and is not obviously better here.
SPAN_PROBE_CHARS = 60


@dataclass(frozen=True)
class GradeResult:
    """One grader's verdict on one task."""

    name: str
    version: str
    score: float
    passed: bool
    detail: str
    data: dict[str, object] = field(default_factory=dict)


def _normalized_chunk_texts(retrieved: list[RetrievedChunk]) -> list[str]:
    return [normalize_for_matching(item.chunk.text) for item in retrieved]


def document_recall(task: EvalTask, retrieved: list[RetrievedChunk], k: int) -> GradeResult:
    """Fraction of expected documents appearing in the top k retrieved chunks.

    DECISION: k counts retrieved *chunks*, not distinct documents.
      "Recall@5" in this project means "within the 5 chunks the agent actually saw",
      because that is the budget the agent operates under. Counting distinct documents
      instead would report a number the agent never had access to.

    DECISION: this grader scores 1.0 on an abstain task with nothing retrieved.
      An abstain task has no expected documents. Scoring it 0.0 would punish correct
      behavior; leaving it ungraded would silently shrink n. It is reported as a pass
      with an explicit note so the caption's task count stays honest.
    """
    expected = set(task.expected_document_ids)
    top = retrieved[:k]
    found = {item.chunk.document_id for item in top}

    if not expected:
        return GradeResult(
            name="document_recall",
            version=DOCUMENT_RECALL_VERSION,
            score=1.0,
            passed=True,
            detail="no expected documents (abstain task); trivially satisfied",
            data={"k": k, "retrieved_documents": sorted(found)},
        )

    hits = expected & found
    score = len(hits) / len(expected)
    # TODO(author): set the pass threshold. 1.0 means "every expected document was
    # retrieved" -- strict, and appropriate for single-document lookup. A
    # cross-document comparison task over 5 contracts may deserve partial credit.
    return GradeResult(
        name="document_recall",
        version=DOCUMENT_RECALL_VERSION,
        score=score,
        passed=score == 1.0,
        detail=f"{len(hits)}/{len(expected)} expected documents in top {k}",
        data={
            "k": k,
            "expected": sorted(expected),
            "found": sorted(found),
            "missing": sorted(expected - found),
        },
    )


def evidence_span_recall(task: EvalTask, retrieved: list[RetrievedChunk], k: int) -> GradeResult:
    """Fraction of expected evidence spans present in the top k retrieved chunks.

    This is the grader that distinguishes "found the right contract" from "found the
    right clause". Document recall can be 1.0 while every retrieved chunk is the wrong
    part of the right document, which is a failure mode a document-level metric cannot
    see and a page citation would make look authoritative.
    """
    spans = [e for e in task.expected_evidence if e.span_text]
    if not spans:
        return GradeResult(
            name="evidence_span_recall",
            version=EVIDENCE_RECALL_VERSION,
            score=1.0,
            passed=True,
            detail="no expected spans; not applicable",
            data={"k": k},
        )

    haystacks = _normalized_chunk_texts(retrieved[:k])
    found: list[str] = []
    missing: list[str] = []

    for span in spans:
        assert span.span_text is not None
        probe = normalize_for_matching(span.span_text)[:SPAN_PROBE_CHARS]
        if probe and any(probe in text for text in haystacks):
            found.append(probe[:40])
        else:
            missing.append(probe[:40])

    score = len(found) / len(spans)
    return GradeResult(
        name="evidence_span_recall",
        version=EVIDENCE_RECALL_VERSION,
        score=score,
        passed=score == 1.0,
        detail=f"{len(found)}/{len(spans)} expected spans found in top {k}",
        data={"k": k, "found": found, "missing": missing, "probe_chars": SPAN_PROBE_CHARS},
    )


def reciprocal_rank(task: EvalTask, retrieved: list[RetrievedChunk]) -> GradeResult:
    """1/rank of the first chunk from any expected document, or 0.0 if absent.

    Complements recall: recall says whether the right document was reachable at all,
    MRR says whether the agent had to read past four wrong chunks to reach it. With a
    5-chunk budget those are very different situations.
    """
    expected = set(task.expected_document_ids)
    if not expected:
        return GradeResult(
            name="reciprocal_rank",
            version=MRR_VERSION,
            score=1.0,
            passed=True,
            detail="no expected documents (abstain task); not applicable",
        )

    for rank, item in enumerate(retrieved, start=1):
        if item.chunk.document_id in expected:
            return GradeResult(
                name="reciprocal_rank",
                version=MRR_VERSION,
                score=1.0 / rank,
                passed=True,
                detail=f"first expected document at rank {rank}",
                data={"rank": rank},
            )

    return GradeResult(
        name="reciprocal_rank",
        version=MRR_VERSION,
        score=0.0,
        passed=False,
        detail="no expected document retrieved",
    )


def grade_retrieval(task: EvalTask, retrieved: list[RetrievedChunk], k: int) -> list[GradeResult]:
    """Run every retrieval grader over one task's result."""
    return [
        document_recall(task, retrieved, k),
        evidence_span_recall(task, retrieved, k),
        reciprocal_rank(task, retrieved),
    ]
