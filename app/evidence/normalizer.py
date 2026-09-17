"""Evidence normalization. See docs/decisions.md

Deduplicate, group by topic, preserve conflicting evidence, assign stable ids.
"""

from __future__ import annotations

from app.schemas.evidence import Evidence
from app.schemas.retrieval import RetrievedChunk


def normalize_evidence(
    retrieved: list[RetrievedChunk],
    *,
    max_chars: int,
) -> list[Evidence]:
    """Turn retrieved chunks into the evidence set the writer sees.

    DECISION: conflicting evidence is preserved, never resolved here.
      SPEC 13.2 requires it and the reason is the conflicting-versions eval category: the
      corpus deliberately contains two versions of one agreement with different terms. If
      this function silently kept whichever version ranked higher, the system would give
      a confident single answer to a question that has two, and the conflict -- the
      actual finding -- would be invisible to the writer and to the reader.
      Deduplication is therefore on exact citable LOCATION, never on similar text.

    DECISION: cross-referenced evidence is ordered after searched evidence, not merged by
    score.
      Pulled chunks carry fused_score 0.0 because no retriever ranked them. Sorting the
      combined set by score would bury every carve-out at the bottom and, once the
      character budget bites, drop exactly the evidence cross-reference resolution
      exists to add. Retrieval order first, then pulled evidence, each preserving its
      own order.
    """
    searched = [i for i in retrieved if i.pulled_by == "search"]
    pulled = [i for i in retrieved if i.pulled_by == "cross_reference"]

    evidence: list[Evidence] = []
    seen: set[tuple[str, int | None, str | None]] = set()
    used = 0

    for index, item in enumerate(searched + pulled, start=1):
        chunk = item.chunk
        key = (chunk.document_id, chunk.page_number, chunk.section_id)
        if key in seen:
            continue

        # The budget drops whole items, never truncates an excerpt. A half-sentence
        # excerpt is worse than an absent one: the writer cannot tell it was cut, and a
        # cap truncated before its carve-out reads as unqualified.
        if used + len(chunk.text) > max_chars and evidence:
            continue

        seen.add(key)
        used += len(chunk.text)
        evidence.append(
            Evidence(
                evidence_id=f"e{index:02d}",
                topic=item.retrieval_query,
                document_id=chunk.document_id,
                document_title=chunk.document_title,
                section_path=chunk.section_path,
                section_title=chunk.section_title,
                page_number=chunk.page_number,
                excerpt=chunk.text,
                source_chunk_id=chunk.chunk_id,
                pulled_by=item.pulled_by,
            )
        )

    return evidence
