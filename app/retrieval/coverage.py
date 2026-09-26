"""Coverage constraints for explicit comparisons, never inferred from gold labels."""

from __future__ import annotations

import re

from app.schemas.retrieval import RetrievalFilters, RetrievedChunk

_COMPARISON = re.compile(
    r"\b(compare|comparison|differ\w*|across|each|versus|vs\.?|which\s+(?:of|agreement|contract))\b",
    re.IGNORECASE,
)


def comparison_documents(query: str, filters: RetrievalFilters | None) -> list[str]:
    if filters is None or not filters.document_ids or not _COMPARISON.search(query):
        return []
    documents = list(dict.fromkeys(filters.document_ids))
    return documents if len(documents) > 1 else []


def select_with_coverage(
    ranked: list[RetrievedChunk], document_ids: list[str], top_k: int
) -> list[RetrievedChunk]:
    """Reserve the best hit per requested document, then fill by relevance.

    If the number of documents exceeds k, relevance decides which documents fit.
    All returned chunks retain their original scores and source identities.
    """
    selected: set[str] = set()
    covered: set[str] = set()
    for item in ranked:
        if len(selected) >= top_k:
            break
        if item.document_id in document_ids and item.document_id not in covered:
            selected.add(item.chunk_id)
            covered.add(item.document_id)
    for item in ranked:
        if len(selected) >= top_k:
            break
        selected.add(item.chunk_id)
    return [item for item in ranked if item.chunk_id in selected][:top_k]
