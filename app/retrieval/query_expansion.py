"""Bounded, auditable contract vocabulary expansion without model calls.

Rules refer to general legal concepts, never document IDs, parties, eval task IDs,
answer strings, or reference labels. The original query remains a retrieval arm.
"""

from __future__ import annotations

import asyncio
import re

from app.retrieval.base import Retriever
from app.retrieval.fusion import reciprocal_rank_fusion
from app.schemas.retrieval import RetrievalFilters, RetrievedChunk

_CONCEPTS = (
    (
        r"\b(?:effective|execution|commencement)\s+date\b",
        "effective date execution date entered into as of dated",
    ),
    (
        r"\b(?:jurisdiction|governing law|choice of law)\b",
        "governing law governed construed laws state",
    ),
    (
        r"\b(?:terminat\w*|walk away|leave|exit)\b",
        "termination terminate without cause for convenience notice",
    ),
    (
        r"\b(?:loss(?:es)?|damages|liability|recover\w*|risk.allocation|exposure)\b",
        "limitation of liability consequential indirect damages lost profits exceptions indemnification",
    ),
    (r"\b(?:insurance|coverage)\b", "insurance coverage certificate policy"),
    (
        r"\b(?:renew\w*|rolling over|end of the term)\b",
        "renewal expiration notice period prior written notice",
    ),
    (r"\b(?:non.compete|exclusiv\w*|competing)\b", "non-compete exclusivity competing business"),
    (
        r"\b(?:acquired|change of control|acquisition|assignment)\b",
        "change of control acquisition assignment consent",
    ),
    (
        r"\b(?:confidential\w*|nda)\b",
        "confidential information definition disclosed marked identified",
    ),
    (
        r"\b(?:rights and obligations|accrued|surviv\w*|deal ended)\b",
        "survival accrued rights obligations expiration termination",
    ),
)
_RULES = [(re.compile(pattern, re.IGNORECASE), expansion) for pattern, expansion in _CONCEPTS]


def expand_contract_query(query: str) -> str:
    terms = [expansion for pattern, expansion in _RULES if pattern.search(query)]
    if not terms:
        return query
    # Bound extra work even for questions touching many concepts.
    vocabulary = list(dict.fromkeys(" ".join(terms).split()))[:48]
    return query + "\nRelevant contract terms: " + " ".join(vocabulary)


class QueryExpandedRetriever:
    """Fuse original and expanded query results; never change source text or scope."""

    def __init__(self, base: Retriever) -> None:
        self._base = base
        self.name = base.name + "+legal-query"

    async def search(
        self, query: str, top_k: int, filters: RetrievalFilters | None = None
    ) -> list[RetrievedChunk]:
        expanded = expand_contract_query(query)
        if expanded == query:
            return await self._base.search(query, top_k, filters)
        pools = await asyncio.gather(
            self._base.search(query, top_k, filters), self._base.search(expanded, top_k, filters)
        )
        return reciprocal_rank_fusion(list(pools))[:top_k]
