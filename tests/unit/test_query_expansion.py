"""Vocabulary expansion keeps the original query, explicit scope, and source IDs."""

from app.retrieval.query_expansion import QueryExpandedRetriever, expand_contract_query
from app.schemas.retrieval import RetrievalFilters
from tests.conftest import make_chunk, make_retrieved


def test_unknown_topic_is_unchanged_and_expansion_is_bounded():
    assert expand_contract_query("Who signed this document?") == "Who signed this document?"
    query = "What jurisdiction applies, and may we exit without breach?"
    expanded = expand_contract_query(query)
    assert expanded.startswith(query)
    assert "governing law" in expanded
    assert "termination" in expanded
    assert len(expanded.split("Relevant contract terms: ")[1].split()) <= 48


async def test_original_and_expanded_queries_both_contribute_under_the_same_scope():
    class Base:
        name = "fixture"

        def __init__(self):
            self.calls = []

        async def search(self, query, top_k, filters=None):
            self.calls.append((query, filters))
            suffix = "expanded" if "Relevant contract terms:" in query else "original"
            return [make_retrieved(make_chunk(f"doc-900-{suffix}", "doc-900"))]

    base = Base()
    filters = RetrievalFilters(document_ids=["doc-900"], exclude_synthetic=True)
    hits = await QueryExpandedRetriever(base).search("Can we exit?", 5, filters)
    assert {hit.chunk_id for hit in hits} == {"doc-900-expanded", "doc-900-original"}
    assert len(base.calls) == 2
    assert all(call[1] is filters for call in base.calls)


async def test_unknown_topic_does_not_double_retrieval_calls():
    class Base:
        name = "fixture"
        calls = 0

        async def search(self, query, top_k, filters=None):
            self.calls += 1
            return []

    base = Base()
    await QueryExpandedRetriever(base).search("Who signed?", 5)
    assert base.calls == 1
