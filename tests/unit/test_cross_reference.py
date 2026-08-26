"""Cross-reference resolution. SPEC 9.5, the component with the best failure story."""

from __future__ import annotations

from app.evidence.cross_reference import resolve_cross_references
from app.ingestion.store import ChunkStore
from tests.conftest import make_retrieved


def retrieved_from(store: ChunkStore, *chunk_ids: str) -> list:
    items = []
    for rank, chunk_id in enumerate(chunk_ids, start=1):
        chunk = store.get_chunk(chunk_id)
        assert chunk is not None
        items.append(make_retrieved(chunk, rank=rank))
    return items


class TestResolution:
    def test_pulls_the_carve_out_that_qualifies_the_cap(self, store: ChunkStore) -> None:
        """The case the whole component exists for.

        8.1 caps liability "Except as set out in Section 8.3". Retrieval returns 8.1
        because it matches "liability cap". An answer citing only 8.1 states an
        unqualified cap, cites a real document and page, and passes every deterministic
        check. Only this step brings 8.3 along.
        """
        report = resolve_cross_references(retrieved_from(store, "doc-900-c0001"), store)
        assert report.count == 1
        assert report.pulled[0].chunk.chunk_id == "doc-900-c0002"
        assert "8.3" in report.pulled[0].chunk.section_path

    def test_pulled_evidence_is_marked_for_ablation(self, store: ChunkStore) -> None:
        """SPEC 9.5 requires the contribution be measurable and ablatable."""
        report = resolve_cross_references(retrieved_from(store, "doc-900-c0001"), store)
        assert all(p.pulled_by == "cross_reference" for p in report.pulled)
        assert all(p.bm25_rank is None and p.dense_rank is None for p in report.pulled)
        assert all(p.fused_score == 0.0 for p in report.pulled), (
            "a pulled chunk was not ranked by any retriever; a synthetic score would "
            "make it sortable against real ones and hide where it came from"
        )

    def test_already_retrieved_targets_are_not_pulled_twice(self, store: ChunkStore) -> None:
        """Otherwise the ablation would over-credit this component."""
        both = retrieved_from(store, "doc-900-c0001", "doc-900-c0002")
        assert resolve_cross_references(both, store).count == 0


class TestBounds:
    def test_respects_the_chunk_budget(self, store: ChunkStore) -> None:
        report = resolve_cross_references(
            retrieved_from(store, "doc-900-c0001"), store, max_chunks=0
        )
        assert report.count == 0

    def test_depth_zero_disables_resolution(self, store: ChunkStore) -> None:
        """This is how the ablation arm is built."""
        report = resolve_cross_references(
            retrieved_from(store, "doc-900-c0001"), store, max_depth=0
        )
        assert report.count == 0
        assert report.followed == []

    def test_references_do_not_cross_document_boundaries(self, store: ChunkStore) -> None:
        """ "Section 8.3" means 8.3 of THIS contract. Resolving corpus-wide would attach
        evidence to a document that never said it."""
        report = resolve_cross_references(retrieved_from(store, "doc-900-c0001"), store)
        assert all(p.chunk.document_id == "doc-900" for p in report.pulled)

    def test_self_reference_is_ignored(self, store: ChunkStore) -> None:
        report = resolve_cross_references(retrieved_from(store, "doc-901-c0001"), store)
        assert report.count == 0


class TestReporting:
    def test_unresolvable_reference_is_recorded_not_dropped(self, store: ChunkStore) -> None:
        """An unresolved reference means our chunker missed a section the document
        itself cites -- a chunker-quality signal that belongs in the trace."""
        chunk = store.get_chunk("doc-900-c0003")
        assert chunk is not None
        item = make_retrieved(chunk.model_copy(update={"outbound_references": ["99.9"]}))
        report = resolve_cross_references([item], store)
        assert report.count == 0
        assert ("doc-900", "99.9") in report.skipped_unresolved
        assert "unresolved" in report.summary()

    def test_summary_is_readable_when_nothing_happened(self, store: ChunkStore) -> None:
        report = resolve_cross_references(retrieved_from(store, "doc-900-c0003"), store)
        assert "no outbound references" in report.summary()


class TestSpecificityOrdering:
    """A reference must never be crowded out of the budget by a subsection.

    `section_path` is derived by splitting on dots, so §11.7 is a descendant of §11 and
    "Section 11" resolves to both. Observed live on doc-003: the budget went to the
    referent plus an unrelated COUNTERPARTS subsection before a second reference was
    reached.
    """

    def test_exact_match_wins_the_budget_over_a_subsection(self, store: ChunkStore) -> None:
        from tests.conftest import make_chunk, make_document

        documents = [make_document("doc-910", page_count=3)]
        chunks = [
            make_chunk(
                "doc-910-c1",
                "doc-910",
                page_number=1,
                section_path=["8", "8.1"],
                text="8.1 Cap. Liability is capped, except as set out in Section 11.",
                outbound_references=["11"],
            ),
            # Document order puts the subsection FIRST, so only specificity ordering
            # can rescue the referent when the budget is 1.
            make_chunk(
                "doc-910-c2",
                "doc-910",
                page_number=2,
                section_path=["11", "11.7"],
                text="11.7 Counterparts. This Agreement may be executed in counterparts.",
            ),
            make_chunk(
                "doc-910-c3",
                "doc-910",
                page_number=3,
                section_path=["11"],
                text="11. Confidentiality. Each party shall keep the other's information secret.",
            ),
        ]
        scoped = ChunkStore(documents, chunks)
        source = make_retrieved(chunks[0])

        report = resolve_cross_references([source], scoped, max_chunks=1)
        assert report.count == 1
        assert report.pulled[0].chunk.section_id == "11", (
            "the referent lost the budget to its own subsection"
        )

    def test_subsections_still_arrive_when_budget_allows(self, store: ChunkStore) -> None:
        """Rejecting descendants outright would defeat the component: under a cap at
        8.3 the carve-outs often ARE 8.3(a)-(c)."""
        report = resolve_cross_references(
            [make_retrieved(store.get_chunk("doc-900-c0001"))], store, max_chunks=4
        )
        assert report.count >= 1
