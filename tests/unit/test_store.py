"""The chunk store, which is what the citation verifier consults."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.store import ChunkStore, StoreError
from tests.conftest import make_chunk, make_document


class TestLookup:
    def test_page_lookup_reflects_indexed_pages_not_page_count(self, store: ChunkStore) -> None:
        """A page that produced no text produces no chunk, so nothing can be cited from it."""
        assert store.has_page("doc-900", 1) is True
        assert store.has_page("doc-900", 3) is True
        assert store.has_page("doc-900", 4) is False

    def test_section_lookup_covers_every_path_component(self, store: ChunkStore) -> None:
        assert store.has_section("doc-900", "8") is True
        assert store.has_section("doc-900", "8.1") is True
        assert store.has_section("doc-900", "9.9") is False

    def test_unknown_document(self, store: ChunkStore) -> None:
        assert store.has_document("doc-999") is False
        assert store.get_document("doc-999") is None
        assert store.has_page("doc-999", 1) is False

    def test_find_section_returns_chunks_in_document_order(self, store: ChunkStore) -> None:
        found = store.find_section("doc-900", "8.3")
        assert [c.chunk_id for c in found] == ["doc-900-c0002"]

    def test_find_page(self, store: ChunkStore) -> None:
        assert [c.chunk_id for c in store.find_page("doc-900", 2)] == ["doc-900-c0002"]


class TestIntegrity:
    def test_duplicate_chunk_id_is_rejected(self) -> None:
        with pytest.raises(StoreError, match="Duplicate chunk_id"):
            ChunkStore([make_document()], [make_chunk(), make_chunk()])

    def test_chunk_referencing_an_absent_document_is_rejected(self) -> None:
        """Catches a partial ingest before retrieval returns uncitable chunks."""
        with pytest.raises(StoreError, match="not in the store"):
            ChunkStore([make_document("doc-900")], [make_chunk(document_id="doc-901")])


class TestPersistence:
    def test_round_trips_preserving_provenance(self, store: ChunkStore, tmp_path: Path) -> None:
        store.save(tmp_path)
        loaded = ChunkStore.load(tmp_path)
        assert len(loaded) == len(store)

        original = store.get_chunk("doc-900-c0001")
        restored = loaded.get_chunk("doc-900-c0001")
        assert original is not None and restored is not None
        assert restored.section_path == original.section_path
        assert restored.page_number == original.page_number
        assert restored.outbound_references == original.outbound_references

    def test_missing_store_names_the_command_that_creates_it(self, tmp_path: Path) -> None:
        with pytest.raises(StoreError, match=r"ingest\.py"):
            ChunkStore.load(tmp_path)
