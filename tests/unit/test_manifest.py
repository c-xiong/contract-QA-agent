"""The corpus manifest's append-only invariant.

These tests exist because the failure they prevent is silent: a repointed document_id
still resolves to a document, so nothing raises and no other test fails, while every
eval task's ground truth quietly becomes wrong. See docs/decisions.md, 2026-08-25.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from app.ingestion.manifest import CorpusManifest, ManifestError, manifest_path


def add(manifest: CorpusManifest, name: str) -> str:
    entry, _ = manifest.add(
        source_filename=name,
        relative_path=f"CUAD_v1/full_contract_pdf/{name}",
        corpus_source="cuad",
        added=date(2026, 8, 25),
    )
    return entry.document_id


class TestIdAssignment:
    def test_ids_are_sequential_and_zero_padded(self) -> None:
        manifest = CorpusManifest()
        assert add(manifest, "a.pdf") == "doc-001"
        assert add(manifest, "b.pdf") == "doc-002"

    def test_re_adding_a_source_file_returns_its_existing_id(self) -> None:
        """Re-running the sampler over an overlapping selection must not renumber."""
        manifest = CorpusManifest()
        first = add(manifest, "a.pdf")
        second = add(manifest, "a.pdf")
        assert first == second
        assert len(manifest.documents) == 1

    def test_retired_ids_are_never_reused(self) -> None:
        """A retired ID stays spent, so a stale ground-truth label cannot be inherited."""
        manifest = CorpusManifest()
        add(manifest, "a.pdf")
        add(manifest, "b.pdf")
        manifest.retire("doc-002", reason="scanned, no text layer")

        assert add(manifest, "c.pdf") == "doc-003"
        assert manifest.get("doc-002").retired is True
        assert "doc-002" not in manifest.active

    def test_growing_the_corpus_preserves_existing_ids(self) -> None:
        manifest = CorpusManifest()
        original = {name: add(manifest, name) for name in ("a.pdf", "b.pdf", "c.pdf")}
        for name in ("d.pdf", "e.pdf"):
            add(manifest, name)
        for name, doc_id in original.items():
            assert manifest.find_by_source_filename(name) is not None
            assert manifest.find_by_source_filename(name).document_id == doc_id  # type: ignore[union-attr]


class TestLookup:
    def test_unknown_id_raises_rather_than_registering(self) -> None:
        with pytest.raises(ManifestError, match="Unknown document_id"):
            CorpusManifest().get("doc-999")

    def test_active_excludes_retired(self) -> None:
        manifest = CorpusManifest()
        add(manifest, "a.pdf")
        add(manifest, "b.pdf")
        manifest.retire("doc-001", reason="withdrawn")
        assert set(manifest.active) == {"doc-002"}


class TestPersistence:
    def test_round_trips(self, tmp_path: Path) -> None:
        manifest = CorpusManifest(sampling_criterion="test criterion")
        add(manifest, "a.pdf")
        path = manifest_path(tmp_path)
        manifest.save(path)

        loaded = CorpusManifest.load(path)
        assert loaded.sampling_criterion == "test criterion"
        assert loaded.get("doc-001").source_filename == "a.pdf"

    def test_keys_are_sorted_so_diffs_stay_reviewable(self, tmp_path: Path) -> None:
        manifest = CorpusManifest()
        for name in ("c.pdf", "a.pdf", "b.pdf"):
            add(manifest, name)
        path = manifest_path(tmp_path)
        manifest.save(path)
        keys = list(json.loads(path.read_text())["documents"])
        assert keys == sorted(keys)

    def test_missing_manifest_names_the_command_that_creates_it(self, tmp_path: Path) -> None:
        with pytest.raises(ManifestError, match=r"sample_contracts\.py"):
            CorpusManifest.load(manifest_path(tmp_path))

    def test_hand_edited_key_mismatch_is_caught(self, tmp_path: Path) -> None:
        path = manifest_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "manifest_version": 1,
                    "sampling_criterion": "",
                    "documents": {
                        "doc-001": {
                            "document_id": "doc-002",
                            "source_filename": "a.pdf",
                            "relative_path": "a.pdf",
                            "corpus_source": "cuad",
                            "annotated_categories": [],
                            "added": "2026-08-25",
                        }
                    },
                }
            )
        )
        with pytest.raises(ManifestError, match=r"does not match entry\.document_id"):
            CorpusManifest.load(path)

    def test_one_source_file_cannot_hold_two_ids(self, tmp_path: Path) -> None:
        manifest = CorpusManifest()
        add(manifest, "a.pdf")
        # Force the invariant violation the add() path prevents.
        entry = manifest.get("doc-001")
        manifest.documents["doc-002"] = entry.model_copy(update={"document_id": "doc-002"})
        with pytest.raises(ManifestError, match="registered under both"):
            manifest.save(manifest_path(tmp_path))

    def test_malformed_json_is_reported_with_the_path(self, tmp_path: Path) -> None:
        path = manifest_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json")
        with pytest.raises(ManifestError, match="not valid JSON"):
            CorpusManifest.load(path)
