"""ContractNLI adapter, including the span round-trip SPEC 6.3 requires.

The bug these guard against is silent: reading annotation spans as character offsets
yields valid substrings of the document at the wrong place, with no error raised.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ingestion.contractnli import (
    ContractNliFormatError,
    load_split,
)

# A miniature release with the same shape as the real one. Span 1 and span 2 are the
# evidence for nda-1; note the annotation refers to them by INDEX, not by offset.
FIXTURE = {
    "labels": {
        "nda-1": {
            "short_description": "Sharing with employees",
            "hypothesis": "Receiving Party may share information with employees.",
        },
        "nda-2": {
            "short_description": "Survival",
            "hypothesis": "Some obligations may survive termination.",
        },
    },
    "documents": [
        {
            "id": 1,
            "file_name": "example.pdf",
            "document_type": "search-pdf",
            "url": "https://example.invalid/example.pdf",
            # offsets:   0123456789...
            "text": "HEADER LINE\nFIRST SPAN TEXT\nSECOND SPAN TEXT\nTHIRD SPAN TEXT",
            "spans": [[0, 11], [12, 27], [28, 44], [45, 60]],
            "annotation_sets": [
                {
                    "annotations": {
                        "nda-1": {"choice": "Entailment", "spans": [1, 2]},
                        "nda-2": {"choice": "NotMentioned", "spans": []},
                    }
                }
            ],
        }
    ],
}


@pytest.fixture
def split(tmp_path: Path) -> Path:
    path = tmp_path / "train.json"
    path.write_text(json.dumps(FIXTURE), encoding="utf-8")
    return path


class TestSpanRoundTrip:
    def test_annotation_spans_resolve_to_their_literal_text(self, split: Path) -> None:
        """SPEC 6.3's required test. `spans: [1, 2]` means doc.spans[1] and doc.spans[2]."""
        _, documents = load_split(split)
        document = documents[0]
        annotation = document.annotations["nda-1"]

        assert annotation.span_indices == [1, 2]
        assert annotation.span_texts == ["FIRST SPAN TEXT", "SECOND SPAN TEXT"]

    def test_reading_indices_as_character_offsets_would_give_wrong_text(self, split: Path) -> None:
        """The bug, demonstrated. This is why the test above exists.

        Slicing text[1:2] is a legal operation returning a legal substring. Nothing
        raises. The result is simply a different part of the contract.
        """
        _, documents = load_split(split)
        document = documents[0]

        wrong = document.text[1:2]
        right = document.annotations["nda-1"].span_texts[0]

        assert wrong == "E"
        assert right == "FIRST SPAN TEXT"
        assert wrong != right

    def test_resolve_span_matches_the_parsed_annotation(self, split: Path) -> None:
        _, documents = load_split(split)
        document = documents[0]
        assert document.resolve_span(1) == "FIRST SPAN TEXT"
        assert document.resolve_span(3) == "THIRD SPAN TEXT"

    def test_out_of_range_index_raises_with_an_explanatory_message(self, split: Path) -> None:
        _, documents = load_split(split)
        with pytest.raises(ContractNliFormatError, match=r"indices into document\.spans"):
            documents[0].resolve_span(99)

    def test_offsets_are_retained_alongside_text(self, split: Path) -> None:
        _, documents = load_split(split)
        annotation = documents[0].annotations["nda-1"]
        assert annotation.span_offsets == [(12, 27), (28, 44)]


class TestLabels:
    def test_not_mentioned_carries_no_evidence(self, split: Path) -> None:
        """Expert-labelled abstention ground truth: the label IS the absence."""
        _, documents = load_split(split)
        annotation = documents[0].annotations["nda-2"]
        assert annotation.choice == "NotMentioned"
        assert annotation.span_texts == []
        assert annotation.has_evidence is False

    def test_entailment_carries_evidence(self, split: Path) -> None:
        _, documents = load_split(split)
        assert documents[0].annotations["nda-1"].has_evidence is True

    def test_hypotheses_are_loaded(self, split: Path) -> None:
        hypotheses, _ = load_split(split)
        assert hypotheses["nda-2"].hypothesis == "Some obligations may survive termination."


class TestValidation:
    def test_missing_file_names_the_download_command(self, tmp_path: Path) -> None:
        with pytest.raises(ContractNliFormatError, match=r"download_contractnli\.py"):
            load_split(tmp_path / "absent.json")

    def test_unexpected_choice_is_rejected(self, tmp_path: Path) -> None:
        payload = json.loads(json.dumps(FIXTURE))
        payload["documents"][0]["annotation_sets"][0]["annotations"]["nda-1"]["choice"] = "Maybe"
        path = tmp_path / "train.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ContractNliFormatError, match="unexpected choice"):
            load_split(path)

    def test_span_index_beyond_the_span_table_is_rejected(self, tmp_path: Path) -> None:
        """Catches a truncated or mismatched release rather than silently mislabelling."""
        payload = json.loads(json.dumps(FIXTURE))
        payload["documents"][0]["annotation_sets"][0]["annotations"]["nda-1"]["spans"] = [99]
        path = tmp_path / "train.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ContractNliFormatError, match="span index 99"):
            load_split(path)

    def test_malformed_json_is_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "train.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ContractNliFormatError, match="not valid JSON"):
            load_split(path)


class TestRealRelease:
    """Runs only when the corpus has been downloaded."""

    def test_round_trip_against_the_real_corpus(self) -> None:
        root = Path("data/raw/contract-nli")
        if not (root / "train.json").exists():
            pytest.skip("ContractNLI not downloaded")

        _, documents = load_split(root / "train.json")
        checked = 0
        for document in documents[:25]:
            for annotation in document.annotations.values():
                for index, text in zip(annotation.span_indices, annotation.span_texts, strict=True):
                    assert text == document.resolve_span(index)
                    assert text.strip(), "a resolved span must not be empty"
                    checked += 1
        assert checked > 0, "no annotated spans found to verify"
