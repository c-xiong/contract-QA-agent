"""Text normalization. Shared by the parser, chunker, and span matcher."""

from __future__ import annotations

import pytest

from app.ingestion.text import (
    clean_extracted_text,
    estimate_tokens,
    fold_punctuation,
    normalize_for_matching,
)


def test_typographic_punctuation_folds_to_ascii() -> None:
    assert fold_punctuation("“quoted” ‘word’") == "\"quoted\" 'word'"
    assert fold_punctuation("30–60 days") == "30-60 days"


def test_non_breaking_and_zero_width_spaces_normalize() -> None:
    assert fold_punctuation("a b") == "a b"
    assert fold_punctuation("a​b") == "ab"


def test_normalize_collapses_whitespace_and_casefolds() -> None:
    assert normalize_for_matching("  Governing   LAW\n\n applies ") == "governing law applies"


def test_normalize_repairs_words_split_across_a_line_break() -> None:
    """PDF extraction hyphenates at line ends; a naive matcher misses every such span."""
    assert "indemnify" in normalize_for_matching("shall indem-\nnify the other party")


def test_normalize_makes_annotation_and_extraction_comparable() -> None:
    """The case this function exists for: CUAD's ASCII vs the PDF's typography."""
    annotation = 'The "Term" shall be 30-60 days.'
    extracted = "The “Term” shall  be\n30–60   days."
    assert normalize_for_matching(annotation) == normalize_for_matching(extracted)


def test_clean_extracted_text_preserves_line_structure() -> None:
    """The chunker detects headings on line starts, so newlines must survive."""
    cleaned = clean_extracted_text("8.1 Liability\n\n\n\n\nText here   \n")
    assert cleaned.startswith("8.1 Liability")
    assert "\n" in cleaned, "line breaks are load-bearing for heading detection"
    # At most two consecutive blank lines survive, i.e. never four newlines in a row.
    assert "\n\n\n\n" not in cleaned
    assert cleaned.endswith("Text here"), "trailing whitespace is stripped"


def test_estimate_tokens_scales_with_length() -> None:
    assert estimate_tokens("", 4.0) == 0
    assert estimate_tokens("a" * 400, 4.0) == 100


def test_estimate_tokens_rejects_nonpositive_divisor() -> None:
    with pytest.raises(ValueError, match="positive"):
        estimate_tokens("text", 0.0)
