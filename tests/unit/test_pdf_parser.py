"""PDF extraction and the scanned-document guard."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from app.ingestion.pdf_parser import (
    SCANNED_CHARS_PER_PAGE,
    ParsedDocument,
    ParsedPage,
    PdfParseError,
    parse_pdf,
)


def write_pdf(path: Path, pages: list[str]) -> Path:
    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text, fontsize=11)
    document.save(path)
    document.close()
    return path


class TestParsing:
    def test_extracts_text_with_one_based_page_numbers(self, tmp_path: Path) -> None:
        path = write_pdf(tmp_path / "a.pdf", ["First page text", "Second page text"])
        parsed = parse_pdf(path)
        assert parsed.page_count == 2
        assert parsed.pages[0].page_number == 1, "pages are numbered as a reader cites them"
        assert "First page" in parsed.pages[0].text
        assert "Second page" in parsed.pages[1].text

    def test_unreadable_file_raises_with_the_path(self, tmp_path: Path) -> None:
        broken = tmp_path / "broken.pdf"
        broken.write_bytes(b"this is not a pdf")
        with pytest.raises(PdfParseError, match=r"broken\.pdf"):
            parse_pdf(broken)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(PdfParseError):
            parse_pdf(tmp_path / "absent.pdf")


class TestScannedDetection:
    def test_pdf_with_no_text_layer_is_flagged_not_raised(self, tmp_path: Path) -> None:
        """At least one CUAD PDF is a scanned image (SPEC 6.2).

        It must be detected and excluded, or it becomes an unanswerable eval task for a
        parser reason rather than a contract reason. That is a corpus finding for the
        caller to record, so it is a flag rather than an exception.
        """
        path = write_pdf(tmp_path / "scan.pdf", ["", "", ""])
        parsed = parse_pdf(path)
        assert parsed.page_count == 3
        assert parsed.looks_scanned is True

    def test_normal_document_is_not_flagged(self, tmp_path: Path) -> None:
        body = "This Agreement is governed by the laws of Delaware. " * 6
        path = write_pdf(tmp_path / "ok.pdf", [body, body])
        assert parse_pdf(path).looks_scanned is False

    def test_threshold_is_per_page_not_total(self) -> None:
        """A long scanned document must not pass on accumulated stamp text."""
        pages = [ParsedPage(page_number=i, text="x" * 10) for i in range(1, 101)]
        parsed = ParsedDocument(source_path=Path("x.pdf"), pages=pages)
        assert parsed.char_count == 1000
        assert parsed.chars_per_page < SCANNED_CHARS_PER_PAGE
        assert parsed.looks_scanned is True

    def test_document_with_zero_pages_is_not_flagged_as_scanned(self) -> None:
        parsed = ParsedDocument(source_path=Path("x.pdf"), pages=[])
        assert parsed.looks_scanned is False
