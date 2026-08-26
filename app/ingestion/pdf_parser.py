"""PDF text extraction with page boundaries preserved.

SPEC 6.2 requires ingesting the PDFs rather than CUAD's bundled plaintext, so the
parser is exercised on real layout problems: running headers, footers, page breaks
mid-clause, and signature blocks. It also requires detecting documents with no text
layer, because at least one CUAD PDF is a scanned image and would otherwise become
an unanswerable eval task for a parser reason rather than a contract reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf

from app.ingestion.text import clean_extracted_text

# Below this many extracted characters per page on average, a document is treated as
# having no usable text layer. A born-digital contract page yields hundreds to
# thousands of characters; a scanned page yields zero, or a handful from a stamp.
# The gap is wide enough that the exact threshold does not matter much.
SCANNED_CHARS_PER_PAGE = 50


class PdfParseError(RuntimeError):
    """The file could not be opened or produced no usable text."""


@dataclass(frozen=True, slots=True)
class ParsedPage:
    """One page of extracted text. ``page_number`` is 1-based, as a reader would cite it."""

    page_number: int
    text: str

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """The full extraction result for one PDF."""

    source_path: Path
    pages: list[ParsedPage]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def char_count(self) -> int:
        return sum(page.char_count for page in self.pages)

    @property
    def chars_per_page(self) -> float:
        return self.char_count / self.page_count if self.page_count else 0.0

    @property
    def looks_scanned(self) -> bool:
        """True when the PDF has pages but effectively no text layer."""
        return self.page_count > 0 and self.chars_per_page < SCANNED_CHARS_PER_PAGE

    def full_text(self) -> str:
        return "\n".join(page.text for page in self.pages)


def parse_pdf(path: Path) -> ParsedDocument:
    """Extract per-page text from a PDF.

    Raises PdfParseError if the file cannot be opened. A file that opens but yields
    no text is returned normally with ``looks_scanned`` True: that is a corpus
    quality finding for the caller to record and act on, not an exception.
    """
    try:
        document = pymupdf.open(path)
    except (RuntimeError, OSError) as exc:
        raise PdfParseError(f"Cannot open {path}: {exc}") from exc

    pages: list[ParsedPage] = []
    try:
        for index in range(document.page_count):
            # "text" mode preserves reading order and line breaks, which the chunker
            # needs for section-number detection. Layout modes reflow into columns and
            # destroy exactly that signal.
            raw: str = document.load_page(index).get_text("text")
            pages.append(ParsedPage(page_number=index + 1, text=clean_extracted_text(raw)))
    finally:
        document.close()

    return ParsedDocument(source_path=path, pages=pages)
