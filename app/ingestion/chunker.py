"""Section-aware chunking.

AUTHOR-OWNED (CLAUDE.md rule 2). Every `# DECISION:` below is a choice you must be
able to defend, not a default to accept silently. See the handoff question list.

The chunker's job is to turn per-page extracted text into retrievable units that
each carry enough provenance to be cited and enough context to be understood. SPEC
9.3 constrains it:

  - Preserve page boundaries for citation.
  - Detect contract numbering (Article N, Section N.M, subsection (a)).
  - Prefer section and paragraph boundaries; split only when a section is too long.
  - Add overlap only to mechanically split sections.
  - Never merge text across unrelated sections.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.ingestion.pdf_parser import ParsedDocument, ParsedPage
from app.ingestion.text import estimate_tokens
from app.schemas.chunk import Chunk

# --- Section-heading detection -------------------------------------------------
#
# DECISION: detect headings with a small set of anchored regexes, not a general
# grammar and not an LLM call.
#   Chosen because contract numbering is genuinely regular in the CUAD corpus and a
#   regex is inspectable, free, and deterministic -- three properties an LLM call
#   does not have, on a path that runs over every page of every document.
#   Rejected: (a) an LLM structure pass, which costs money per page and makes
#   ingestion non-reproducible; (b) PDF font-size/bold heuristics via PyMuPDF's
#   "dict" mode, which is genuinely more robust to unnumbered headings but couples
#   the chunker to per-publisher styling and is much harder to unit test.
#   If the author switches: the fallback in `_page_chunks` already handles documents
#   where zero headings are found, so a better detector strictly improves results.

# "ARTICLE V", "ARTICLE 5", "Article 5 - Definitions".
# No en/em dash alternatives: clean_extracted_text folds them to "-" before the
# chunker ever sees the text. Adding them back here would be dead branches.
_ARTICLE = re.compile(
    r"^\s*(?:ARTICLE|Article|ARTICLE\s+NO\.?)\s+([IVXLCDM]+|\d+)\s*[.:\-]?\s*(.*)$"
)
# "8.", "8.1", "8.1.2", optionally "Section 8.1", with an optional trailing title.
_SECTION = re.compile(
    r"^\s*(?:SECTION|Section|SEC\.|Sec\.)?\s*(\d+(?:\.\d+){0,3})\s*[.:\-)]?\s+(\S.*)?$"
)
# "(a)", "(iv)" at the start of a line.
_SUBSECTION = re.compile(r"^\s*\(([a-z]{1,2}|[ivxlcdm]{1,5})\)\s+(\S.*)$")

# DECISION: a heading line must be short. A paragraph that merely begins with a
# number ("15 days after termination, ...") is not a heading. 90 characters is
# comfortably longer than a real contract heading and shorter than a sentence that
# happens to start with a numeral.
#   Rejected: requiring ALL-CAPS or a trailing colon -- too many CUAD headings are
#   title-case with no punctuation.
_MAX_HEADING_CHARS = 90

# --- Cross-reference extraction ------------------------------------------------
#
# Populates Chunk.outbound_references, which SPEC 9.5 consumes in Sprint 2 to pull
# qualifying carve-outs into the evidence set. Extracted at chunk time because the
# chunk text is the only place the reference appears in context.
_REFERENCE = re.compile(
    r"(?:Section|Sections|SECTION|Article|Articles|ARTICLE|clause|Clause|paragraph|Paragraph)"
    r"\s+(\d+(?:\.\d+){0,3}|[IVXLCDM]+)"
)


@dataclass
class _Block:
    """A contiguous run of text under one heading, on one page."""

    page_number: int
    section_path: list[str]
    section_title: str | None
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


def _roman_to_int(value: str) -> int | None:
    numerals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    if not value or any(ch not in numerals for ch in value.upper()):
        return None
    total = 0
    previous = 0
    for char in reversed(value.upper()):
        current = numerals[char]
        total = total - current if current < previous else total + current
        previous = max(previous, current)
    return total


def _match_heading(line: str, current_path: list[str]) -> tuple[list[str], str | None] | None:
    """Return (section_path, title) if `line` starts a new section, else None.

    DECISION: a numbered clause written as a long paragraph is still a section
    boundary, but only when its number is dotted.
      Measured on the first five contracts: 332 numbered lines are short enough to
      look like headings, while 455 more are long paragraphs that open with a dotted
      number -- "1.2 \"Acquired Party Family\" means ...". Treating only the short
      ones as boundaries loses the majority of section granularity, and with it most
      of what SPEC 9.5's cross-reference resolution has to resolve *to*: a reference
      to "Section 8.3" is useless if no chunk is labeled 8.3.
      The dotted requirement is the guard against false positives. On the same five
      contracts, 10 long lines open with a bare integer, and inspection shows they
      are prose, not headings -- "1 Clinical Trials, and Phase 2 ...", "Section 306
      of the FFDCA or analogous provisions ...". A bare integer opening a long line
      is far more often a sentence than a section, so it is rejected; a dotted
      number opening a long line is almost always a clause.
      Rejected: raising _MAX_HEADING_CHARS. It admits the prose cases too, and there
      is no length that separates them -- the distinguishing feature is the dot.
    """
    stripped = line.strip()
    if not stripped:
        return None
    is_short = len(stripped) <= _MAX_HEADING_CHARS

    if is_short and (match := _ARTICLE.match(stripped)):
        raw, title = match.group(1), (match.group(2) or "").strip()
        number = _roman_to_int(raw) if not raw.isdigit() else int(raw)
        if number is None:
            return None
        return [str(number)], title or None

    if match := _SECTION.match(stripped):
        dotted, title = match.group(1), (match.group(2) or "").strip()
        if not is_short:
            if "." not in dotted:
                return None
            # The remainder of a long line is the clause body, not a section title.
            # Storing a 200-character sentence as `section_title` would put prose
            # into a field that retrieval and citation rendering treat as a label.
            title = ""
        # DECISION: "8.1" implies membership in Article/Section 8, so the path is
        # built by splitting on dots rather than tracking enclosing headings.
        #   Chosen because it is robust to a missing or unparsed parent heading,
        #   which happens whenever an article title falls at a page break.
        #   Rejected: maintaining a heading stack across pages. More faithful to
        #   document structure, but a single missed ARTICLE line silently misfiles
        #   every subsequent section under the wrong parent, and the failure is
        #   invisible until a cross-reference resolves to the wrong place.
        parts = dotted.split(".")
        path = [".".join(parts[: i + 1]) for i in range(len(parts))]
        return path, title or None

    # DECISION: a subsection marker extends the enclosing section path; it never
    # replaces it, and it only counts as a boundary on a short line.
    #   An earlier version returned ["(a)"] alone, which dropped the parent entirely
    #   -- a carve-out in 8.3(a) became section "(a)", citable as §(a), which is not
    #   a location in any document.
    #   The short-line restriction matters more than it looks: "(a)", "(b)", "(c)"
    #   under a liability cap ARE the carve-outs. Splitting each into its own chunk
    #   severs the cap from its exceptions, which is precisely the failure SPEC 9.5
    #   exists to prevent. Long "(a) ..." paragraphs therefore stay inside the parent
    #   section's block, and are broken up only by the token-budget splitter.
    if is_short and (match := _SUBSECTION.match(stripped)):
        return [*current_path, f"({match.group(1)})"], (match.group(2) or "").strip() or None

    return None


def extract_references(text: str) -> list[str]:
    """Section identifiers cited inside `text`, deduplicated, in order of appearance."""
    seen: dict[str, None] = {}
    for match in _REFERENCE.finditer(text):
        value = match.group(1)
        if not value[0].isdigit():
            number = _roman_to_int(value)
            if number is None:
                continue
            value = str(number)
        seen.setdefault(value, None)
    return list(seen)


def _split_long_text(
    text: str, max_tokens: int, overlap_tokens: int, chars_per_token: float
) -> list[str]:
    """Split text that exceeds the token budget, on paragraph then sentence boundaries.

    DECISION: overlap is applied ONLY here, never between adjacent sections.
      SPEC 9.3 requires this, and the reason is worth stating: overlap exists to
      avoid severing a thought that had to be cut mechanically. Two different
      sections were never one thought, so overlapping them manufactures false
      adjacency -- a chunk that appears to contain both a liability cap and an
      unrelated indemnity, which is exactly the confusion the cross-reference
      resolver is meant to handle honestly.
    """
    max_chars = int(max_tokens * chars_per_token)
    if len(text) <= max_chars:
        return [text]

    # Prefer paragraph boundaries; fall back to sentence-ish boundaries.
    units = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    if any(len(u) > max_chars for u in units):
        units = [s for s in re.split(r"(?<=[.;])\s+", text) if s.strip()]

    parts: list[str] = []
    current: list[str] = []
    size = 0
    for unit in units:
        unit_len = len(unit) + 1
        if size + unit_len > max_chars and current:
            parts.append("\n\n".join(current).strip())
            if overlap_tokens > 0:
                tail = "\n\n".join(current)[-int(overlap_tokens * chars_per_token) :]
                current = [tail, unit]
                size = len(tail) + unit_len
            else:
                current = [unit]
                size = unit_len
        else:
            current.append(unit)
            size += unit_len
    if current:
        parts.append("\n\n".join(current).strip())

    # A single unit longer than the budget cannot be split on structure; cut it hard
    # rather than emit an oversized chunk that will be truncated downstream anyway.
    final: list[str] = []
    for part in parts:
        if len(part) <= max_chars:
            final.append(part)
            continue
        step = max_chars - int(overlap_tokens * chars_per_token) or max_chars
        final.extend(part[i : i + max_chars] for i in range(0, len(part), step))
    return [p for p in final if p.strip()]


def _page_blocks(
    page: ParsedPage, carried: tuple[list[str], str | None]
) -> tuple[list[_Block], tuple[list[str], str | None]]:
    """Split one page into heading-delimited blocks.

    DECISION: chunks never cross a page boundary.
      SPEC 9.3 requires page boundaries be preserved for citation, and the cheapest
      way to guarantee a chunk has exactly one page number is to never let it span
      two. The cost is real: a section straddling a page break becomes two chunks,
      and retrieval may return only the half that matches lexically.
      Rejected: section-spanning chunks with a page *range*. Better semantics, but
      then `Chunk.page_number` becomes `page_start`/`page_end`, the citation format
      in SPEC 13.3 has to express ranges, and the deterministic verifier's "page
      exists" check gets fuzzier. Not worth it before Experiment A says the split is
      costing recall.
      The `carried` heading is the mitigation: a block continuing from the previous
      page inherits that page's section path, so the second half is still labeled.
    """
    blocks: list[_Block] = []
    path, title = carried
    current = _Block(page_number=page.page_number, section_path=list(path), section_title=title)

    for line in page.text.splitlines():
        heading = _match_heading(line, path)
        if heading is not None:
            if current.lines:
                blocks.append(current)
            path, title = heading
            current = _Block(
                page_number=page.page_number, section_path=list(path), section_title=title
            )
            # Keep the heading line in the block: it carries the section title, which
            # is strong lexical signal for BM25 ("Governing Law", "Limitation of
            # Liability") and is what a reader would expect to see quoted.
            current.lines.append(line.strip())
        else:
            current.lines.append(line)

    if current.lines:
        blocks.append(current)
    return [b for b in blocks if b.text], (path, title)


def chunk_document(
    parsed: ParsedDocument,
    *,
    document_id: str,
    document_title: str,
    max_tokens: int,
    overlap_tokens: int,
    chars_per_token: float,
) -> list[Chunk]:
    """Chunk a parsed document into provenance-carrying, retrievable units."""
    chunks: list[Chunk] = []
    carried: tuple[list[str], str | None] = ([], None)
    ordinal = 0

    for page in parsed.pages:
        blocks, carried = _page_blocks(page, carried)

        # DECISION: a page with no detectable headings still produces chunks, using
        # the whole page as one block.
        #   A silently empty result for a badly-formatted document is the worst
        #   outcome: the document is in the corpus, the manifest says so, and it is
        #   simply unretrievable with nothing to indicate why. Degrading to
        #   page-sized chunks loses section_path but keeps document_id and
        #   page_number, so the document stays citable.
        if not blocks and page.text.strip():
            blocks = [
                _Block(
                    page_number=page.page_number,
                    section_path=list(carried[0]),
                    section_title=carried[1],
                    lines=page.text.splitlines(),
                )
            ]

        for block in blocks:
            for part in _split_long_text(block.text, max_tokens, overlap_tokens, chars_per_token):
                ordinal += 1
                chunks.append(
                    Chunk(
                        chunk_id=f"{document_id}-c{ordinal:04d}",
                        document_id=document_id,
                        document_title=document_title,
                        section_path=block.section_path,
                        section_title=block.section_title,
                        page_number=block.page_number,
                        text=part,
                        token_count=estimate_tokens(part, chars_per_token),
                        outbound_references=extract_references(part),
                    )
                )

    return chunks
