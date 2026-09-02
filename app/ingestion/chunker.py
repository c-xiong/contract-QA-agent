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


def _strip_decimal_article(parts: list[str]) -> list[str]:
    """Drop trailing zero components: "9.0" is Article 9. Never empties the list."""
    while len(parts) > 1 and parts[-1] == "0":
        parts = parts[:-1]
    return parts


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
    # True when a detected heading opened this block, as opposed to the block being
    # the continuation carried in from the previous page. Only a heading-opened block
    # can be a *bare* heading; a carried block's single line is body text.
    from_heading: bool = False

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()

    @property
    def is_bare_heading(self) -> bool:
        """The block holds its heading line and no body."""
        return self.from_heading and sum(1 for line in self.lines if line.strip()) == 1


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


def _match_heading(
    line: str, current_path: list[str], *, logical_length: int | None = None
) -> tuple[list[str], str | None] | None:
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

    DECISION: "short" is a property of the logical paragraph, not of the physical line.
      `logical_length` is the length of the whole paragraph this line opens, computed by
      `_logical_lengths` and supplied by `_page_blocks`; callers that pass nothing fall
      back to the line's own length, which is correct only for unwrapped text.
      This exists because the corpus is hard-wrapped: the median physical line in the
      processed CUAD text is 76 characters, comfortably under _MAX_HEADING_CHARS. So the
      first line of *every* long subsection looked short --
      "(a)      FOR ANY BREACH OR DEFAULT BY CHANGEPOINT OF ANY OF THE PROVISIONS OF"
      is 77 characters, while the subsection it opens is 1137 -- and the guard below
      that keeps a carve-out attached to the cap it qualifies never fired on real data.
      Measured before the fix: 79 chunks were a section heading severed from its own
      body, and section titles like "OTHER THAN THE WARRANTIES EXPRESSLY SET FORTH IN
      SECTION 6.1(A) AND" were wrap-truncated prose stored in a label field.
      Rejected: unwrapping the page text before chunking. It fixes the measurement the
      same way, but rewrites every chunk's `text`, so quoted spans no longer match the
      source byte-for-byte and every citation in a stored trace has to be re-verified.
      Measuring differently changes which boundaries are found; it does not change what
      the chunk says.
    """
    stripped = line.strip()
    if not stripped:
        return None
    is_short = (len(stripped) if logical_length is None else logical_length) <= _MAX_HEADING_CHARS

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
        # DECISION: a trailing ".0" is an article heading written in decimal form, so
        # it is stripped: "9.0" is Article 9, not subsection 0 of Article 9.
        #   Three documents number their articles this way -- "1.0 DEFINITIONS.",
        #   "6.0 Force Majeure.", "2.0 IBM Services Responsibilities" -- and every one
        #   of the 101 chunks involved is an article heading, never a numbered clause
        #   beneath one. Without the strip, "9.0 Limitation of Liability" gets the path
        #   ["9", "9.0"] while the clause below it gets ["9", "9.1"], the two are
        #   siblings rather than parent and child, and `_fold_bare_headings` cannot
        #   reunite them -- so the heading survives as a 7-token chunk that still
        #   outranks real text on a liability question. Stripping makes the heading
        #   ["9"], the clause its descendant, and the existing fold does the rest.
        #   `extract_references` strips the same way, so "Section 9.0" still resolves.
        #   Rejected: teaching the fold to merge siblings. `1.64 "Term" means ten
        #   years.` and `1.65 "Territory" means Japan.` are structurally identical to
        #   the 9.0/9.1 pair -- same shape, same token count -- and are two complete
        #   definitions that must stay apart. The numbering is the only real signal.
        parts = _strip_decimal_article(dotted.split("."))
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


def _opens_paragraph(line: str) -> bool:
    """True if `line` begins a new logical paragraph rather than continuing one.

    DECISION: continuation is decided on the physical line, by the same heading
    regexes, and deliberately not recursively on logical length.
      A continuation run has to terminate at *something*, and "a line that would read
      as a heading on its own" is the only local, non-circular test available. A false
      positive here (a wrapped line opening "10 days after notice...") only truncates
      the run early, which under-measures the paragraph and degrades that one case back
      to the pre-fix behavior -- never to something worse.
    """
    return _match_heading(line, []) is not None


def _logical_lengths(lines: list[str]) -> list[int]:
    """For each line, the length of the logical paragraph it opens.

    A paragraph is the line plus every following non-blank line that does not itself
    open one. Blank lines measure zero; they are never headings.
    """
    total = len(lines)
    # continuation[i] = combined length of the continuation run starting at line i.
    continuation = [0] * (total + 1)
    for index in range(total - 1, -1, -1):
        stripped = lines[index].strip()
        if not stripped or _opens_paragraph(stripped):
            continuation[index] = 0
        else:
            continuation[index] = len(stripped) + 1 + continuation[index + 1]
    return [len(lines[i].strip()) + continuation[i + 1] for i in range(total)]


def _is_descendant(child: list[str], parent: list[str]) -> bool:
    return len(child) > len(parent) and child[: len(parent)] == parent


def _fold_bare_headings(blocks: list[_Block]) -> list[_Block]:
    """Fold a heading-only block into the descendant block that carries its body.

    DECISION: a chunk is never just a section heading when the body is one block away.
      A heading-only chunk is worse than useless in retrieval, because it is not merely
      empty -- it is a *strong lexical match* for the topic it names. "6.2 LIMIT OF
      LIABILITY" outranks most real text on a liability question, is selected as
      evidence, is cited, and then passes the citation verifier, because the quoted span
      genuinely does occur in the chunk. The gate cannot catch it: the citation is
      valid. Only the content is absent. So the invariant belongs here, at the point
      where the empty unit would otherwise be created.
      The merged block takes the *descendant's* section path, not the heading's, because
      that is where the text actually lives, and `ChunkStore.find_section` matches on
      path containment -- so a reference to "6.2" still resolves to a chunk labeled
      6.2(a). Taking the ancestor path would make the citation less precise for no gain.
      Rejected: (a) dropping the heading line, which loses the section title that is the
      strongest BM25 signal the chunk has; (b) a minimum-token filter at evidence
      selection, which is downstream of the defect and would also discard legitimately
      short chunks -- '1.65 "Territory" means Japan.' is 7 tokens and complete.
      Not handled here: a heading whose body starts on the next page, and short
      enumerated siblings ("(a) bankruptcy;"). Both need merges this function is not
      allowed to make -- across a page boundary, and between siblings -- see
      docs/review-questions.md.
    """
    folded: list[_Block] = []
    pending: list[str] = []
    for index, block in enumerate(blocks):
        following = blocks[index + 1] if index + 1 < len(blocks) else None
        if (
            block.is_bare_heading
            and following is not None
            and _is_descendant(following.section_path, block.section_path)
        ):
            pending.extend(block.lines)
            continue
        if pending:
            block.lines = [*pending, *block.lines]
            pending = []
        folded.append(block)
    if pending:  # pragma: no cover - pending only accrues when a following block exists
        folded.append(_Block(blocks[-1].page_number, blocks[-1].section_path, None, pending))
    return folded


def extract_references(text: str) -> list[str]:
    """Section identifiers cited inside `text`, deduplicated, in order of appearance.

    Identifiers are normalized the same way `_match_heading` builds a path -- roman
    numerals to integers, trailing ".0" stripped -- because a reference is only useful
    if it is spelled the way the target chunk is labeled. 19 references in the corpus
    are of the form "Section 9.0"; without the strip they resolve to nothing.
    """
    seen: dict[str, None] = {}
    for match in _REFERENCE.finditer(text):
        value = match.group(1)
        if not value[0].isdigit():
            number = _roman_to_int(value)
            if number is None:
                continue
            value = str(number)
        seen.setdefault(".".join(_strip_decimal_article(value.split("."))), None)
    return list(seen)


# DECISION: 20 estimated tokens is the floor below which a *fragment* is merged back
# into the part before it. It is not a floor on chunks: a whole section that is genuinely
# short stays short -- '1.65 "Territory" means Japan.' is 7 tokens and complete. The floor
# only applies to a tail the splitter itself produced, where the text before it is the
# rest of the same sentence.
_MIN_CHUNK_TOKENS = 20


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

    # DECISION: a trailing fragment is merged back into the part before it, not emitted.
    #   Both loops above can leave a tail of a few characters -- the unit loop when the
    #   last sentence is short, the hard cut when the length is barely over a multiple
    #   of the step. The corpus had 64 of them: 'ion.', 'es.', 'r.', 'List.'. They are
    #   not wrong, they are unretrievable: an independent chunk that no query can match
    #   and no reader can use, carrying a page number and a section path that make it
    #   look citable. Merging is preferred to dropping because dropping loses text that
    #   is genuinely part of the clause.
    #   The merge is allowed to overshoot the budget by one floor's worth. That is
    #   deliberate: the alternative is re-balancing the hard cut so every slice is the
    #   same size, which spreads the distortion across all of them to avoid it in one.
    #   Rejected: dropping short tails (loses text); lowering the floor to zero (the
    #   crumbs come back).
    min_chars = int(_MIN_CHUNK_TOKENS * chars_per_token)
    merged: list[str] = []
    for part in (p for p in final if p.strip()):
        if (
            merged
            and len(part) < min_chars
            and len(merged[-1]) + len(part) <= max_chars + min_chars
        ):
            merged[-1] = f"{merged[-1]}\n{part}"
        else:
            merged.append(part)
    return merged


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

    lines = page.text.splitlines()
    for line, logical in zip(lines, _logical_lengths(lines), strict=True):
        heading = _match_heading(line, path, logical_length=logical)
        if heading is not None:
            if current.lines:
                blocks.append(current)
            path, title = heading
            current = _Block(
                page_number=page.page_number,
                section_path=list(path),
                section_title=title,
                from_heading=True,
            )
            # Keep the heading line in the block: it carries the section title, which
            # is strong lexical signal for BM25 ("Governing Law", "Limitation of
            # Liability") and is what a reader would expect to see quoted.
            current.lines.append(line.strip())
        else:
            current.lines.append(line)

    if current.lines:
        blocks.append(current)
    return [b for b in _fold_bare_headings(blocks) if b.text], (path, title)


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
