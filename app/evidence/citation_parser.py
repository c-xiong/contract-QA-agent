"""Deterministic parser for the internal citation format.

SPEC 13.3 requires the parser be deterministic code with unit tests, including tests
for malformed citations. Three forms are accepted:

    [doc-014, p. 12, §8.1]     document, page, section
    [doc-014, p. 12]           document and page; the chunker found no numbering
    [doc-051, §s37]            document and span id, no page (unpaginated source)

The third form exists because ContractNLI ships extracted text rather than paginated
PDFs (SPEC 6.3), so a page number is genuinely unavailable for that slice. Inventing
one by rendering the text to PDF was rejected: fabricated pagination in a system
whose selling point is provenance is the wrong trade.

Parsing is separate from verification on purpose. This module answers "is this
well-formed and what does it refer to". Whether the referent exists is
`citation_verifier`'s job, and conflating the two makes it impossible to tell a
malformed citation from a hallucinated one.
"""

from __future__ import annotations

import re

from app.schemas.evidence import Citation

# Deliberately permissive about internal spacing and the section marker, because the
# model produces these and near-misses should parse and then fail verification with a
# specific error, rather than fail parsing with a generic one.
#
# Accepted section markers: "§", "section", "sec.", or nothing before the number.
_CITATION = re.compile(
    r"""
    \[\s*
    (?P<document_id>doc-\d{3,})
    (?:\s*,\s*p(?:age|\.)?\s*(?P<page>\d+))?
    (?:\s*,\s*(?:§|sec(?:tion)?\.?\s*)?(?P<section>[0-9]+(?:\.[0-9]+){0,3}|\([a-z]{1,2}\)|s[0-9]{1,5}))?
    \s*\]
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Anything that opens like a citation. Used to find malformed spans that the strict
# pattern rejects, so they can be reported as "unparseable" rather than silently
# ignored -- an uncited claim and a claim with a broken citation are different bugs.
_CITATION_SHAPED = re.compile(r"\[[^\[\]]{0,120}\]")


def parse_citation(text: str) -> Citation | None:
    """Parse a single citation string, or return None if it is not well-formed."""
    match = _CITATION.fullmatch(text.strip())
    if match is None:
        return None
    page = match.group("page")
    return Citation(
        document_id=match.group("document_id").lower(),
        page_number=int(page) if page is not None else None,
        section_id=match.group("section"),
        raw=text.strip(),
    )


def normalize_compound_citations(text: str) -> str:
    """Split a well-formed semicolon list into canonical individual citations.

    Models sometimes render two otherwise valid locators in one pair of brackets::

        [doc-031, p. 19, §14.1; doc-035, p. 11, §9.1]

    The public citation grammar deliberately keeps one locator per bracket.  This
    normalizer repairs the mechanical list form without a model call, but only when
    *every* semicolon-delimited component independently parses as a complete citation.
    An ambiguous or partly malformed span is left untouched so the verifier still
    reports it as ``unparseable`` rather than guessing what the model meant.
    """

    def replace(match: re.Match[str]) -> str:
        span = match.group(0)
        if ";" not in span or parse_citation(span) is not None:
            return span

        parts = [part.strip() for part in span[1:-1].split(";")]
        if len(parts) < 2 or any(not part for part in parts):
            return span

        parsed = [parse_citation(f"[{part}]") for part in parts]
        if any(citation is None for citation in parsed):
            return span

        return " ".join(citation.render() for citation in parsed if citation is not None)

    return _CITATION_SHAPED.sub(replace, text)


def extract_citations(text: str) -> tuple[list[Citation], list[str]]:
    """Find every citation in `text`.

    Returns ``(citations, malformed)`` where `malformed` holds bracketed spans that
    look like citation attempts but did not parse. Reporting those separately is what
    lets the verifier distinguish "the model cited nothing" from "the model tried to
    cite and produced garbage".
    """
    text = normalize_compound_citations(text)
    citations: list[Citation] = []
    malformed: list[str] = []

    # DECISION: deduplicate on the canonical rendering, keeping first occurrence.
    #   A writer that cites the same clause after two sentences has made one claim about
    #   one location, not two. Counting it twice inflates "citations emitted", makes the
    #   citation_validity denominator wrong, and renders two identical cards in the
    #   inspector. Observed on the first live Sonnet run.
    #   Position is not lost: the renderer replaces every occurrence of the citation
    #   text, so both mentions still become chips pointing at the one card.
    seen: set[str] = set()

    for match in _CITATION_SHAPED.finditer(text):
        span = match.group(0)
        parsed = parse_citation(span)
        if parsed is not None:
            key = parsed.render()
            if key not in seen:
                seen.add(key)
                citations.append(parsed)
        elif "doc-" in span.lower() or _looks_like_attempt(span):
            malformed.append(span)

    return citations, malformed


def _looks_like_attempt(span: str) -> bool:
    """True for bracketed text that reads like a citation but is not one.

    Kept narrow. A contract quotation containing "[***]" (CUAD redaction marks are
    common) or "[sic]" must not be reported as a broken citation.
    """
    lowered = span.lower()
    if set(span.strip("[]")) <= {"*", " "}:
        return False
    return any(marker in lowered for marker in ("p.", "page", "§", "section"))


def strip_citations(text: str) -> str:
    """Remove well-formed citations, leaving the prose. Used for claim-level checks."""
    return _CITATION.sub("", normalize_compound_citations(text))
