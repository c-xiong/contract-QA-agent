"""Text normalization shared by the parser, the chunker, and span matching.

One module so that "does this clause appear in this document" is answered the same
way everywhere. If the span matcher normalized differently from the chunker, the
measured clause-span match rate would not describe the text that retrieval actually
sees, and the number reported in docs/eval-methodology.md would be meaningless.
"""

from __future__ import annotations

import re
import unicodedata

# PDF extraction yields typographic characters that the CSV annotations spell as
# ASCII. Left as-is they cause span matching to fail on punctuation alone.
_PUNCTUATION_FOLD = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "′": "'",
    "″": '"',
    "–": "-",  # en dash
    "—": "-",  # em dash
    "―": "-",
    "‐": "-",
    "‑": "-",
    "…": "...",
    " ": " ",  # non-breaking space
    " ": " ",
    " ": " ",
    "​": "",  # zero-width space
    "‌": "",
    "‍": "",
    "﻿": "",
    "­": "",  # soft hyphen
}

_TRANSLATION = str.maketrans(_PUNCTUATION_FOLD)
_WHITESPACE = re.compile(r"\s+")
# A hyphen at a line break, splitting one word across two lines.
_LINEBREAK_HYPHEN = re.compile(r"(\w)-\s*\n\s*(\w)")


def fold_punctuation(text: str) -> str:
    """Map typographic punctuation to ASCII without touching structure."""
    return unicodedata.normalize("NFKC", text).translate(_TRANSLATION)


def normalize_for_matching(text: str) -> str:
    """Reduce text to a form suitable for locating a clause span inside a page.

    Collapses all whitespace to single spaces, folds punctuation, repairs words
    broken across a line break, and casefolds. Deliberately lossy: the result is a
    search key, never content to display or cite.
    """
    text = fold_punctuation(text)
    text = _LINEBREAK_HYPHEN.sub(r"\1\2", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip().casefold()


def clean_extracted_text(text: str) -> str:
    """Tidy PDF-extracted text while preserving line structure.

    Line breaks are load-bearing for the chunker: contract numbering is detected on
    line starts, so unlike `normalize_for_matching` this keeps newlines. It only
    folds punctuation, repairs hyphenated line breaks, strips trailing whitespace
    per line, and collapses runs of three or more blank lines.
    """
    text = fold_punctuation(text)
    text = _LINEBREAK_HYPHEN.sub(r"\1\2", text)
    lines = [line.rstrip() for line in text.splitlines()]
    out: list[str] = []
    blanks = 0
    for line in lines:
        if line.strip():
            blanks = 0
            out.append(line)
        else:
            blanks += 1
            if blanks <= 2:
                out.append("")
    return "\n".join(out).strip()


def estimate_tokens(text: str, chars_per_token: float) -> int:
    """Approximate a token count from character length.

    Deliberately a heuristic. Anthropic's count_tokens endpoint is a network round
    trip per chunk, which is too slow for ingestion, and tiktoken counts for the
    wrong tokenizer. Every consumer of Chunk.token_count treats it as approximate;
    nothing in the system makes a correctness decision on this number, only budget
    and sizing decisions. See app/config.py.
    """
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    return max(0, round(len(text) / chars_per_token))
