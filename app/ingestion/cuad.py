"""Adapter for the CUAD v1 release: column mapping, filename resolution, annotation parsing.

CUAD ships two artifacts this project needs, and neither is as regular as it looks:

- ``master_clauses.csv``: 510 rows x 83 columns. Column 0 is the filename; the rest are
  (clause span, normalized answer) pairs, one pair per annotation category.
- ``full_contract_pdf/``: 510 PDFs whose names do not all match the CSV's ``Filename``.

Every irregularity handled here was observed in the actual release, not anticipated. See
docs/decisions.md for the measurements.
"""

from __future__ import annotations

import ast
import csv
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# --- Category to column mapping ------------------------------------------------
#
# SPEC 6.2 warns that column naming in master_clauses.csv "is not perfectly uniform"
# and that deriving names by string formatting "will silently drop categories."
# That is exactly right, and there is precisely one offender in the release:
#
#     'Notice Period To Terminate Renewal- Answer'    <- space after the hyphen
#     'Governing Law-Answer'                          <- every other one of the 41
#
# f"{category}-Answer" produces a KeyError or, with .get(), a silent empty column for
# the one numeric, phrasing-diverse category the SPEC specifically wants for
# query-refinement tasks. Hence: explicit pairs, verified by a test, never derived.
#
# The eight categories are the SPEC 6.2 recommended set. The mix is deliberate --
# short and exact (Governing Law), long and paraphrase-heavy (Non-Compete), numeric
# (Notice Period, Insurance) -- because Experiment A is only informative if the
# categories differ in retrieval character.

CATEGORY_COLUMNS: dict[str, tuple[str, str]] = {
    "Governing Law": ("Governing Law", "Governing Law-Answer"),
    "Cap On Liability": ("Cap On Liability", "Cap On Liability-Answer"),
    "Uncapped Liability": ("Uncapped Liability", "Uncapped Liability-Answer"),
    "Termination For Convenience": (
        "Termination For Convenience",
        "Termination For Convenience-Answer",
    ),
    "Notice Period To Terminate Renewal": (
        "Notice Period To Terminate Renewal",
        "Notice Period To Terminate Renewal- Answer",  # irregular; see above
    ),
    "Non-Compete": ("Non-Compete", "Non-Compete-Answer"),
    "Change Of Control": ("Change Of Control", "Change Of Control-Answer"),
    "Insurance": ("Insurance", "Insurance-Answer"),
}

# --- Filename resolution -------------------------------------------------------
#
# 509 of 510 CSV filenames resolve to a PDF under one deterministic rule: NFC
# normalize, strip a stray trailing quote, trim whitespace around the extension,
# map filesystem-hostile characters to underscore, casefold. The characters below
# were replaced with '_' when the PDFs were packaged but left intact in the CSV.
_UNSAFE_IN_FILENAMES = "&'"

# The remaining one is a genuine rename: the packaged PDF carries an extra
# "_Option Agreement" suffix. It is listed explicitly rather than recovered by fuzzy
# matching. Fuzzy matching a filename means silently guessing which contract an
# expert annotation belongs to, and a wrong guess here mislabels ground truth in a
# way no downstream test can detect.
_FILENAME_OVERRIDES: dict[str, str] = {
    "HarpoonTherapeuticsInc_20200312_10-K_EX-10.18_12051356_EX-10.18_Development Agreement.PDF": (
        "HarpoonTherapeuticsInc_20200312_10-K_EX-10.18_12051356_EX-10.18"
        "_Development Agreement_Option Agreement.pdf"
    ),
}


class CuadFormatError(RuntimeError):
    """The CUAD release on disk does not have the structure this adapter expects."""


def normalize_filename(name: str) -> str:
    """Reduce a CUAD filename to a key that matches across the CSV and the PDF tree."""
    name = unicodedata.normalize("NFC", name).strip().rstrip("'\"").strip()
    stem, _, ext = name.rpartition(".")
    if stem:
        name = f"{stem.strip()}.{ext}"
    for char in _UNSAFE_IN_FILENAMES:
        name = name.replace(char, "_")
    return name.lower()


def index_contract_pdfs(pdf_root: Path) -> dict[str, Path]:
    """Map normalized filename to PDF path, raising if two files collide on one key."""
    index: dict[str, Path] = {}
    for path in sorted(pdf_root.rglob("*")):
        if path.suffix.lower() != ".pdf":
            continue
        key = normalize_filename(path.name)
        if key in index:
            raise CuadFormatError(
                f"Two PDFs normalize to the same key {key!r}: {index[key]} and {path}. "
                "Normalization is too aggressive; tighten it before ingesting."
            )
        index[key] = path
    if not index:
        raise CuadFormatError(f"No PDFs found under {pdf_root}")
    return index


def resolve_pdf(csv_filename: str, pdf_index: dict[str, Path]) -> Path | None:
    """Return the PDF for a CSV row's Filename, or None if it cannot be resolved."""
    override = _FILENAME_OVERRIDES.get(csv_filename)
    if override is not None:
        return pdf_index.get(normalize_filename(override))
    return pdf_index.get(normalize_filename(csv_filename))


# --- Annotation parsing --------------------------------------------------------


def parse_clause_spans(cell: str) -> list[str]:
    """Parse a clause cell into its verbatim spans.

    Clause cells are Python list literals stored as text -- ``"['span one', 'span two']"``
    -- because a category can be annotated in several places in one contract. An empty
    cell, ``[]``, or a cell that does not parse means no annotation for this
    (contract, category) pair.

    A malformed cell returns an empty list rather than raising: it is indistinguishable
    from an absent annotation for retrieval purposes, and the caller counts absences
    anyway. It must never be promoted to an unanswerable eval task without the hand
    check SPEC 6.2 requires.

    Measured on CUAD v1: all 4080 clause cells across the 8 categories in scope are
    list literals -- none bare, none unparseable. The fallbacks below are therefore
    defensive against a future release, not load-bearing today.
    """
    text = cell.strip()
    if not text:
        return []
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


@dataclass(frozen=True, slots=True)
class CuadRow:
    """One row of master_clauses.csv, restricted to the categories in scope."""

    filename: str
    pdf_path: Path | None
    clause_spans: dict[str, list[str]]
    answers: dict[str, str]

    @property
    def annotated_categories(self) -> frozenset[str]:
        """Categories with at least one verbatim clause span in this contract."""
        return frozenset(cat for cat, spans in self.clause_spans.items() if spans)

    @property
    def coverage(self) -> int:
        return len(self.annotated_categories)


def load_master_clauses(
    csv_path: Path,
    pdf_index: dict[str, Path],
    categories: dict[str, tuple[str, str]] | None = None,
) -> Iterator[CuadRow]:
    """Stream master_clauses.csv as CuadRow objects, restricted to `categories`."""
    categories = categories or CATEGORY_COLUMNS

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise CuadFormatError(f"{csv_path} has no header row")

        present = set(reader.fieldnames)
        required = {col for pair in categories.values() for col in pair} | {"Filename"}
        if missing := sorted(required - present):
            raise CuadFormatError(
                "master_clauses.csv is missing expected columns:\n"
                + "\n".join(f"  {c!r}" for c in missing)
                + "\nThe release layout changed. Re-run scripts/inspect_cuad_columns.py "
                "and update CATEGORY_COLUMNS."
            )

        for row in reader:
            filename = (row.get("Filename") or "").strip()
            if not filename:
                continue
            yield CuadRow(
                filename=filename,
                pdf_path=resolve_pdf(filename, pdf_index),
                clause_spans={
                    cat: parse_clause_spans(row.get(clause_col) or "")
                    for cat, (clause_col, _) in categories.items()
                },
                answers={
                    cat: (row.get(answer_col) or "").strip()
                    for cat, (_, answer_col) in categories.items()
                },
            )
