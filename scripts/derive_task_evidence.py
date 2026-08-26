"""Transcribe CUAD expert annotations into eval-task evidence fields.

This script does NOT author eval tasks. It transcribes expert annotation into the
schema's shape: expected_document_ids, and expected_evidence with the verbatim span
and its resolved page number. .claude/rules/evals.md permits exactly this and calls
it what it is -- transcription, not authorship.

What it deliberately leaves empty, because they are author-written:
  question, required_points, forbidden_claims, partial_coverage_note, category

    uv run python scripts/derive_task_evidence.py --doc doc-001 --category "Governing Law"
    uv run python scripts/derive_task_evidence.py --doc doc-001 --list
"""

from __future__ import annotations

import argparse
import json
import sys

from app.config import get_settings
from app.ingestion.cuad import (
    CATEGORY_COLUMNS,
    CuadFormatError,
    index_contract_pdfs,
    load_master_clauses,
)
from app.ingestion.manifest import CorpusManifest, ManifestError, manifest_path
from app.ingestion.pdf_parser import parse_pdf
from app.ingestion.text import normalize_for_matching


def resolve_page(span: str, pages: list[str]) -> int | None:
    """Locate a clause span's page by normalized containment."""
    needle = normalize_for_matching(span)
    if not needle:
        return None
    for number, page in enumerate(pages, start=1):
        if needle in page:
            return number
    # Fall back to the longest leading fragment that still matches uniquely enough.
    for probe_len in (200, 120, 60):
        probe = needle[:probe_len]
        for number, page in enumerate(pages, start=1):
            if probe and probe in page:
                return number
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", required=True, help="document_id, e.g. doc-001")
    parser.add_argument("--category", help=f"one of: {', '.join(sorted(CATEGORY_COLUMNS))}")
    parser.add_argument("--list", action="store_true", help="list annotated categories and exit")
    args = parser.parse_args()

    settings = get_settings()
    cuad_root = settings.raw_dir / "CUAD_v1"

    try:
        manifest = CorpusManifest.load(manifest_path(settings.data_dir))
        entry = manifest.get(args.doc)
        pdf_index = index_contract_pdfs(cuad_root / "full_contract_pdf")
        rows = {
            r.filename: r for r in load_master_clauses(cuad_root / "master_clauses.csv", pdf_index)
        }
    except (ManifestError, CuadFormatError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    row = rows.get(entry.source_filename)
    if row is None:
        print(f"FAILED: no CSV row for {entry.source_filename!r}", file=sys.stderr)
        return 1

    if args.list or not args.category:
        print(f"{args.doc}  ({entry.agreement_type})")
        print(f"  source: {entry.source_filename}\n")
        print("  annotated categories:")
        for category in sorted(CATEGORY_COLUMNS):
            spans = row.clause_spans.get(category, [])
            answer = row.answers.get(category, "")
            mark = "x" if spans else " "
            print(f"    [{mark}] {category:36s} {len(spans)} span(s)  answer={answer[:40]!r}")
        print('\nRe-run with --category "<name>" to transcribe one.')
        return 0

    if args.category not in CATEGORY_COLUMNS:
        print(f"FAILED: unknown category {args.category!r}", file=sys.stderr)
        return 1

    spans = row.clause_spans.get(args.category, [])
    if not spans:
        print(
            f"{args.doc} has no {args.category!r} annotation.\n"
            "An empty cell is a WEAK negative: CUAD annotators can miss a clause, so "
            "absence here is not proof of absence in the document. SPEC 6.2 requires a "
            "hand check before promoting this to an unanswerable task.",
        )
        return 0

    parsed = parse_pdf(settings.raw_dir / entry.relative_path)
    pages = [normalize_for_matching(p.text) for p in parsed.pages]

    evidence = []
    for span in spans:
        page = resolve_page(span, pages)
        evidence.append(
            {
                "document_id": args.doc,
                "section_id": None,
                "page_number": page,
                "span_text": span,
                "source": "cuad",
            }
        )
        if page is None:
            print(
                "WARNING: could not resolve a page for one span; leaving page_number null.",
                file=sys.stderr,
            )

    print(f"# Transcribed from CUAD master_clauses.csv, category {args.category!r}")
    print(f"# Normalized answer (CUAD's own): {row.answers.get(args.category, '')!r}")
    print("#")
    print("# Paste into a task object. The fields below are expert annotation.")
    print("# You must still write: task_id, category, question, required_points,")
    print("# forbidden_claims, expected_behavior, dataset_version.")
    print()
    print(
        json.dumps(
            {
                "expected_document_ids": [args.doc],
                "allowed_document_ids": [args.doc],
                "expected_evidence": evidence,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
