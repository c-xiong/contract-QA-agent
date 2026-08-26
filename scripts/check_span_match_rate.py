"""Measure how often a CUAD clause span can be located in our own parsed PDF text.

SPEC 6.9 item 5. CUAD gives the verbatim clause text but no page number, so every
page citation this system produces depends on finding that text in our extraction.
The match rate is therefore a parser-quality metric, and SPEC 6.2 requires it be
logged and reported in docs/eval-methodology.md rather than assumed.

Matching is layered, and each layer is counted separately because they are not
equally trustworthy:

  exact      the normalized span occurs verbatim in one page
  joined     it occurs only after pages are concatenated -- the clause straddles a
             page break, so a page number is ambiguous and we report the first page
  fuzzy      no substring match; best difflib ratio over a sliding window clears
             --fuzzy-threshold
  miss       none of the above

    uv run python scripts/check_span_match_rate.py
    uv run python scripts/check_span_match_rate.py --limit 5 --verbose
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher

from app.config import get_settings
from app.ingestion.cuad import (
    CATEGORY_COLUMNS,
    CuadFormatError,
    index_contract_pdfs,
    load_master_clauses,
)
from app.ingestion.manifest import CorpusManifest, ManifestError, manifest_path
from app.ingestion.pdf_parser import PdfParseError, parse_pdf
from app.ingestion.text import normalize_for_matching

# SPEC 6.9: "If it is below roughly 85 percent, fix the parser or the normalization
# before proceeding; every downstream page citation depends on it."
TARGET_RATE = 0.85

DEFAULT_FUZZY_THRESHOLD = 0.90
# Spans shorter than this are skipped: a 20-character span matches by accident often
# enough to inflate the rate without evidencing anything about the parser.
MIN_SPAN_CHARS = 40


@dataclass
class SpanResult:
    document_id: str
    category: str
    outcome: str
    page_number: int | None
    ratio: float
    span_preview: str


def locate_exact(needle: str, pages: list[str]) -> int | None:
    """Return the 1-based page containing `needle`, or None."""
    for index, page in enumerate(pages, start=1):
        if needle in page:
            return index
    return None


def locate_fuzzy(needle: str, pages: list[str]) -> tuple[int | None, float]:
    """Best fuzzy match of `needle` across pages, as (page_number, ratio).

    Compares the span against same-length windows of each page, stepping by a
    quarter of the span length. A full pairwise ratio over an entire page would be
    dominated by the page's non-clause text and would score near zero even on a
    perfect match.
    """
    best_page: int | None = None
    best_ratio = 0.0
    window = len(needle)
    step = max(1, window // 4)

    for index, page in enumerate(pages, start=1):
        if not page:
            continue
        limit = max(1, len(page) - window + 1)
        for start in range(0, limit, step):
            candidate = page[start : start + window]
            ratio = SequenceMatcher(None, needle, candidate, autojunk=False).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_page = index
    return best_page, best_ratio


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="documents to check (0 = all)")
    parser.add_argument("--fuzzy-threshold", type=float, default=DEFAULT_FUZZY_THRESHOLD)
    parser.add_argument("--min-span-chars", type=int, default=MIN_SPAN_CHARS)
    parser.add_argument("--verbose", action="store_true", help="print every miss")
    args = parser.parse_args()

    settings = get_settings()
    cuad_root = settings.raw_dir / "CUAD_v1"

    try:
        manifest = CorpusManifest.load(manifest_path(settings.data_dir))
        pdf_index = index_contract_pdfs(cuad_root / "full_contract_pdf")
        rows = {
            row.filename: row
            for row in load_master_clauses(cuad_root / "master_clauses.csv", pdf_index)
        }
    except (ManifestError, CuadFormatError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    entries = [e for e in manifest.active.values() if e.corpus_source == "cuad"]
    entries.sort(key=lambda e: e.document_id)
    if args.limit:
        entries = entries[: args.limit]

    print(f"Checking {len(entries)} documents, {len(CATEGORY_COLUMNS)} categories each")
    print(f"fuzzy threshold {args.fuzzy_threshold}, minimum span {args.min_span_chars} chars\n")

    results: list[SpanResult] = []
    scanned: list[str] = []
    parse_failures: list[str] = []

    for entry in entries:
        row = rows.get(entry.source_filename)
        if row is None:
            print(f"  {entry.document_id}: no CSV row for {entry.source_filename!r}")
            continue

        pdf_path = settings.raw_dir / entry.relative_path
        try:
            parsed = parse_pdf(pdf_path)
        except PdfParseError as exc:
            parse_failures.append(f"{entry.document_id}: {exc}")
            continue

        if parsed.looks_scanned:
            scanned.append(
                f"{entry.document_id}: {parsed.chars_per_page:.0f} chars/page "
                f"over {parsed.page_count} pages -- no text layer"
            )
            continue

        pages = [normalize_for_matching(p.text) for p in parsed.pages]
        joined = " ".join(pages)

        for category, spans in row.clause_spans.items():
            for span in spans:
                needle = normalize_for_matching(span)
                if len(needle) < args.min_span_chars:
                    continue

                preview = needle[:60]
                page = locate_exact(needle, pages)
                if page is not None:
                    results.append(
                        SpanResult(entry.document_id, category, "exact", page, 1.0, preview)
                    )
                    continue

                if needle in joined:
                    results.append(
                        SpanResult(entry.document_id, category, "joined", None, 1.0, preview)
                    )
                    continue

                page, ratio = locate_fuzzy(needle, pages)
                outcome = "fuzzy" if ratio >= args.fuzzy_threshold else "miss"
                results.append(
                    SpanResult(entry.document_id, category, outcome, page, ratio, preview)
                )

        located = sum(1 for r in results if r.document_id == entry.document_id)
        print(f"  {entry.document_id}  {parsed.page_count:3d}p  {located:3d} spans checked")

    if not results:
        print("\nNo spans checked. Nothing to report.", file=sys.stderr)
        return 1

    outcomes = Counter(r.outcome for r in results)
    total = len(results)
    matched = outcomes["exact"] + outcomes["joined"] + outcomes["fuzzy"]
    rate = matched / total
    page_resolvable = outcomes["exact"] + outcomes["fuzzy"]

    print("\n" + "=" * 72)
    print(f"CLAUSE-SPAN MATCH RATE   {matched}/{total} = {rate:.1%}")
    print("=" * 72)
    for name in ("exact", "joined", "fuzzy", "miss"):
        count = outcomes[name]
        print(f"  {name:8s} {count:5d}  {count / total:6.1%}")
    print(
        f"\n  page-resolvable (exact+fuzzy): {page_resolvable}/{total} = {page_resolvable / total:.1%}"
    )
    print(f"  page-ambiguous  (joined)     : {outcomes['joined']}/{total}")

    print("\nPer category:")
    by_cat: dict[str, Counter[str]] = {}
    for result in results:
        by_cat.setdefault(result.category, Counter())[result.outcome] += 1
    for category in sorted(by_cat):
        counts = by_cat[category]
        n = sum(counts.values())
        hit = counts["exact"] + counts["joined"] + counts["fuzzy"]
        print(f"  {category:36s} {hit:4d}/{n:<4d} {hit / n:6.1%}")

    if scanned:
        print(f"\nScanned documents excluded ({len(scanned)}):")
        for line in scanned:
            print(f"  {line}")
    if parse_failures:
        print(f"\nParse failures ({len(parse_failures)}):")
        for line in parse_failures:
            print(f"  {line}")

    if args.verbose:
        misses = [r for r in results if r.outcome == "miss"]
        print(f"\nMisses ({len(misses)}):")
        for result in misses[:40]:
            print(f"  {result.document_id} {result.category:30s} r={result.ratio:.2f}")
            print(f"      {result.span_preview}...")

    print()
    if rate < TARGET_RATE:
        print(
            f"BELOW TARGET: {rate:.1%} < {TARGET_RATE:.0%}. SPEC 6.9 says fix the parser or "
            "the normalization before proceeding; page citations depend on this.",
            file=sys.stderr,
        )
        return 2
    print(
        f"At or above the {TARGET_RATE:.0%} target. Record this number in docs/eval-methodology.md."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
