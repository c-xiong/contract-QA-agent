"""Select the CUAD corpus subset and register it in the manifest.

SPEC 6.9 item 3: sample contracts spread across 4 to 5 agreement types, requiring
annotation coverage in at least 4 of the chosen categories, and save the sampling
script rather than only its result. This is that script.

Selection is deterministic given (--seed, --size, --min-coverage, --types) and the
release on disk. Re-running with a larger --size keeps every ID already assigned
and appends new ones, because the manifest is append-only. See docs/decisions.md.

    uv run python scripts/sample_contracts.py                  # 40 contracts
    uv run python scripts/sample_contracts.py --size 5         # Sprint 0 slice
    uv run python scripts/sample_contracts.py --dry-run
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

from app.config import get_settings
from app.ingestion.cuad import (
    CATEGORY_COLUMNS,
    CuadFormatError,
    CuadRow,
    index_contract_pdfs,
    load_master_clauses,
)
from app.ingestion.manifest import CorpusManifest, ManifestError, manifest_path

DEFAULT_SIZE = 40
DEFAULT_MIN_COVERAGE = 4
DEFAULT_TYPE_COUNT = 5


def agreement_type_of(pdf_path: Path) -> str:
    """CUAD encodes agreement type as the directory holding the PDF.

    Layout is ``full_contract_pdf/Part_{I,II,III}/<AgreementType>/<file>.pdf``.
    """
    return pdf_path.parent.name


def select(
    rows: list[CuadRow],
    *,
    size: int,
    min_coverage: int,
    type_count: int,
    seed: int,
) -> tuple[list[CuadRow], dict[str, str]]:
    """Pick `size` contracts spread across `type_count` agreement types.

    Eligibility: the PDF resolved, and the contract carries verbatim clause spans in
    at least `min_coverage` of the categories in scope. Coverage is the point -- a
    contract annotated in one category yields one eval task and cannot support a
    cross-document comparison, so it is worth less than its page count costs.

    Spread is enforced by round-robin over the chosen types rather than by sampling
    the pool at random. Random sampling over a pool whose types are unbalanced (CUAD
    ranges from 34 Maintenance contracts to 1 Affiliate Agreement) reproduces that
    imbalance, and a corpus that is 60 percent one agreement type makes the
    per-category retrieval breakdown in Experiment A hard to read.
    """
    eligible = [r for r in rows if r.pdf_path is not None and r.coverage >= min_coverage]

    by_type: dict[str, list[CuadRow]] = defaultdict(list)
    for row in eligible:
        assert row.pdf_path is not None
        by_type[agreement_type_of(row.pdf_path)].append(row)

    # Prefer types with enough eligible contracts to contribute evenly. Ties broken
    # by name so the ordering does not depend on dict insertion order.
    ranked = sorted(by_type.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    chosen_types = [name for name, _ in ranked[:type_count]]

    rng = random.Random(seed)
    pools: dict[str, list[CuadRow]] = {}
    for name in chosen_types:
        pool = sorted(by_type[name], key=lambda r: (-r.coverage, r.filename))
        # Shuffle within a coverage tier so the seed matters, without discarding the
        # preference for better-annotated contracts.
        tiers: dict[int, list[CuadRow]] = defaultdict(list)
        for row in pool:
            tiers[row.coverage].append(row)
        ordered: list[CuadRow] = []
        for coverage in sorted(tiers, reverse=True):
            tier = tiers[coverage]
            rng.shuffle(tier)
            ordered.extend(tier)
        pools[name] = ordered

    selected: list[CuadRow] = []
    exhausted: set[str] = set()
    while len(selected) < size and len(exhausted) < len(chosen_types):
        for name in chosen_types:
            if len(selected) >= size:
                break
            if pools[name]:
                selected.append(pools[name].pop(0))
            else:
                exhausted.add(name)

    stats = {
        "total_rows": str(len(rows)),
        "eligible": str(len(eligible)),
        "types_available": str(len(by_type)),
        "types_chosen": ", ".join(chosen_types),
    }
    return selected, stats


def criterion_text(*, size: int, min_coverage: int, type_count: int, seed: int) -> str:
    categories = ", ".join(sorted(CATEGORY_COLUMNS))
    return (
        f"CUAD v1. Target {size} contracts, round-robin across the {type_count} agreement "
        f"types with the most eligible contracts, seed {seed}. A contract is eligible if "
        f"its PDF resolves from master_clauses.csv and it carries verbatim clause spans in "
        f"at least {min_coverage} of the {len(CATEGORY_COLUMNS)} categories in scope "
        f"({categories}). Within a type, contracts are ordered by descending category "
        f"coverage and shuffled within each coverage tier."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--min-coverage", type=int, default=DEFAULT_MIN_COVERAGE)
    parser.add_argument("--types", type=int, default=DEFAULT_TYPE_COUNT)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--dry-run", action="store_true", help="print the selection, write nothing")
    args = parser.parse_args()

    settings = get_settings()
    cuad_root = settings.raw_dir / "CUAD_v1"
    csv_path = cuad_root / "master_clauses.csv"
    pdf_root = cuad_root / "full_contract_pdf"

    if not csv_path.exists():
        print(f"Not found: {csv_path}", file=sys.stderr)
        print("Run: uv run python scripts/download_cuad.py", file=sys.stderr)
        return 1

    try:
        pdf_index = index_contract_pdfs(pdf_root)
        rows = list(load_master_clauses(csv_path, pdf_index))
    except CuadFormatError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    unresolved = [r.filename for r in rows if r.pdf_path is None]
    print(f"master_clauses.csv rows : {len(rows)}")
    print(f"PDFs indexed            : {len(pdf_index)}")
    print(f"filename resolution     : {len(rows) - len(unresolved)}/{len(rows)}")
    if unresolved:
        print("  UNRESOLVED (excluded from sampling):")
        for name in unresolved:
            print(f"    {name}")

    selected, stats = select(
        rows,
        size=args.size,
        min_coverage=args.min_coverage,
        type_count=args.types,
        seed=args.seed,
    )
    print(f"eligible (coverage>={args.min_coverage})   : {stats['eligible']}")
    print(f"agreement types available: {stats['types_available']}")
    print(f"agreement types chosen  : {stats['types_chosen']}")
    print(f"selected                : {len(selected)}")

    if len(selected) < args.size:
        print(
            f"\nWARNING: asked for {args.size}, only {len(selected)} met the criteria. "
            "Lower --min-coverage or raise --types.",
            file=sys.stderr,
        )

    path = manifest_path(settings.data_dir)
    try:
        manifest = CorpusManifest.load_or_empty(path)
    except ManifestError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    manifest.sampling_criterion = criterion_text(
        size=args.size, min_coverage=args.min_coverage, type_count=args.types, seed=args.seed
    )

    created = 0
    print()
    for row in selected:
        assert row.pdf_path is not None
        entry, is_new = manifest.add(
            source_filename=row.filename,
            relative_path=str(row.pdf_path.relative_to(settings.raw_dir)),
            corpus_source="cuad",
            agreement_type=agreement_type_of(row.pdf_path),
            annotated_categories=sorted(row.annotated_categories),
        )
        created += is_new
        marker = "NEW " if is_new else "kept"
        print(f"  {marker} {entry.document_id}  cov={row.coverage}  {row.filename[:64]}")

    print(f"\n{created} new, {len(selected) - created} already registered")

    if args.dry_run:
        print("\n--dry-run: manifest not written")
        return 0

    try:
        manifest.save(path)
    except ManifestError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"Manifest written: {path}  ({len(manifest.documents)} documents)")
    print("Next: uv run python scripts/ingest.py --limit 5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
