"""Print the master_clauses.csv column list and verify the hardcoded mapping.

SPEC 6.9 item 2 requires printing the columns once and hardcoding an explicit
mapping rather than deriving names by string formatting. This script does both
jobs: it shows the columns, and it fails loudly if CATEGORY_COLUMNS has drifted
from the release on disk.

    uv run python scripts/inspect_cuad_columns.py
    uv run python scripts/inspect_cuad_columns.py --all-pairs
"""

from __future__ import annotations

import argparse
import csv
import sys

from app.config import get_settings
from app.ingestion.cuad import CATEGORY_COLUMNS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="check every one of the 41 categories, not just the 8 in scope",
    )
    args = parser.parse_args()

    csv_path = get_settings().raw_dir / "CUAD_v1" / "master_clauses.csv"
    if not csv_path.exists():
        print(f"Not found: {csv_path}", file=sys.stderr)
        print("Run: uv run python scripts/download_cuad.py", file=sys.stderr)
        return 1

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        columns = next(reader)
        row_count = sum(1 for _ in reader)

    print(f"{csv_path}")
    print(f"{len(columns)} columns, {row_count} data rows\n")
    for i, name in enumerate(columns):
        print(f"{i:3d}  {name!r}")

    # The irregularity check that motivates the hardcoded mapping. Columns after
    # 'Filename' come in (clause, answer) pairs; the answer column is *usually*
    # f"{clause}-Answer".
    print("\n" + "=" * 70)
    print("Pairs where a derived f'{category}-Answer' would not match the release:")
    body = columns[1:]
    irregular = 0
    for i in range(0, len(body) - 1, 2):
        clause, answer = body[i], body[i + 1]
        if answer != f"{clause}-Answer":
            irregular += 1
            print(f"  {clause!r}")
            print(f"    actual: {answer!r}")
            print(f"    derived: {clause + '-Answer'!r}")
    print(f"\n  {irregular} irregular of {len(body) // 2} pairs")
    if not args.all_pairs:
        print("  (all 41 pairs checked; --all-pairs only changes the mapping check below)")

    # Guard: the mapping this project actually uses must exist in this release.
    print("\n" + "=" * 70)
    print(f"Verifying CATEGORY_COLUMNS ({len(CATEGORY_COLUMNS)} categories in scope):")
    present = set(columns)
    broken = []
    for category, (clause_col, answer_col) in sorted(CATEGORY_COLUMNS.items()):
        missing = [c for c in (clause_col, answer_col) if c not in present]
        status = "OK " if not missing else "FAIL"
        print(f"  [{status}] {category}")
        if missing:
            broken.append((category, missing))

    if broken:
        print("\nCATEGORY_COLUMNS does not match this release:", file=sys.stderr)
        for category, missing in broken:
            for column in missing:
                print(f"  {category}: no such column {column!r}", file=sys.stderr)
        print("\nUpdate app/ingestion/cuad.py before ingesting.", file=sys.stderr)
        return 1

    print("\nAll mapped columns present.")
    print("Next: uv run python scripts/sample_contracts.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
