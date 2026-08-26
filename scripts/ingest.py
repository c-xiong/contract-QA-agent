"""Ingest the corpus: parse PDFs, chunk them, write the store.

uv run python scripts/ingest.py --limit 5
uv run python scripts/ingest.py
"""

from __future__ import annotations

import argparse
import sys

from app.config import get_settings
from app.ingestion.manifest import CorpusManifest, ManifestError, manifest_path
from app.ingestion.pipeline import ingest
from app.ingestion.store import StoreError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="ingest only the first N documents")
    args = parser.parse_args()

    settings = get_settings()
    try:
        manifest = CorpusManifest.load(manifest_path(settings.data_dir))
    except ManifestError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"Manifest: {len(manifest.active)} active documents")
    if args.limit:
        print(f"Limit: {args.limit}")
    print()

    try:
        store, report = ingest(manifest, settings, limit=args.limit)
    except StoreError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print(report.summary())

    if not report.ingested:
        print("\nNothing ingested.", file=sys.stderr)
        return 1

    store.save(settings.processed_dir)
    print(f"\nStore written: {settings.processed_dir}")
    print('Next: uv run python scripts/demo.py "<question>"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
