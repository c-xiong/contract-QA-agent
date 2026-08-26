"""Build the synthetic overlay and register it in the corpus manifest.

SPEC 6.6: exactly three uses, 5 to 8 documents total. Every output is derived from a
real CUAD contract already in the corpus, so ground truth is known by construction.

    uv run python scripts/build_synthetic.py
"""

from __future__ import annotations

import argparse
import sys

from app.config import get_settings
from app.ingestion.manifest import CorpusManifest, ManifestError, manifest_path
from app.ingestion.pdf_parser import PdfParseError, parse_pdf
from app.ingestion.synthetic import (
    make_conflicting_version,
    make_near_duplicate,
    make_prompt_injection,
)

# Which real documents each transformation derives from. Fixed rather than sampled: a
# synthetic overlay that changes when the seed changes cannot be reasoned about, and the
# eval tasks built on it name specific documents.
PLAN = [
    ("doc-001", "conflicting_versions"),
    ("doc-002", "conflicting_versions"),
    ("doc-001", "prompt_injection"),
    ("doc-003", "prompt_injection"),
    ("doc-001", "near_duplicate"),
    ("doc-004", "near_duplicate"),
]

BUILDERS = {
    "conflicting_versions": make_conflicting_version,
    "prompt_injection": make_prompt_injection,
    "near_duplicate": make_near_duplicate,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    out_dir = settings.raw_dir / "synthetic"

    try:
        manifest = CorpusManifest.load(manifest_path(settings.data_dir))
    except ManifestError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"Building {len(PLAN)} synthetic documents\n")
    created = 0

    for index, (source_id, transformation) in enumerate(PLAN, start=1):
        try:
            source = manifest.get(source_id)
        except ManifestError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1

        if source.corpus_source != "cuad":
            print(f"  skip {source_id}: only CUAD sources are supported", file=sys.stderr)
            continue

        try:
            parsed = parse_pdf(settings.raw_dir / source.relative_path)
        except PdfParseError as exc:
            print(f"  FAILED {source_id}: {exc}", file=sys.stderr)
            return 1

        synthetic_id = f"synthetic-{index:03d}-{transformation}"
        document = BUILDERS[transformation](
            parsed,
            synthetic_id=synthetic_id,
            source_document_id=source_id,
            source_filename=source.source_filename,
        )

        if not args.dry_run:
            document.save(out_dir)

        entry, is_new = manifest.add(
            source_filename=f"synthetic:{synthetic_id}",
            relative_path=f"synthetic/{synthetic_id}.json",
            corpus_source="synthetic",
            agreement_type=source.agreement_type,
            is_synthetic=True,
        )
        created += is_new
        marker = "NEW " if is_new else "kept"
        print(
            f"  {marker} {entry.document_id}  {transformation:22s} from {source_id}  "
            f"{len(document.pages):3d} pages, {len(document.edits)} edit(s)"
        )
        for edit in document.edits[:2]:
            before = (edit.before or "(inserted)")[:52].replace("\n", " ")
            after = edit.after[:52].replace("\n", " ")
            print(f"        p.{edit.page_number} {edit.kind}: {before!r} -> {after!r}")

    print(f"\n{created} new")
    if args.dry_run:
        print("--dry-run: nothing written")
        return 0

    try:
        manifest.save(manifest_path(settings.data_dir))
    except ManifestError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"Synthetic documents: {out_dir}")
    print(f"Manifest: {len(manifest.documents)} documents total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
