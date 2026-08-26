"""Download and verify the CUAD v1 corpus.

CUAD is 510 real commercial contracts drawn from SEC EDGAR, with 13,101 clause
annotations across 41 categories, labeled under attorney supervision. Licensed
CC BY 4.0. See docs/SPEC.md section 6.2.

Idempotent: a verified archive is not re-downloaded, and an existing extraction
is not re-extracted unless --force is given.

    uv run python scripts/download_cuad.py
    uv run python scripts/download_cuad.py --force
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from app.config import get_settings

ZENODO_RECORD = "https://zenodo.org/records/4595826"
ARCHIVE_URL = "https://zenodo.org/records/4595826/files/CUAD_v1.zip?download=1"
ARCHIVE_MD5 = "c38f490a984420b8a62600db401fafd5"
ARCHIVE_BYTES = 105_883_672

# Members whose absence means the archive is not what we think it is. Checked
# after extraction so a truncated or substituted archive fails here rather than
# three modules downstream.
REQUIRED_MEMBERS = (
    "CUAD_v1/master_clauses.csv",
    "CUAD_v1/CUAD_v1.json",
    "CUAD_v1/full_contract_pdf",
    "CUAD_v1/full_contract_txt",
)

_CHUNK = 1 << 20


class DownloadError(RuntimeError):
    """The archive could not be fetched or did not match its published checksum."""


def _md5(path: Path) -> str:
    # Integrity check against a published checksum, not a security boundary.
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while block := handle.read(_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url}")
    print(f"        to {dest}  ({ARCHIVE_BYTES / 1e6:.0f} MB)")

    request = urllib.request.Request(
        url, headers={"User-Agent": "eval-driven-contract-research-agent (research use)"}
    )
    read = 0
    with urllib.request.urlopen(request, timeout=120) as response:
        total = int(response.headers.get("Content-Length") or ARCHIVE_BYTES)
        with partial.open("wb") as handle:
            while block := response.read(_CHUNK):
                handle.write(block)
                read += len(block)
                pct = 100.0 * read / total if total else 0.0
                print(f"\r  {read / 1e6:7.1f} / {total / 1e6:.0f} MB  ({pct:5.1f}%)", end="")
    print()
    partial.replace(dest)


def fetch_archive(raw_dir: Path, *, force: bool = False) -> Path:
    """Return a path to a checksum-verified CUAD_v1.zip, downloading if needed."""
    archive = raw_dir / "CUAD_v1.zip"

    if archive.exists() and not force:
        print(f"Archive present, verifying checksum: {archive}")
        if _md5(archive) == ARCHIVE_MD5:
            print("  checksum OK, skipping download")
            return archive
        print("  checksum MISMATCH, re-downloading")

    _download(ARCHIVE_URL, archive)

    actual = _md5(archive)
    if actual != ARCHIVE_MD5:
        raise DownloadError(
            f"MD5 mismatch for {archive}\n"
            f"  expected {ARCHIVE_MD5}\n"
            f"  actual   {actual}\n"
            f"Delete the file and retry, or check {ZENODO_RECORD} for a new revision."
        )
    print(f"Checksum OK: md5:{actual}")
    return archive


def extract_archive(archive: Path, raw_dir: Path, *, force: bool = False) -> Path:
    """Extract CUAD_v1.zip and verify the members this project depends on."""
    target = raw_dir / "CUAD_v1"

    if target.exists() and not force:
        print(f"Extraction present, skipping: {target}")
    else:
        if target.exists():
            shutil.rmtree(target)
        print(f"Extracting to {raw_dir}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(raw_dir)

    missing = [m for m in REQUIRED_MEMBERS if not (raw_dir / m).exists()]
    if missing:
        raise DownloadError(
            "Archive extracted but required members are missing:\n"
            + "\n".join(f"  {m}" for m in missing)
        )

    pdf_count = len(list((target / "full_contract_pdf").rglob("*.[pP][dD][fF]")))
    print(f"Verified: {len(REQUIRED_MEMBERS)} required members present, {pdf_count} PDFs")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="re-download and re-extract even if present"
    )
    args = parser.parse_args()

    raw_dir = get_settings().raw_dir
    try:
        archive = fetch_archive(raw_dir, force=args.force)
        target = extract_archive(archive, raw_dir, force=args.force)
    except (DownloadError, zipfile.BadZipFile, OSError) as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 1

    print(f"\nCUAD v1 ready at {target}")
    print("Licensed CC BY 4.0. Attribution is required; see data/README.md.")
    print("Next: uv run python scripts/inspect_cuad_columns.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
