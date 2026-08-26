"""Download and verify the ContractNLI corpus.

607 real non-disclosure agreements, each annotated against the same 17 hypotheses as
Entailment, Contradiction, or NotMentioned, with evidence spans for the first two.
Koreeda and Manning, Findings of EMNLP 2021. CC BY 4.0.

This corpus solves the two grading problems CUAD cannot (SPEC 6.3):

- `NotMentioned` is expert-labelled abstention ground truth. The hardest eval category
  to construct by hand comes free, with more authority than the author could supply.
- `Contradiction` is a natural claim-support test set: the evidence exists, is
  retrievable, and is topically on point, but does not support the proposed claim. That
  is exactly the case a naive citation checker passes and a real one must fail.

    uv run python scripts/download_contractnli.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from app.config import get_settings

ARCHIVE_URL = "https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip"
LANDING_PAGE = "https://stanfordnlp.github.io/contract-nli/"

REQUIRED_MEMBERS = (
    "contract-nli/train.json",
    "contract-nli/dev.json",
    "contract-nli/test.json",
)

_CHUNK = 1 << 20


class DownloadError(RuntimeError):
    """The archive could not be fetched or is not what we expect."""


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url}")
    request = urllib.request.Request(
        url, headers={"User-Agent": "eval-driven-contract-research-agent (research use)"}
    )
    read = 0
    with urllib.request.urlopen(request, timeout=180) as response:
        total = int(response.headers.get("Content-Length") or 0)
        with partial.open("wb") as handle:
            while block := response.read(_CHUNK):
                handle.write(block)
                read += len(block)
                pct = f"({100.0 * read / total:5.1f}%)" if total else ""
                print(f"\r  {read / 1e6:7.1f} MB {pct}", end="")
    print()
    partial.replace(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    raw_dir = get_settings().raw_dir
    archive = raw_dir / "contract-nli.zip"
    target = raw_dir / "contract-nli"

    try:
        if not archive.exists() or args.force:
            _download(ARCHIVE_URL, archive)
        else:
            print(f"Archive present: {archive}")

        if target.exists() and not args.force:
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
                "Extracted but required members are missing:\n"
                + "\n".join(f"  {m}" for m in missing)
                + f"\nCheck {LANDING_PAGE} for a layout change."
            )
    except (DownloadError, zipfile.BadZipFile, OSError) as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 1

    print(f"\nContractNLI ready at {target}")
    print("Licensed CC BY 4.0. Attribution required; see data/README.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
