"""Write the frozen hand-v1 judge calibration/audit split. Refuses to overwrite a different one."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.judge_split import SEED, build_manifest
from evals.loader import load_suite, suite_path

OUTPUT = Path("evals/datasets/hand/judge_split-v1.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true", help="verify the checked-in split")
    args = parser.parse_args()
    manifest = build_manifest(load_suite("hand"), suite_path("hand"), args.seed)
    text = json.dumps(manifest, indent=2) + "\n"
    if args.output.exists():
        if args.output.read_text() != text:
            raise SystemExit(f"{args.output} differs from a fresh build; the split is frozen.")
        print(f"{args.output} is up to date.")
        return
    if args.check:
        raise SystemExit(f"{args.output} is missing.")
    args.output.write_text(text)
    print(f"Wrote {args.output}: {manifest['balance']}")


if __name__ == "__main__":
    main()
