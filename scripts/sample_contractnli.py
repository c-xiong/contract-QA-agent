"""Select ContractNLI NDAs and register them in the corpus manifest.

SPEC 6.7 targets ~10 NDAs. This slice exists for its labels, not for corpus size:
`NotMentioned` is expert-labelled abstention ground truth and `Contradiction` is a
natural claim-support test set.

    uv run python scripts/sample_contractnli.py --size 10
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter

from app.config import get_settings
from app.ingestion.contractnli import ContractNliFormatError, load_corpus
from app.ingestion.manifest import CorpusManifest, ManifestError, manifest_path

DEFAULT_SIZE = 10


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    root = settings.raw_dir / "contract-nli"

    try:
        hypotheses, documents = load_corpus(root)
    except ContractNliFormatError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"ContractNLI: {len(documents)} documents, {len(hypotheses)} hypotheses")

    label_totals: Counter[str] = Counter()
    for document in documents:
        for annotation in document.annotations.values():
            label_totals[annotation.choice] += 1
    print(f"label distribution (corpus-wide): {dict(label_totals)}")

    # Selection: prefer documents carrying BOTH Contradiction and NotMentioned labels.
    # Those two are the entire reason this corpus is here -- a document with neither
    # contributes nothing the CUAD slice does not already provide.
    def score(doc) -> tuple[int, int, str]:
        choices = Counter(a.choice for a in doc.annotations.values())
        has_both = int(choices["Contradiction"] > 0 and choices["NotMentioned"] > 0)
        return (has_both, choices["Contradiction"], doc.file_name)

    eligible = [d for d in documents if d.annotations and len(d.spans) >= 10]
    eligible.sort(key=score, reverse=True)

    both = [d for d in eligible if score(d)[0] == 1]
    print(f"eligible: {len(eligible)}, with both Contradiction and NotMentioned: {len(both)}")

    pool = both or eligible
    rng = random.Random(args.seed)
    top = pool[: max(args.size * 3, args.size)]
    rng.shuffle(top)
    selected = top[: args.size]

    path = manifest_path(settings.data_dir)
    try:
        manifest = CorpusManifest.load_or_empty(path)
    except ManifestError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print()
    created = 0
    for document in selected:
        choices = Counter(a.choice for a in document.annotations.values())
        entry, is_new = manifest.add(
            source_filename=f"contractnli:{document.doc_id}",
            relative_path=f"contract-nli/{document.file_name}",
            corpus_source="contractnli",
            agreement_type="Non-Disclosure Agreement",
            annotated_categories=sorted(
                k for k, a in document.annotations.items() if a.choice != "NotMentioned"
            ),
        )
        created += is_new
        marker = "NEW " if is_new else "kept"
        print(
            f"  {marker} {entry.document_id}  spans={len(document.spans):3d}  "
            f"E={choices['Entailment']:2d} C={choices['Contradiction']:2d} "
            f"NM={choices['NotMentioned']:2d}  {document.file_name[:48]}"
        )

    print(f"\n{created} new, {len(selected) - created} already registered")
    if args.dry_run:
        print("--dry-run: manifest not written")
        return 0

    try:
        manifest.save(path)
    except ManifestError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"Manifest written: {path}  ({len(manifest.documents)} documents)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
