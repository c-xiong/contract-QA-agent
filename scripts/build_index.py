"""Build the dense FAISS index over the ingested corpus.

Run after scripts/ingest.py, and again whenever the chunker changes -- the index is
keyed on chunk ids, and a stale index is detected and refused at load rather than
silently returning chunks that no longer exist.

    uv run python scripts/build_index.py
"""

from __future__ import annotations

import argparse
import sys
import time

from app.config import get_settings
from app.ingestion.store import ChunkStore, StoreError
from app.retrieval.dense import DenseIndexError, DenseRetriever
from app.retrieval.embeddings import DEFAULT_MODEL
from app.retrieval.windowed import WindowedDenseRetriever


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--windows", action="store_true", help="build a separate token-window index"
    )
    args = parser.parse_args()

    settings = get_settings()
    try:
        store = ChunkStore.load(settings.processed_dir)
    except StoreError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"corpus: {len(store.documents)} documents, {len(store)} chunks")
    print(f"model : {args.model}")
    print("embedding...")

    started = time.monotonic()
    try:
        retriever_type = WindowedDenseRetriever if args.windows else DenseRetriever
        retriever = retriever_type.build(store, model_id=args.model, batch_size=args.batch_size)
    except DenseIndexError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    elapsed = time.monotonic() - started

    destination = settings.index_dir / "windows" if args.windows else settings.index_dir
    retriever.save(destination)
    rate = len(store) / elapsed if elapsed else 0.0
    print(f"embedded {len(store)} chunks in {elapsed:.1f}s ({rate:.0f} chunks/s)")
    print(f"index written: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
