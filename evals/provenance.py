"""Small, shared run provenance; historical artifacts are never backfilled."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.ingestion.store import ChunkStore

ARTIFACT_SCHEMA_VERSION = "2"


def chunks_sha256(store: ChunkStore) -> str:
    """Hash canonical chunk content and locators, independent of insertion order."""
    # DECISION: IDs alone miss edits that preserve identifiers. Full canonical chunks
    # detect changed text, boundaries, and locators without hashing unrelated files.
    digest = hashlib.sha256()
    for chunk in sorted(store.chunks, key=lambda item: item.chunk_id):
        encoded = json.dumps(
            chunk.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        digest.update(encoded.encode("utf-8") + b"\n")
    return digest.hexdigest()


def git_state(repo: Path | None = None) -> tuple[str | None, bool | None]:
    root = repo or Path(__file__).resolve().parents[1]
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        # An exported source tree has no git metadata. Null is honest; a guessed SHA is not.
        return None, None
    return sha, bool(dirty)


def capture_provenance(
    store: ChunkStore,
    *,
    dataset_name: str,
    dataset_version: str,
    retrieval_arm: str,
    top_k: int,
    model_id: str,
    grader_versions: dict[str, str] | None = None,
    started_at: str | None = None,
) -> dict[str, object]:
    """Capture before running; fill actual grader versions when scores are available."""
    sha, dirty = git_state()
    dataset_path = Path(__file__).parent / "datasets" / dataset_name / "tasks.jsonl"
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": str(uuid4()),
        "started_at": started_at or datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": sha,
        "git_worktree_dirty": dirty,
        "source_sha256": source_sha256(),
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        if dataset_path.is_file()
        else None,
        "document_count": len(store.documents),
        "chunk_count": len(store),
        "chunks_sha256": chunks_sha256(store),
        "retrieval_arm": retrieval_arm,
        "top_k": top_k,
        "model_id": model_id,
        "grader_versions": dict(grader_versions or {}),
    }


def source_sha256() -> str:
    """Identify the actual Python source even when the worktree is not committed."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    paths = sorted(
        path
        for folder in ("app", "evals", "experiments", "scripts")
        for path in (root / folder).rglob("*.py")
    )
    for path in paths:
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()
