"""The corpus manifest: the single authority on what ``doc-NNN`` means.

Every citation, every eval task's ``expected_document_ids``, and every retrieval
allowlist keys on a document ID. If an ID can ever come to mean a different
contract, all three break silently -- the ID still resolves to *a* document, so
nothing raises and no test fails. See docs/decisions.md, 2026-08-25.

The invariant this module exists to hold:

    **An assigned document_id is never reused, renumbered, or repointed.**

Consequences, enforced below rather than documented and hoped for:

- IDs are assigned by appending. ``next_id`` is max(existing) + 1, including retired IDs.
- Re-adding a source file returns its existing ID instead of minting a new one.
- Removing a document retires the entry; the entry stays so its ID stays spent.
- Resolving an unknown ID is an error, never an implicit registration.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.document import CorpusSource

MANIFEST_FILENAME = "corpus_manifest.json"
MANIFEST_VERSION = 1


class ManifestError(RuntimeError):
    """The manifest is missing, malformed, or being asked to violate its invariant."""


class ManifestEntry(BaseModel):
    """One document's identity, fixed at the moment it entered the corpus."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(pattern=r"^doc-\d{3,}$")
    source_filename: str = Field(
        description=(
            "Filename exactly as it appears in the ground-truth source "
            "(CUAD master_clauses.csv). This is the join key to expert annotations, "
            "so it is stored verbatim, irregularities and all."
        )
    )
    relative_path: str = Field(description="Path to the source file, relative to data/raw.")
    corpus_source: CorpusSource
    agreement_type: str | None = None
    annotated_categories: list[str] = Field(default_factory=list)
    added: date
    is_synthetic: bool = False
    retired: bool = False
    retired_reason: str | None = None


class CorpusManifest(BaseModel):
    """Append-only registry mapping document IDs to source files."""

    model_config = ConfigDict(extra="forbid")

    manifest_version: int = MANIFEST_VERSION
    sampling_criterion: str = Field(
        default="",
        description=(
            "Prose statement of how this corpus was selected. SPEC 6.2 requires the "
            "criterion be recorded so the corpus is reproducible, not just the result."
        ),
    )
    documents: dict[str, ManifestEntry] = Field(default_factory=dict)

    # --- Loading and saving ---------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> CorpusManifest:
        if not path.exists():
            raise ManifestError(
                f"No corpus manifest at {path}. Run: uv run python scripts/sample_contracts.py"
            )
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ManifestError(f"{path} is not valid JSON: {exc}") from exc
        manifest = cls.model_validate(raw)
        manifest._check_invariants(path)
        return manifest

    @classmethod
    def load_or_empty(cls, path: Path) -> CorpusManifest:
        return cls.load(path) if path.exists() else cls()

    def save(self, path: Path) -> None:
        """Write the manifest with stable key ordering so diffs stay reviewable.

        This file is checked in. A merge conflict in it is a real conflict about
        corpus identity, so it must never be reordered gratuitously.
        """
        self._check_invariants(path)
        ordered = CorpusManifest(
            manifest_version=self.manifest_version,
            sampling_criterion=self.sampling_criterion,
            documents={k: self.documents[k] for k in sorted(self.documents)},
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = ordered.model_dump(mode="json", exclude_none=False)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _check_invariants(self, path: Path) -> None:
        for doc_id, entry in self.documents.items():
            if entry.document_id != doc_id:
                raise ManifestError(
                    f"{path}: key {doc_id!r} does not match entry.document_id "
                    f"{entry.document_id!r}. The manifest has been hand-edited incorrectly."
                )
        seen: dict[str, str] = {}
        for doc_id, entry in sorted(self.documents.items()):
            prior = seen.get(entry.source_filename)
            if prior is not None:
                raise ManifestError(
                    f"{path}: {entry.source_filename!r} is registered under both {prior} "
                    f"and {doc_id}. One source file gets exactly one ID."
                )
            seen[entry.source_filename] = doc_id

    # --- Lookup ---------------------------------------------------------------

    def get(self, document_id: str) -> ManifestEntry:
        entry = self.documents.get(document_id)
        if entry is None:
            raise ManifestError(
                f"Unknown document_id {document_id!r}. Known IDs are assigned only by "
                "scripts/sample_contracts.py; this one was never registered."
            )
        return entry

    def find_by_source_filename(self, source_filename: str) -> ManifestEntry | None:
        for entry in self.documents.values():
            if entry.source_filename == source_filename:
                return entry
        return None

    @property
    def active(self) -> dict[str, ManifestEntry]:
        return {k: v for k, v in self.documents.items() if not v.retired}

    # --- Mutation -------------------------------------------------------------

    def next_id(self) -> str:
        """Return the next unused ID, counting retired entries as used."""
        highest = 0
        for doc_id in self.documents:
            highest = max(highest, int(doc_id.removeprefix("doc-")))
        return f"doc-{highest + 1:03d}"

    def add(
        self,
        *,
        source_filename: str,
        relative_path: str,
        corpus_source: CorpusSource,
        agreement_type: str | None = None,
        annotated_categories: list[str] | None = None,
        is_synthetic: bool = False,
        added: date | None = None,
    ) -> tuple[ManifestEntry, bool]:
        """Register a source file, or return its existing entry.

        Returns ``(entry, created)``. Re-running the sampler over an overlapping
        selection is therefore safe: already-registered documents keep their IDs.
        """
        existing = self.find_by_source_filename(source_filename)
        if existing is not None:
            return existing, False

        entry = ManifestEntry(
            document_id=self.next_id(),
            source_filename=source_filename,
            relative_path=relative_path,
            corpus_source=corpus_source,
            agreement_type=agreement_type,
            annotated_categories=sorted(annotated_categories or []),
            added=added or date.today(),
            is_synthetic=is_synthetic,
        )
        self.documents[entry.document_id] = entry
        return entry, True

    def retire(self, document_id: str, reason: str) -> ManifestEntry:
        """Mark a document withdrawn, permanently spending its ID."""
        entry = self.get(document_id)
        retired = entry.model_copy(update={"retired": True, "retired_reason": reason})
        self.documents[document_id] = retired
        return retired


def manifest_path(data_dir: Path) -> Path:
    """The manifest lives beside the corpus it describes, and is checked in."""
    return data_dir / MANIFEST_FILENAME
