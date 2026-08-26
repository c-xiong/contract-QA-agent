"""On-disk store for ingested documents and chunks.

JSONL, one object per line, two files. Deliberately not a database: the corpus is
about sixty documents and a few thousand chunks, it is rebuilt from scratch by
``scripts/ingest.py`` whenever chunking changes, and a plain text format means an
ingestion bug can be diagnosed with `grep`. See docs/decisions.md.

The store is also the authority the citation verifier consults. "Does doc-014 have a
page 12" is answered from here, not from the model's recollection.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from app.schemas.chunk import Chunk
from app.schemas.document import Document

DOCUMENTS_FILE = "documents.jsonl"
CHUNKS_FILE = "chunks.jsonl"


class StoreError(RuntimeError):
    """The store is missing, unreadable, or internally inconsistent."""


class ChunkStore:
    """In-memory index over the ingested corpus, loaded from JSONL."""

    def __init__(self, documents: Iterable[Document], chunks: Iterable[Chunk]) -> None:
        self._documents: dict[str, Document] = {d.document_id: d for d in documents}
        self._chunks: dict[str, Chunk] = {}
        self._by_document: dict[str, list[Chunk]] = {}
        self._pages: dict[str, set[int]] = {}
        self._sections: dict[str, set[str]] = {}

        for chunk in chunks:
            if chunk.chunk_id in self._chunks:
                raise StoreError(f"Duplicate chunk_id {chunk.chunk_id!r}")
            self._chunks[chunk.chunk_id] = chunk
            self._by_document.setdefault(chunk.document_id, []).append(chunk)
            pages = self._pages.setdefault(chunk.document_id, set())
            if chunk.page_number is not None:
                pages.add(chunk.page_number)
            sections = self._sections.setdefault(chunk.document_id, set())
            sections.update(chunk.section_path)

        orphans = sorted(set(self._by_document) - set(self._documents))
        if orphans:
            raise StoreError(
                f"Chunks reference documents not in the store: {', '.join(orphans[:5])}. "
                "Re-run scripts/ingest.py."
            )

    # --- Persistence ----------------------------------------------------------

    @classmethod
    def load(cls, processed_dir: Path) -> ChunkStore:
        docs_path = processed_dir / DOCUMENTS_FILE
        chunks_path = processed_dir / CHUNKS_FILE
        for path in (docs_path, chunks_path):
            if not path.exists():
                raise StoreError(
                    f"No ingested corpus at {path}. Run: uv run python scripts/ingest.py"
                )
        documents = [Document.model_validate_json(line) for line in _lines(docs_path)]
        chunks = [Chunk.model_validate_json(line) for line in _lines(chunks_path)]
        return cls(documents, chunks)

    def save(self, processed_dir: Path) -> None:
        processed_dir.mkdir(parents=True, exist_ok=True)
        docs_path = processed_dir / DOCUMENTS_FILE
        chunks_path = processed_dir / CHUNKS_FILE

        with docs_path.open("w", encoding="utf-8") as handle:
            for document_id in sorted(self._documents):
                payload = self._documents[document_id].model_dump(mode="json")
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

        with chunks_path.open("w", encoding="utf-8") as handle:
            for chunk_id in sorted(self._chunks):
                payload = self._chunks[chunk_id].model_dump(mode="json")
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # --- Lookup ---------------------------------------------------------------

    @property
    def documents(self) -> list[Document]:
        return [self._documents[k] for k in sorted(self._documents)]

    @property
    def chunks(self) -> list[Chunk]:
        return [self._chunks[k] for k in sorted(self._chunks)]

    def __len__(self) -> int:
        return len(self._chunks)

    def get_document(self, document_id: str) -> Document | None:
        return self._documents.get(document_id)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def chunks_for(self, document_id: str) -> list[Chunk]:
        return list(self._by_document.get(document_id, ()))

    def has_document(self, document_id: str) -> bool:
        return document_id in self._documents

    def has_page(self, document_id: str, page_number: int) -> bool:
        """True if the document was ingested and has a chunk on that page.

        Deliberately stricter than "page_number <= page_count": a page that produced
        no text produces no chunk, so nothing can have been cited from it.

        An unpaginated document (ContractNLI) has an empty page set, so every page
        citation against it is correctly rejected -- there is no page 1 to cite.
        """
        return page_number in self._pages.get(document_id, set())

    def is_paginated(self, document_id: str) -> bool:
        """False for sources that ship extracted text rather than paginated PDFs."""
        return bool(self._pages.get(document_id))

    def has_section(self, document_id: str, section_id: str) -> bool:
        return section_id in self._sections.get(document_id, set())

    def pages_of(self, document_id: str) -> set[int]:
        return set(self._pages.get(document_id, set()))

    def sections_of(self, document_id: str) -> set[str]:
        return set(self._sections.get(document_id, set()))

    def find_section(self, document_id: str, section_id: str) -> list[Chunk]:
        """Chunks whose section_path contains `section_id`, in document order."""
        return [c for c in self.chunks_for(document_id) if section_id in c.section_path]

    def find_page(self, document_id: str, page_number: int) -> list[Chunk]:
        return [c for c in self.chunks_for(document_id) if c.page_number == page_number]


def _lines(path: Path) -> Iterable[str]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield line
