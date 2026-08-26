"""Shared test fixtures.

Fixture content is deliberately synthetic and obviously so (doc-900 and up, invented
clause text). It exists to exercise code paths, and must never be mistaken for eval
dataset content, which is author-written -- see .claude/rules/evals.md.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.ingestion.store import ChunkStore
from app.schemas.chunk import Chunk
from app.schemas.document import Document
from app.schemas.retrieval import RetrievedChunk


def make_chunk(
    chunk_id: str = "doc-900-c0001",
    document_id: str = "doc-900",
    page_number: int = 1,
    section_path: list[str] | None = None,
    text: str = "sample clause text",
    outbound_references: list[str] | None = None,
    section_title: str | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_title="Test Agreement",
        section_path=section_path if section_path is not None else ["8", "8.1"],
        section_title=section_title,
        page_number=page_number,
        text=text,
        token_count=max(1, len(text) // 4),
        outbound_references=outbound_references or [],
    )


def make_document(document_id: str = "doc-900", page_count: int = 3) -> Document:
    return Document(
        document_id=document_id,
        title="Test Agreement",
        source_path=f"synthetic/{document_id}.pdf",
        page_count=page_count,
        corpus_source="cuad",
    )


def make_retrieved(chunk: Chunk, rank: int = 1, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=chunk,
        bm25_rank=rank,
        fused_score=score,
        retrieval_query="test query",
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


@pytest.fixture
def store() -> ChunkStore:
    """A two-document store with known pages and sections."""
    documents = [make_document("doc-900", page_count=3), make_document("doc-901", page_count=1)]
    chunks = [
        make_chunk(
            "doc-900-c0001",
            "doc-900",
            page_number=1,
            section_path=["8", "8.1"],
            section_title="Limitation of Liability",
            text=(
                "8.1 Limitation of Liability. Except as set out in Section 8.3, the "
                "aggregate liability of either party shall not exceed the fees paid in "
                "the twelve months preceding the claim."
            ),
            outbound_references=["8.3"],
        ),
        make_chunk(
            "doc-900-c0002",
            "doc-900",
            page_number=2,
            section_path=["8", "8.3"],
            section_title="Exclusions",
            text=(
                "8.3 Exclusions. The limitation in Section 8.1 shall not apply to "
                "liability arising from gross negligence or wilful misconduct."
            ),
            outbound_references=["8.1"],
        ),
        make_chunk(
            "doc-900-c0003",
            "doc-900",
            page_number=3,
            section_path=["14", "14.2"],
            section_title="Governing Law",
            text="14.2 Governing Law. This Agreement is governed by the laws of Delaware.",
        ),
        make_chunk(
            "doc-901-c0001",
            "doc-901",
            page_number=1,
            section_path=["1"],
            section_title="Definitions",
            text="1. Definitions. Confidential Information means non-public information.",
        ),
    ]
    return ChunkStore(documents, chunks)
