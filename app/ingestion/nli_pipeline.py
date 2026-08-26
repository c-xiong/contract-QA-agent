"""Ingestion path for unpaginated text sources (ContractNLI).

A second, deliberately separate path from `pipeline.py`. ContractNLI ships extracted
text with a span table, not PDFs, so there is no parser stage and no pagination. Trying
to force it through the PDF path would mean synthesizing pages, which SPEC 6.3 rejects:
fabricated pagination in a system whose selling point is provenance is the wrong trade.

Chunks from this path carry `page_number = None` and a section id of the form `s37`,
derived from the document's own span table. Citations render as `[doc-051, §s37]`,
which the parser and verifier already accept.
"""

from __future__ import annotations

from app.config import Settings
from app.ingestion.contractnli import NliDocument
from app.ingestion.manifest import ManifestEntry
from app.ingestion.text import clean_extracted_text, estimate_tokens
from app.schemas.chunk import Chunk
from app.schemas.document import Document

# DECISION: one chunk per annotation span, not fixed-size windows.
#   ContractNLI's span table is the corpus author's own segmentation, and the evidence
#   labels are expressed as indices into it. Chunking on those exact boundaries makes
#   "was the labelled evidence retrieved?" an exact set-membership question rather than
#   an overlap heuristic -- the retrieval graders get a clean signal on this slice.
#   Rejected: re-chunking the text with our own chunker for consistency with CUAD. It
#   would be more uniform and strictly worse: every evidence label would then need fuzzy
#   alignment back onto our boundaries, adding a measurement error to the one slice
#   whose labels are exact.
#   Consequence to state: chunk-size distribution differs between the CUAD and
#   ContractNLI slices, so a per-slice breakdown is required when reporting retrieval
#   numbers. An aggregate over both would confound corpus with chunking.
MIN_SPAN_CHARS = 20


def title_from_nli(document: NliDocument) -> str:
    """Human-readable title. Display only; nothing keys on it."""
    name = document.file_name.rsplit("/", 1)[-1]
    for suffix in (".pdf", ".PDF", ".txt"):
        name = name.removesuffix(suffix)
    return name.replace("_", " ").strip() or f"NDA {document.doc_id}"


def ingest_nli_document(
    entry: ManifestEntry,
    document: NliDocument,
    settings: Settings,
) -> tuple[Document, list[Chunk]]:
    """Build the Document and Chunks for one ContractNLI NDA."""
    title = title_from_nli(document)

    doc = Document(
        document_id=entry.document_id,
        title=title,
        agreement_type="Non-Disclosure Agreement",
        source_path=entry.relative_path,
        # page_count is honestly zero: this source has no pages. Callers must not read
        # it as "empty document" -- Document.is_empty is about extraction failure, and
        # `corpus_source` is what distinguishes the two.
        page_count=0,
        corpus_source="contractnli",
        is_synthetic=entry.is_synthetic,
    )

    chunks: list[Chunk] = []
    for index, (start, end) in enumerate(document.spans):
        raw = document.text[start:end]
        text = clean_extracted_text(raw)
        if len(text.strip()) < MIN_SPAN_CHARS:
            continue

        chunks.append(
            Chunk(
                chunk_id=f"{entry.document_id}-s{index:04d}",
                document_id=entry.document_id,
                document_title=title,
                # The span index IS the addressable location for this source. Prefixed
                # with "s" so a span id can never be confused with a contract section
                # number like 8.1 when it appears inside a citation.
                section_path=[f"s{index}"],
                section_title=None,
                page_number=None,
                text=text,
                token_count=estimate_tokens(text, settings.chars_per_token),
                outbound_references=[],
            )
        )

    return doc, chunks
