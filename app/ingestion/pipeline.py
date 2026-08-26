"""Ingestion pipeline: manifest -> parse -> chunk -> store.

Orchestration only; every decision worth defending lives in the chunker.

Three source types converge here, and they do NOT share a parse stage:

    cuad         PDF  -> pdf_parser -> chunker          paginated, section-numbered
    synthetic    JSON -> (already text) -> chunker      paginated, derived from CUAD
    contractnli  JSON -> nli_pipeline                   unpaginated, span-segmented

The dispatch is on `corpus_source` rather than on file extension, because the manifest
is the authority on what a document is. Trying to force ContractNLI through the PDF path
would mean synthesizing pages, which SPEC 6.3 rejects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings
from app.ingestion.chunker import chunk_document
from app.ingestion.contractnli import ContractNliFormatError, NliDocument, load_corpus
from app.ingestion.manifest import CorpusManifest, ManifestEntry
from app.ingestion.nli_pipeline import ingest_nli_document
from app.ingestion.pdf_parser import ParsedDocument, PdfParseError, parse_pdf
from app.ingestion.store import ChunkStore
from app.ingestion.synthetic import SyntheticDocument
from app.schemas.chunk import Chunk
from app.schemas.document import Document


@dataclass
class IngestionReport:
    """What happened, in enough detail to diagnose a bad corpus without re-running."""

    ingested: list[str] = field(default_factory=list)
    skipped_scanned: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    chunk_count: int = 0
    page_count: int = 0
    char_count: int = 0
    by_source: dict[str, int] = field(default_factory=dict)

    @property
    def attempted(self) -> int:
        return len(self.ingested) + len(self.skipped_scanned) + len(self.failed)

    def summary(self) -> str:
        lines = [
            f"documents ingested : {len(self.ingested)}/{self.attempted}",
            f"pages              : {self.page_count}",
            f"chunks             : {self.chunk_count}",
            f"characters         : {self.char_count:,}",
        ]
        if self.by_source:
            counts = ", ".join(f"{k}={v}" for k, v in sorted(self.by_source.items()))
            lines.append(f"by corpus_source   : {counts}")
        if self.skipped_scanned:
            lines.append(f"skipped (no text layer): {len(self.skipped_scanned)}")
            lines.extend(f"  {d}" for d in self.skipped_scanned)
        if self.failed:
            lines.append(f"failed: {len(self.failed)}")
            lines.extend(f"  {d}: {why}" for d, why in self.failed)
        return "\n".join(lines)


def title_from_entry(entry: ManifestEntry) -> str:
    """Human-readable title derived from the source filename.

    Display only. Nothing keys on a title: the corpus deliberately contains multiple
    versions of the same agreement, so titles are not unique. See CLAUDE.md.
    """
    stem = Path(entry.source_filename).stem
    # CUAD filenames are either "Company_date_form_EX-n_id_EX-n_Type" or
    # "COMPANY_date-EX-n-TITLE". The trailing component is the readable part.
    for separator in ("_EX-", "-EX-"):
        if separator in stem:
            tail = stem.rsplit(separator, 1)[-1]
            parts = tail.split("_", 1)
            if len(parts) == 2 and parts[1]:
                return parts[1].strip()
            if "-" in tail:
                return tail.split("-", 1)[-1].strip()
    return stem.strip()


def ingest_document(
    entry: ManifestEntry,
    parsed: ParsedDocument,
    settings: Settings,
) -> tuple[Document, list[Chunk]]:
    """Build the Document and its Chunks from a parsed PDF."""
    title = title_from_entry(entry)
    document = Document(
        document_id=entry.document_id,
        title=title,
        agreement_type=entry.agreement_type,
        source_path=entry.relative_path,
        page_count=parsed.page_count,
        corpus_source=entry.corpus_source,
        is_synthetic=entry.is_synthetic,
    )
    chunks = chunk_document(
        parsed,
        document_id=entry.document_id,
        document_title=title,
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        chars_per_token=settings.chars_per_token,
    )
    return document, chunks


def _load_nli_index(settings: Settings) -> dict[int, NliDocument]:
    """Load ContractNLI once and key by document id, or return empty if absent."""
    try:
        _, documents = load_corpus(settings.raw_dir / "contract-nli")
    except ContractNliFormatError:
        return {}
    return {d.doc_id: d for d in documents}


def ingest(
    manifest: CorpusManifest,
    settings: Settings,
    *,
    limit: int | None = None,
) -> tuple[ChunkStore, IngestionReport]:
    """Ingest the active corpus into a ChunkStore."""
    report = IngestionReport()
    documents: list[Document] = []
    chunks: list[Chunk] = []

    entries = sorted(manifest.active.values(), key=lambda e: e.document_id)
    if limit is not None:
        entries = entries[:limit]

    needs_nli = any(e.corpus_source == "contractnli" for e in entries)
    nli_index = _load_nli_index(settings) if needs_nli else {}

    for entry in entries:
        path = settings.raw_dir / entry.relative_path

        # --- ContractNLI: no parser stage, no pagination ---------------------
        if entry.corpus_source == "contractnli":
            doc_id = int(entry.source_filename.removeprefix("contractnli:"))
            nli_document = nli_index.get(doc_id)
            if nli_document is None:
                report.failed.append(
                    (entry.document_id, f"ContractNLI document {doc_id} not found in release")
                )
                continue
            document, document_chunks = ingest_nli_document(entry, nli_document, settings)
            documents.append(document)
            chunks.extend(document_chunks)
            report.ingested.append(entry.document_id)
            report.chunk_count += len(document_chunks)
            report.char_count += len(nli_document.text)
            report.by_source["contractnli"] = report.by_source.get("contractnli", 0) + 1
            continue

        # --- Synthetic: already text, but paginated ---------------------------
        if entry.corpus_source == "synthetic":
            try:
                parsed = SyntheticDocument.load(path).as_parsed()
            except (OSError, KeyError, ValueError) as exc:
                report.failed.append((entry.document_id, f"cannot load {path}: {exc}"))
                continue
            document, document_chunks = ingest_document(entry, parsed, settings)
            documents.append(document)
            chunks.extend(document_chunks)
            report.ingested.append(entry.document_id)
            report.chunk_count += len(document_chunks)
            report.page_count += parsed.page_count
            report.char_count += parsed.char_count
            report.by_source["synthetic"] = report.by_source.get("synthetic", 0) + 1
            continue

        # --- CUAD: the PDF path ----------------------------------------------
        try:
            parsed = parse_pdf(path)
        except PdfParseError as exc:
            report.failed.append((entry.document_id, str(exc)))
            continue

        # A document with no text layer must not silently become an empty,
        # unretrievable entry in the corpus. SPEC 6.2 requires detecting these; an
        # eval task built on one would be unanswerable for a parser reason.
        if parsed.looks_scanned:
            report.skipped_scanned.append(
                f"{entry.document_id} ({parsed.chars_per_page:.0f} chars/page)"
            )
            continue

        document, document_chunks = ingest_document(entry, parsed, settings)
        documents.append(document)
        chunks.extend(document_chunks)
        report.ingested.append(entry.document_id)
        report.chunk_count += len(document_chunks)
        report.page_count += parsed.page_count
        report.char_count += parsed.char_count
        report.by_source[entry.corpus_source] = report.by_source.get(entry.corpus_source, 0) + 1

    return ChunkStore(documents, chunks), report
