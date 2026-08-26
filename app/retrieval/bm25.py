"""BM25 retrieval over the chunk store.

BM25 is the lexical arm of Experiment A. It is strong on exactly what dense
retrieval is weak on: section labels, party names, dates, dollar amounts, and legal
terms of art that must match exactly ("indemnify", "gross negligence", "Nevada").

One property of the implementation is worth knowing before reading any Experiment A
result. `rank_bm25`'s BM25Okapi computes IDF as log((N - n + 0.5) / (n + 0.5)), with
no +1 smoothing, so a term occurring in more than roughly half the chunks gets a
NEGATIVE idf. Its epsilon floor scales with average_idf, which is itself negative
when many terms are common, so it does not rescue those terms either.

The practical effect is that high-document-frequency words suppress themselves. That
is usually desirable -- it is stopword removal without a stopword list -- but it also
means a query composed entirely of common contract vocabulary ("the agreement shall
provide") can score every chunk at or below zero and retrieve nothing. On this corpus
that is rare; on a small or homogeneous corpus it is easy to hit. Worth stating in the
Experiment A writeup rather than discovering from an anomalous per-category number.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from app.ingestion.store import ChunkStore
from app.retrieval.base import apply_filters
from app.schemas.chunk import Chunk
from app.schemas.retrieval import RetrievalFilters, RetrievedChunk

# Tokenization: lowercase alphanumeric runs, keeping dotted section numbers and
# decimal amounts intact. "8.1" must survive as one token, not become "8" and "1",
# or a query for Section 8.1 matches every chunk containing the digit 8.
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.casefold())


class Bm25Retriever:
    """In-memory BM25 index over every chunk in the store.

    The index is built once at construction. The corpus is a few thousand chunks, so
    this is well under a second and avoids a persistence format that would have to be
    invalidated whenever the chunker changes.
    """

    name = "bm25"

    def __init__(self, store: ChunkStore) -> None:
        self._store = store
        self._chunks: list[Chunk] = store.chunks
        self._synthetic: list[bool] = []
        for chunk in self._chunks:
            document = store.get_document(chunk.document_id)
            self._synthetic.append(bool(document and document.is_synthetic))

        corpus = [self._index_text(chunk) for chunk in self._chunks]
        # BM25Okapi rejects an empty corpus; an empty store is a caller error worth
        # surfacing here rather than as an opaque IndexError at query time.
        if not corpus:
            raise ValueError("Cannot build a BM25 index over an empty store")
        self._bm25 = BM25Okapi([tokenize(text) for text in corpus])

    @staticmethod
    def _index_text(chunk: Chunk) -> str:
        """Index the section title and path alongside the body text.

        A query for "governing law" should match the chunk headed "Section 14.2
        Governing Law" even when the body says only "the laws of the State of
        Nevada". The heading is already inside `chunk.text`, but repeating the title
        and the section number gives them weight proportional to their diagnostic
        value, and makes a query naming a section number retrievable at all.
        """
        parts = [chunk.text]
        if chunk.section_title:
            parts.append(chunk.section_title)
        parts.extend(chunk.section_path)
        return "\n".join(parts)

    async def search(
        self,
        query: str,
        top_k: int,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievedChunk]:
        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        candidates: list[tuple[float, int]] = []
        for index, score in enumerate(scores):
            chunk = self._chunks[index]
            if not apply_filters(chunk.document_id, self._synthetic[index], filters):
                continue
            # A non-positive BM25 score means the query contributed nothing. Returning
            # such a chunk to fill out top_k gives the writer irrelevant evidence and
            # invites a confident answer built on nothing.
            #
            # Non-positive covers two cases, and both should be dropped:
            #   - no query term appears in the chunk at all;
            #   - every query term appears in (nearly) every chunk, so Okapi assigns it
            #     zero or negative IDF. A term present corpus-wide carries no signal,
            #     and rank_bm25's epsilon floor scales with average_idf, which is itself
            #     negative in that case, so it does not rescue the score.
            # The second case is easy to hit on a tiny corpus and vanishingly rare on a
            # real one, but the right response is the same: return nothing and let the
            # agent abstain rather than answer from chunks BM25 could not rank.
            if score <= 0.0:
                continue
            candidates.append((float(score), index))

        # Sort by score descending, then chunk_id ascending so ties are deterministic.
        # Non-deterministic ordering would make eval runs unreproducible for a reason
        # that has nothing to do with the system under test.
        candidates.sort(key=lambda pair: (-pair[0], self._chunks[pair[1]].chunk_id))

        return [
            RetrievedChunk(
                chunk=self._chunks[index],
                bm25_rank=rank,
                dense_rank=None,
                fused_score=score,
                rerank_score=None,
                retrieval_query=query,
                pulled_by="search",
            )
            for rank, (score, index) in enumerate(candidates[:top_k], start=1)
        ]

    @property
    def total_candidates(self) -> int:
        return len(self._chunks)
