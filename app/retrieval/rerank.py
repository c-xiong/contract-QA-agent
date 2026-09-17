"""Cross-encoder reranking over fused candidates.

DECISION-BEARING MODULE. See docs/decisions.md.

SPEC 7.2 lists this as the highest-priority optional item and the fourth arm of
Experiment A. It is the one place in the retrieval stack where the query and the chunk
are read together rather than compared as independent vectors.

Bi-encoder (dense.py):    embed(query) . embed(chunk)   -- chunk embedded once, offline
Cross-encoder (here):     score(query, chunk)           -- both read together, per pair

The cross-encoder is far more accurate and far more expensive: it cannot precompute
anything, so it costs one forward pass per candidate. That is exactly why it goes last,
over a short candidate list, and never over the corpus.
"""

from __future__ import annotations

import asyncio

from sentence_transformers import CrossEncoder

from app.retrieval.base import Retriever
from app.schemas.retrieval import RetrievalFilters, RetrievedChunk

# DECISION: ms-marco-MiniLM-L-6-v2 as the cross-encoder.
#   ~90 MB, trained on MS MARCO passage ranking. Like the bi-encoder it is
#   general-purpose and has no legal-domain training, so the same caveat applies: its
#   Experiment A number is what THIS reranker does, not what reranking can do.
#   Rejected: a larger cross-encoder (L-12, or a bge reranker). Better, and slow enough
#   that reranking 40 candidates per query becomes noticeable in an eval sweep over
#   dozens of tasks with multiple trials.
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# DECISION: rerank the top 40 fused candidates, not the top 200 and not the top 5.
#   Reranking is O(candidates) forward passes, so the number is a direct cost knob.
#   Too few and the reranker cannot rescue anything -- if the right chunk is at fused
#   rank 50 and we only rerank 40, reranking cannot help, and the arm measures nothing.
#   Too many and the eval sweep slows down for candidates that fusion already ranked as
#   hopeless.
#   40 is 8x the default top_k of 5. It should be tuned against the measured fused-rank
#   distribution of correct chunks -- if Experiment A shows correct chunks routinely
#   land beyond rank 40, this number is wrong and the arm is being under-served.
DEFAULT_RERANK_DEPTH = 40


class CrossEncoderReranker:
    """Wraps a retriever and re-scores its top candidates with a cross-encoder."""

    def __init__(
        self,
        base: Retriever,
        *,
        model_id: str = DEFAULT_RERANK_MODEL,
        depth: int = DEFAULT_RERANK_DEPTH,
    ) -> None:
        self._base = base
        self._depth = depth
        self.model_id = model_id
        self._model = CrossEncoder(model_id)
        self.name = f"{base.name}+rerank"

    def _pair_text(self, item: RetrievedChunk) -> str:
        """What the cross-encoder reads as the document side of the pair.

        Mirrors DenseRetriever.index_text: heading first, then body. Keeping the two
        consistent means a rerank win is a win from joint query-document reading, not
        from the reranker happening to see a section title the bi-encoder did not.
        """
        chunk = item.chunk
        if chunk.section_title:
            return f"{chunk.section_title}\n{chunk.text}"
        return chunk.text

    async def search(
        self,
        query: str,
        top_k: int,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievedChunk]:
        candidates = await self._base.search(query, top_k=self._depth, filters=filters)
        if not candidates:
            return []

        pairs = [(query, self._pair_text(item)) for item in candidates]

        # CrossEncoder.predict is synchronous and CPU-bound. Off-loading it keeps the
        # event loop free, which matters once the API layer serves concurrent requests.
        scores = await asyncio.to_thread(self._model.predict, pairs, show_progress_bar=False)

        # DECISION: rerank_score replaces the ordering, but fused_score is preserved.
        #   Both numbers are kept on the result so a trace can show that fusion ranked a
        #   chunk 12th and the reranker promoted it to 1st. Overwriting fused_score would
        #   erase exactly the evidence needed to say whether reranking helped, which is
        #   the question the fourth arm exists to answer.
        rescored = [
            item.model_copy(update={"rerank_score": float(score)})
            for item, score in zip(candidates, scores, strict=True)
        ]
        rescored.sort(key=lambda i: (-(i.rerank_score or 0.0), i.chunk.chunk_id))
        return rescored[:top_k]
