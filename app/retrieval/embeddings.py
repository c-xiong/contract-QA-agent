"""Embedding backend for dense retrieval.

A local sentence-transformers model, not a hosted embedding API.

DECISION: embeddings run locally rather than through a provider API.
  Chosen for three reasons that all point the same way. Ingestion embeds every chunk in
  the corpus, so a hosted embedder makes `scripts/ingest.py` cost money and require a
  key -- and this project's whole default is that nothing costs money unless
  CRA_LIVE_MODEL=1. Second, Experiment A must be re-runnable at will; a network
  dependency in the inner loop makes it flaky. Third, Anthropic has no embeddings
  endpoint, so a hosted embedder would mean a second provider, which SPEC 7.3 lists as
  a non-goal.
  Rejected: Voyage AI (Anthropic's recommended partner) and OpenAI embeddings. Both are
  better models. Neither is worth a second provider dependency and a per-run bill for a
  portfolio project whose retrieval comparison is the point.
  Consequence to state in the writeup: all-MiniLM-L6-v2 is a small, general-purpose
  model with a 256-token window and no legal-domain training. Dense retrieval's numbers
  in Experiment A are therefore a floor, not a ceiling. That is an honest result -- it
  says what THIS dense arm does -- but it must not be reported as "dense retrieval
  underperforms BM25 on contracts" in general.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

# DECISION: all-MiniLM-L6-v2 over a larger model.
#   90 MB, 384 dimensions, fast enough to embed several thousand chunks in seconds on a
#   laptop CPU. A larger model (bge-large, gte-large) would score better and would make
#   ingestion slow enough to discourage re-running it, which is the operation this
#   project needs to stay cheap because the chunker changes.
#   The model id is configurable, so swapping it is an experiment arm rather than a
#   rewrite.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# The model truncates beyond this. Chunks longer than roughly 256 word-pieces lose their
# tail, which is a real limitation of this arm and is reported rather than hidden.
MODEL_MAX_TOKENS = 256


class Embedder:
    """Encodes text to L2-normalized vectors."""

    def __init__(self, model_id: str = DEFAULT_MODEL) -> None:
        self.model_id = model_id
        self._model = SentenceTransformer(model_id)
        self.dimension = int(self._model.get_embedding_dimension() or 0)

    def encode(self, texts: list[str], *, batch_size: int = 64) -> np.ndarray:
        """Return an (n, dim) float32 array of unit-length vectors.

        DECISION: vectors are L2-normalized at encode time, so inner product IS cosine
        similarity and FAISS can use IndexFlatIP.
          Doing it here rather than at query time means it cannot be forgotten on one
          side of the comparison, which would silently produce a similarity that is not
          cosine and rank results plausibly but wrongly.
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")

    def encode_one(self, text: str) -> np.ndarray:
        vector: np.ndarray = self.encode([text])[0]
        return vector


@lru_cache(maxsize=2)
def get_embedder(model_id: str = DEFAULT_MODEL) -> Embedder:
    """Cached so the model is loaded once per process, not once per retriever."""
    return Embedder(model_id)
