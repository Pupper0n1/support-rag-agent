"""Cross-encoder reranking of the fused candidate list.

The bi-encoder used for kNN scores query and passage independently, which is
what makes it cheap enough to index against, but it cannot model interaction
between the two. A cross-encoder reads (query, passage) jointly and is far
more accurate on the head of the list - at a cost that only makes sense for
a few dozen candidates, hence rerank_depth.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence

from support_agent.types import Reranker, ScoredChunk

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str = DEFAULT_MODEL, max_length: int = 512) -> None:
        self._model_name = model_name
        self._max_length = max_length
        self._model: object | None = None
        self._lock = threading.Lock()

    def _ensure_model(self) -> object:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    logger.info("loading reranker %s", self._model_name)
                    self._model = CrossEncoder(self._model_name, max_length=self._max_length)
        return self._model

    def rerank(self, query: str, chunks: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        if not chunks:
            return []
        model = self._ensure_model()
        pairs = [(query, f"{c.document.title}\n{c.document.text}") for c in chunks]
        raw_scores = model.predict(pairs, show_progress_bar=False)  # type: ignore[attr-defined]
        for chunk, score in zip(chunks, raw_scores, strict=True):
            chunk.rerank_score = float(score)
        return sorted(chunks, key=lambda c: (-(c.rerank_score or 0.0), c.document.doc_id))


class PassthroughReranker(Reranker):
    """Keeps fused order. Used to A/B the reranker's contribution in eval."""

    def rerank(self, query: str, chunks: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        return list(chunks)
