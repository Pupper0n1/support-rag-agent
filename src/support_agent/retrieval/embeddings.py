"""Dense embedding provider.

Wraps a sentence-transformers bi-encoder behind the EmbeddingProvider
protocol so the retrieval layer never imports the model library directly and
tests can substitute a deterministic stub.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence

from support_agent.types import EmbeddingProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"

# The bi-encoder was trained with an asymmetric prefix convention; queries and
# passages are embedded into the same space but with different framing.
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


class SentenceTransformerEmbeddings(EmbeddingProvider):
    """Lazily-loaded bi-encoder.

    Loading is deferred and guarded by a lock because a cold Lambda may take
    several seconds to pull the weights, and two concurrent invocations on the
    same container should not both pay for it.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 32) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model: object | None = None
        self._lock = threading.Lock()
        self._dimensions = 768

    def _ensure_model(self) -> object:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    logger.info("loading embedding model %s", self._model_name)
                    model = SentenceTransformer(self._model_name)
                    self._dimensions = int(model.get_sentence_embedding_dimension())
                    self._model = model
        return self._model

    @property
    def dimensions(self) -> int:
        self._ensure_model()
        return self._dimensions

    def embed_query(self, text: str) -> list[float]:
        return self._encode([QUERY_PREFIX + text])[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode([PASSAGE_PREFIX + t for t in texts])

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._ensure_model()
        vectors = model.encode(  # type: ignore[attr-defined]
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(v) for v in row] for row in vectors]
