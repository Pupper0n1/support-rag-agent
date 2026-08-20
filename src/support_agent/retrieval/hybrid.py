"""Hybrid retriever: BM25 + dense kNN, fused with RRF, optionally reranked."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from support_agent.config import RetrievalSettings
from support_agent.retrieval.es_client import KnowledgeBaseIndex
from support_agent.retrieval.fusion import reciprocal_rank_fusion
from support_agent.types import EmbeddingProvider, Reranker, RetrievalResult

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(
        self,
        index: KnowledgeBaseIndex,
        embeddings: EmbeddingProvider,
        settings: RetrievalSettings,
        reranker: Reranker | None = None,
    ) -> None:
        self._index = index
        self._embeddings = embeddings
        self._settings = settings
        self._reranker = reranker
        # Two arms, two threads. The ES round-trips dominate latency and are
        # independent, so running them concurrently roughly halves p50.
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="retrieval")

    def retrieve(self, query: str, product_area: str | None = None) -> RetrievalResult:
        s = self._settings
        query = query.strip()
        if not query:
            return RetrievalResult(query=query)

        vector = self._embeddings.embed_query(query)

        bm25_future = self._pool.submit(self._index.bm25_search, query, s.bm25_size, product_area)
        knn_future = self._pool.submit(
            self._index.knn_search, vector, s.knn_size, s.knn_num_candidates, product_area
        )
        bm25_hits = bm25_future.result()
        knn_hits = knn_future.result()

        fused = reciprocal_rank_fusion(bm25_hits, knn_hits, k=s.rrf_k)
        logger.debug(
            "query=%r bm25=%d knn=%d fused=%d", query[:60], len(bm25_hits), len(knn_hits), len(fused)
        )

        reranked = False
        if self._reranker is not None and fused:
            head = fused[: s.rerank_depth]
            fused = self._reranker.rerank(query, head)
            reranked = True

        return RetrievalResult(
            query=query,
            chunks=fused[: s.top_k],
            bm25_hits=len(bm25_hits),
            knn_hits=len(knn_hits),
            reranked=reranked,
        )
