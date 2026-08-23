"""FastMCP server wiring the retrieval pipeline into an MCP tool.

The server is built by a factory rather than at import time so the retriever
(and its model weights) can be injected - tests pass a stub, Lambda passes the
warm singleton.
"""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from support_agent.mcp_tools.schemas import SearchHit, SearchResult
from support_agent.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)

SEARCH_TEXT_PREVIEW_CHARS = 900


def build_server(retriever: HybridRetriever) -> FastMCP:
    mcp = FastMCP("support-tools")

    @mcp.tool()
    def search_knowledge_base(query: str, product_area: str | None = None) -> SearchResult:
        """Search the help-center knowledge base for articles relevant to a ticket.

        Use this first for every ticket. `query` should be the customer's
        problem in their own words plus the subject line. Pass `product_area`
        only when the ticket clearly belongs to one of: billing, auth, api,
        data, integrations. Returns ranked hits with a relevance score in
        [0, 1]; a top_score below ~0.45 means the KB probably does not cover
        the question.
        """
        result = retriever.retrieve(query, product_area=product_area)
        hits: list[SearchHit] = [
            {
                "doc_id": c.document.doc_id,
                "title": c.document.title,
                "text": c.document.text[:SEARCH_TEXT_PREVIEW_CHARS],
                "source_url": c.document.source_url,
                "product_area": c.document.product_area,
                "score": round(_normalise(c.final_score, result.reranked), 4),
            }
            for c in result.chunks
        ]
        top = max((h["score"] for h in hits), default=0.0)
        logger.info("search q=%r hits=%d top=%.3f", query[:60], len(hits), top)
        return {
            "query": result.query,
            "hits": hits,
            "top_score": top,
            "bm25_hits": result.bm25_hits,
            "knn_hits": result.knn_hits,
            "reranked": result.reranked,
        }

    return mcp


def _normalise(score: float, reranked: bool) -> float:
    """Map a stage score onto [0, 1] so the model sees one scale.

    Cross-encoder logits are unbounded and roughly centred on 0, so they go
    through a sigmoid. RRF scores are already small positives bounded by
    (2 / (k + 1)); scale them so a document ranked first in both arms is ~1.
    """
    import math

    if reranked:
        return 1.0 / (1.0 + math.exp(-score))
    return min(1.0, score * 30.5)
