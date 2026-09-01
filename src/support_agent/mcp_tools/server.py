"""FastMCP server wiring the retrieval pipeline into an MCP tool.

The server is built by a factory rather than at import time so the retriever
(and its model weights) can be injected - tests pass a stub, Lambda passes the
warm singleton.
"""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from support_agent.mcp_tools.escalation import EscalationSink, suggest_team
from support_agent.mcp_tools.schemas import (
    DraftReplyResult,
    EscalationRecord,
    EscalationResult,
    InfoRequestResult,
    SearchHit,
    SearchResult,
)
from support_agent.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)

SEARCH_TEXT_PREVIEW_CHARS = 900
VALID_PRIORITIES = frozenset({"low", "normal", "high", "urgent"})


def build_server(retriever: HybridRetriever, escalations: EscalationSink) -> FastMCP:
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

    @mcp.tool()
    def escalate_ticket(
        ticket_id: str,
        reason: str,
        summary: str,
        priority: str = "normal",
        product_area: str | None = None,
    ) -> EscalationResult:
        """Hand the ticket to a human support team.

        Call this when the knowledge base does not answer the question
        (low top_score), when the customer is asking for something only a
        human can do (refunds outside policy, account recovery, legal or
        GDPR requests, bug reports with reproduction steps), or when the
        customer is clearly frustrated after a prior automated reply.
        `summary` is what the human agent reads first: two or three
        sentences with the concrete facts from the ticket. `reason` is one
        short phrase categorising why automation stopped.
        """
        if priority not in VALID_PRIORITIES:
            priority = "normal"
        record: EscalationRecord = {
            "ticket_id": ticket_id,
            "reason": reason.strip(),
            "priority": priority,
            "summary": summary.strip(),
            "suggested_team": suggest_team(product_area),
        }
        escalation_id = escalations.submit(record)
        return {"escalation_id": escalation_id, "queued": True, "record": record}

    @mcp.tool()
    def draft_reply(ticket_id: str, reply_text: str, cited_doc_ids: list[str]) -> DraftReplyResult:
        """Record the customer-facing reply for a ticket the KB fully answers.

        Only call this after search_knowledge_base returned hits that
        actually resolve the customer's question. `reply_text` must be the
        final wording sent to the customer: plain text, no markdown, opens
        by acknowledging their specific situation, gives the steps, and
        does not promise anything the cited articles do not say.
        `cited_doc_ids` lists every doc_id whose content the reply relies on.
        """
        text = reply_text.strip()
        return {
            "ticket_id": ticket_id,
            "reply_text": text,
            "citations": list(dict.fromkeys(cited_doc_ids)),
            "word_count": len(text.split()),
        }

    @mcp.tool()
    def request_information(
        ticket_id: str, questions: list[str], reply_text: str
    ) -> InfoRequestResult:
        """Ask the customer for details needed before the ticket can be resolved.

        Use this instead of escalating when the KB likely covers the problem
        but the ticket is missing a fact required to pick the right article or
        step - for example which browser, an error message, a job id, or which
        of two settings is in use. `questions` lists each missing fact as a
        short question. `reply_text` is the customer-facing message that asks
        them. Do not use this to stall on a ticket the KB clearly answers.
        """
        cleaned = [q.strip() for q in questions if q.strip()]
        return {
            "ticket_id": ticket_id,
            "questions": cleaned,
            "reply_text": reply_text.strip(),
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
