"""Wire types for MCP tool inputs and outputs.

Kept as TypedDicts (not dataclasses) because they are serialised straight
into tool results and read back by the model; the shape is the contract.
"""

from __future__ import annotations

from typing import TypedDict


class SearchHit(TypedDict):
    doc_id: str
    title: str
    text: str
    source_url: str
    product_area: str
    score: float


class SearchResult(TypedDict):
    query: str
    hits: list[SearchHit]
    top_score: float
    bm25_hits: int
    knn_hits: int
    reranked: bool


class EscalationRecord(TypedDict):
    ticket_id: str
    reason: str
    priority: str
    summary: str
    suggested_team: str


class EscalationResult(TypedDict):
    escalation_id: str
    queued: bool
    record: EscalationRecord


class DraftReplyResult(TypedDict):
    ticket_id: str
    reply_text: str
    citations: list[str]
    word_count: int


class InfoRequestResult(TypedDict):
    ticket_id: str
    questions: list[str]
    reply_text: str
