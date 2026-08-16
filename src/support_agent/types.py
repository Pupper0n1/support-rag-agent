"""Shared domain types.

Every structure that crosses a module boundary is declared here so the
retrieval, tool, and agent layers agree on shapes without importing each
other's internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol, Sequence, TypedDict


class TicketChannel(StrEnum):
    EMAIL = "email"
    CHAT = "chat"
    WEB_FORM = "web_form"


class TicketPriority(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class RouteDecision(StrEnum):
    """Terminal branch the agent picked for a ticket."""

    AUTO_REPLY = "auto_reply"
    ESCALATE = "escalate"
    NEEDS_INFO = "needs_info"


@dataclass(frozen=True, slots=True)
class Ticket:
    ticket_id: str
    subject: str
    body: str
    channel: TicketChannel = TicketChannel.EMAIL
    priority: TicketPriority = TicketPriority.NORMAL
    customer_tier: str = "standard"
    product_area: str | None = None

    def as_query(self) -> str:
        """Flatten the ticket into the text used for retrieval."""
        return f"{self.subject}\n\n{self.body}".strip()


@dataclass(frozen=True, slots=True)
class KBDocument:
    """A chunk of the support knowledge base as stored in Elasticsearch."""

    doc_id: str
    title: str
    text: str
    source_url: str
    product_area: str
    updated_at: datetime


@dataclass(slots=True)
class ScoredChunk:
    """A KB chunk carrying the scores each retrieval stage assigned it."""

    document: KBDocument
    bm25_score: float | None = None
    knn_score: float | None = None
    bm25_rank: int | None = None
    knn_rank: int | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None

    @property
    def final_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.fused_score


@dataclass(slots=True)
class RetrievalResult:
    query: str
    chunks: list[ScoredChunk] = field(default_factory=list)
    bm25_hits: int = 0
    knn_hits: int = 0
    reranked: bool = False

    @property
    def top_score(self) -> float:
        return max((c.final_score for c in self.chunks), default=0.0)


class Citation(TypedDict):
    doc_id: str
    title: str
    source_url: str
    score: float


@dataclass(slots=True)
class AgentAnswer:
    ticket_id: str
    route: RouteDecision
    reply_text: str
    citations: list[Citation] = field(default_factory=list)
    escalation_reason: str | None = None
    confidence: float = 0.0
    tool_calls: list[str] = field(default_factory=list)


class EmbeddingProvider(Protocol):
    """Anything that can turn text into a dense vector of fixed width."""

    @property
    def dimensions(self) -> int: ...

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...


class Reranker(Protocol):
    def rerank(self, query: str, chunks: Sequence[ScoredChunk]) -> list[ScoredChunk]: ...
