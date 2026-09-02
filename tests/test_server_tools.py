from __future__ import annotations

import math

import pytest

from support_agent.mcp_tools.escalation import (
    DEFAULT_TEAM,
    InMemoryEscalationSink,
    suggest_team,
)
from support_agent.mcp_tools.server import _normalise, build_server
from support_agent.types import RetrievalResult, ScoredChunk
from tests.conftest import make_doc


class StubRetriever:
    def __init__(self, chunks: list[ScoredChunk], reranked: bool) -> None:
        self._chunks = chunks
        self._reranked = reranked
        self.calls: list[tuple[str, str | None]] = []

    def retrieve(self, query: str, product_area: str | None = None) -> RetrievalResult:
        self.calls.append((query, product_area))
        return RetrievalResult(query=query, chunks=self._chunks, reranked=self._reranked)


def test_normalise_reranked_is_sigmoid() -> None:
    assert _normalise(0.0, reranked=True) == pytest.approx(0.5)
    assert _normalise(4.0, reranked=True) == pytest.approx(1 / (1 + math.exp(-4.0)))


def test_normalise_rrf_top_in_both_arms_is_near_one() -> None:
    both_first = 2 / 61
    assert 0.95 <= _normalise(both_first, reranked=False) <= 1.0
    assert _normalise(1 / 110, reranked=False) < 0.3


def test_suggest_team() -> None:
    assert suggest_team("auth") == "security-support"
    assert suggest_team(None) == DEFAULT_TEAM
    assert suggest_team("unknown") == DEFAULT_TEAM


@pytest.mark.asyncio
async def test_tools_round_trip_through_mcp() -> None:
    from mcp.shared.memory import create_connected_server_and_client_session

    chunk = ScoredChunk(document=make_doc("kb-a"), rerank_score=2.0)
    retriever = StubRetriever([chunk], reranked=True)
    sink = InMemoryEscalationSink()
    server = build_server(retriever, sink)  # type: ignore[arg-type]

    async with create_connected_server_and_client_session(server._mcp_server) as session:
        listed = await session.list_tools()
        names = {t.name for t in listed.tools}
        assert names == {
            "search_knowledge_base",
            "escalate_ticket",
            "draft_reply",
            "request_information",
        }

        search = await session.call_tool(
            "search_knowledge_base", {"query": "charged twice", "product_area": "billing"}
        )
        assert search.structuredContent is not None
        assert search.structuredContent["hits"][0]["doc_id"] == "kb-a"
        assert search.structuredContent["top_score"] == pytest.approx(
            1 / (1 + math.exp(-2.0)), abs=1e-3
        )
        assert retriever.calls == [("charged twice", "billing")]

        esc = await session.call_tool(
            "escalate_ticket",
            {
                "ticket_id": "T-1",
                "reason": "refund exception",
                "summary": "Annual plan, 5 months left.",
                "priority": "bogus",
                "product_area": "billing",
            },
        )
        assert esc.structuredContent is not None
        assert esc.structuredContent["record"]["priority"] == "normal"
        assert esc.structuredContent["record"]["suggested_team"] == "billing-specialists"
        assert len(sink.records) == 1

        draft = await session.call_tool(
            "draft_reply",
            {"ticket_id": "T-1", "reply_text": "  Hi there. ", "cited_doc_ids": ["kb-a", "kb-a"]},
        )
        assert draft.structuredContent is not None
        assert draft.structuredContent["citations"] == ["kb-a"]
        assert draft.structuredContent["word_count"] == 2
