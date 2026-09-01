"""Agent loop: Claude via LangChain, tools via an in-process MCP session.

The MCP server runs in the same process over the SDK's in-memory transport.
That keeps one deployable unit for Lambda while still exercising the real
MCP request/response path, so the same server can be pointed at from an
external MCP client (or the inspector) without code changes.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session

from support_agent.agent.prompts import system_prompt, ticket_message
from support_agent.config import AgentSettings
from support_agent.types import AgentAnswer, Citation, RouteDecision, Ticket

logger = logging.getLogger(__name__)


class _ToolCall(TypedDict):
    name: str
    args: dict[str, object]
    id: str


class _SearchHitPayload(TypedDict):
    doc_id: str
    title: str
    source_url: str
    score: float


class _SearchPayload(TypedDict):
    hits: list[_SearchHitPayload]
    top_score: float


class _DraftPayload(TypedDict):
    reply_text: str
    citations: list[str]


class _EscalationPayload(TypedDict):
    record: dict[str, str]


class _InfoRequestPayload(TypedDict):
    questions: list[str]
    reply_text: str


class SupportAgent:
    def __init__(self, server: FastMCP, settings: AgentSettings) -> None:
        self._server = server
        self._settings = settings
        # langchain-anthropic exposes its fields under pydantic aliases; the
        # alias names are the ones the type checker sees, and it also insists
        # on timeout/stop being passed. With no key given the client falls
        # back to ANTHROPIC_API_KEY in the environment.
        if settings.anthropic_api_key is not None:
            self._llm = ChatAnthropic(
                model_name=settings.agent_model,
                max_tokens_to_sample=settings.max_tokens,
                api_key=settings.anthropic_api_key,
                timeout=None,
                stop=None,
            )
        else:
            self._llm = ChatAnthropic(
                model_name=settings.agent_model,
                max_tokens_to_sample=settings.max_tokens,
                timeout=None,
                stop=None,
            )

    async def handle(self, ticket: Ticket) -> AgentAnswer:
        async with create_connected_server_and_client_session(self._server._mcp_server) as session:
            tools = await load_mcp_tools(session)
            return await self._run(ticket, tools)

    async def _run(self, ticket: Ticket, tools: Sequence[BaseTool]) -> AgentAnswer:
        by_name = {t.name: t for t in tools}
        llm = self._llm.bind_tools(list(tools))

        messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt(self._settings.escalation_confidence_floor)),
            HumanMessage(content=ticket_message(ticket)),
        ]
        answer = AgentAnswer(
            ticket_id=ticket.ticket_id, route=RouteDecision.ESCALATE, reply_text=""
        )
        latest_hits: list[_SearchHitPayload] = []

        for _ in range(self._settings.max_tool_iterations):
            response = await llm.ainvoke(messages)
            if not isinstance(response, AIMessage):
                raise TypeError(f"unexpected message type {type(response).__name__}")
            messages.append(response)

            calls: list[_ToolCall] = [
                {"name": c["name"], "args": dict(c["args"]), "id": c["id"] or ""}
                for c in response.tool_calls
            ]
            if not calls:
                break

            for call in calls:
                tool = by_name.get(call["name"])
                if tool is None:
                    messages.append(
                        ToolMessage(
                            content=f"unknown tool {call['name']}",
                            tool_call_id=call["id"],
                            status="error",
                        )
                    )
                    continue

                raw = await tool.ainvoke(call["args"])
                payload = _coerce_payload(raw)
                answer.tool_calls.append(call["name"])
                messages.append(ToolMessage(content=payload, tool_call_id=call["id"]))
                self._apply_side_effects(call["name"], payload, answer, latest_hits)

        if not answer.tool_calls:
            answer.escalation_reason = "agent produced no tool calls"
        return answer

    def _apply_side_effects(
        self,
        tool_name: str,
        payload: str,
        answer: AgentAnswer,
        latest_hits: list[_SearchHitPayload],
    ) -> None:
        if tool_name == "search_knowledge_base":
            search: _SearchPayload = json.loads(payload)
            latest_hits[:] = search["hits"]
            answer.confidence = float(search["top_score"])
        elif tool_name == "draft_reply":
            draft: _DraftPayload = json.loads(payload)
            answer.route = RouteDecision.AUTO_REPLY
            answer.reply_text = draft["reply_text"]
            cited = set(draft["citations"])
            answer.citations = [_citation(h) for h in latest_hits if h["doc_id"] in cited]
        elif tool_name == "escalate_ticket":
            escalation: _EscalationPayload = json.loads(payload)
            answer.route = RouteDecision.ESCALATE
            answer.escalation_reason = escalation["record"]["reason"]
            answer.reply_text = escalation["record"]["summary"]
        elif tool_name == "request_information":
            info: _InfoRequestPayload = json.loads(payload)
            answer.route = RouteDecision.NEEDS_INFO
            answer.reply_text = info["reply_text"]
            answer.escalation_reason = None


def _citation(hit: _SearchHitPayload) -> Citation:
    return {
        "doc_id": hit["doc_id"],
        "title": hit["title"],
        "source_url": hit["source_url"],
        "score": hit["score"],
    }


def _coerce_payload(raw: object) -> str:
    """MCP adapter returns either a JSON string or a list of content blocks."""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        texts: list[str] = []
        for block in raw:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(str(block["text"]))
            elif isinstance(block, str):
                texts.append(block)
        return "\n".join(texts)
    return json.dumps(raw)
