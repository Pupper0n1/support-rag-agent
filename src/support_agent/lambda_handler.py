"""AWS Lambda entrypoint behind API Gateway (HTTP API, payload v2).

    POST /tickets   body: {ticket_id, subject, body, channel?, priority?,
                           customer_tier?, product_area?}

Heavy objects (embedding model, reranker, ES client, MCP server) are built
once per container via build_runtime() and reused across warm invocations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import TypedDict

from support_agent.agent.runner import SupportAgent
from support_agent.config import Settings, get_settings
from support_agent.mcp_tools.escalation import EscalationSink, InMemoryEscalationSink, SQSEscalationSink
from support_agent.mcp_tools.server import build_server
from support_agent.observability.conversation_log import (
    ConversationLog,
    S3ConversationLog,
    build_record,
)
from support_agent.retrieval.embeddings import SentenceTransformerEmbeddings
from support_agent.retrieval.es_client import KnowledgeBaseIndex
from support_agent.retrieval.hybrid import HybridRetriever
from support_agent.retrieval.rerank import CrossEncoderReranker
from support_agent.secrets import hydrate_secrets
from support_agent.types import Ticket, TicketChannel, TicketPriority

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class TicketRequest(TypedDict, total=False):
    ticket_id: str
    subject: str
    body: str
    channel: str
    priority: str
    customer_tier: str
    product_area: str


class RequestContext(TypedDict, total=False):
    requestId: str


class HttpApiEvent(TypedDict, total=False):
    body: str
    isBase64Encoded: bool
    requestContext: RequestContext


class HttpResponse(TypedDict):
    statusCode: int
    headers: dict[str, str]
    body: str


@dataclass(frozen=True)
class Runtime:
    settings: Settings
    agent: SupportAgent
    escalations: EscalationSink
    conversation_log: ConversationLog


@lru_cache(maxsize=1)
def build_runtime() -> Runtime:
    hydrate_secrets(os.environ.get("AWS_REGION", "us-east-1"))
    settings = get_settings()
    retriever = HybridRetriever(
        index=KnowledgeBaseIndex(settings.elasticsearch),
        embeddings=SentenceTransformerEmbeddings(),
        settings=settings.retrieval,
        reranker=CrossEncoderReranker(),
    )
    queue_url = os.environ.get("ESCALATION_QUEUE_URL")
    escalations: EscalationSink = (
        SQSEscalationSink(queue_url, settings.aws.aws_region)
        if queue_url
        else InMemoryEscalationSink()
    )
    server = build_server(retriever, escalations)
    log = S3ConversationLog(
        bucket=settings.aws.conversation_log_bucket,
        prefix=settings.aws.conversation_log_prefix,
        region=settings.aws.aws_region,
    )
    return Runtime(
        settings=settings,
        agent=SupportAgent(server, settings.agent),
        escalations=escalations,
        conversation_log=log,
    )


def parse_ticket(payload: TicketRequest) -> Ticket:
    missing = [k for k in ("ticket_id", "subject", "body") if not payload.get(k)]
    if missing:
        raise ValueError(f"missing required field(s): {', '.join(missing)}")
    return Ticket(
        ticket_id=str(payload["ticket_id"]),
        subject=str(payload["subject"]),
        body=str(payload["body"]),
        channel=TicketChannel(payload.get("channel", "email")),
        priority=TicketPriority(payload.get("priority", "normal")),
        customer_tier=str(payload.get("customer_tier", "standard")),
        product_area=payload.get("product_area"),
    )


def _response(status: int, body: dict[str, object]) -> HttpResponse:
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json"},
        "body": json.dumps(body),
    }


def handler(event: HttpApiEvent, _context: object) -> HttpResponse:
    request_id = event.get("requestContext", {}).get("requestId")
    raw_body = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        import base64

        raw_body = base64.b64decode(raw_body).decode("utf-8")

    try:
        payload: TicketRequest = json.loads(raw_body)
        ticket = parse_ticket(payload)
    except (ValueError, TypeError) as exc:
        return _response(400, {"error": str(exc), "request_id": request_id})

    runtime = build_runtime()
    started = time.perf_counter()
    try:
        answer = asyncio.run(runtime.agent.handle(ticket))
    except Exception:
        logger.exception("agent failure ticket=%s request=%s", ticket.ticket_id, request_id)
        return _response(502, {"error": "agent failure", "request_id": request_id})
    latency_ms = int((time.perf_counter() - started) * 1000)

    record = build_record(
        ticket, answer, runtime.settings.agent.agent_model, latency_ms, request_id
    )
    try:
        location = runtime.conversation_log.record(record)
    except Exception:
        # Logging must never fail the customer-facing response.
        logger.exception("conversation log write failed ticket=%s", ticket.ticket_id)
        location = None

    return _response(
        200,
        {
            "ticket_id": answer.ticket_id,
            "route": answer.route.value,
            "reply_text": answer.reply_text,
            "citations": answer.citations,
            "escalation_reason": answer.escalation_reason,
            "confidence": answer.confidence,
            "tool_calls": answer.tool_calls,
            "latency_ms": latency_ms,
            "log_location": location,
            "request_id": request_id,
        },
    )
