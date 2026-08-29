"""Persist one record per handled ticket for offline evaluation.

Records are the raw material for the next prompt iteration: the eval
harness can replay them, and graders can pull real tickets into the graded
set. Layout is s3://<bucket>/<prefix>/YYYY/MM/DD/<ticket_id>-<ts>.json so a
day's traffic is one prefix listing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, TypedDict

from support_agent.types import AgentAnswer, Citation, Ticket

logger = logging.getLogger(__name__)


class ConversationRecord(TypedDict):
    schema_version: int
    recorded_at: str
    model: str
    ticket: dict[str, str | None]
    route: str
    reply_text: str
    citations: list[Citation]
    escalation_reason: str | None
    confidence: float
    tool_calls: list[str]
    latency_ms: int
    request_id: str | None


class ConversationLog(Protocol):
    def record(self, entry: ConversationRecord) -> str:
        """Persist the record and return its location."""
        ...


def build_record(
    ticket: Ticket,
    answer: AgentAnswer,
    model: str,
    latency_ms: int,
    request_id: str | None = None,
) -> ConversationRecord:
    ticket_fields = {k: (v.value if hasattr(v, "value") else v) for k, v in asdict(ticket).items()}
    return {
        "schema_version": 1,
        "recorded_at": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "model": model,
        "ticket": ticket_fields,
        "route": answer.route.value,
        "reply_text": answer.reply_text,
        "citations": answer.citations,
        "escalation_reason": answer.escalation_reason,
        "confidence": answer.confidence,
        "tool_calls": answer.tool_calls,
        "latency_ms": latency_ms,
        "request_id": request_id,
    }


def _object_key(prefix: str, entry: ConversationRecord) -> str:
    ts = datetime.fromisoformat(entry["recorded_at"])
    ticket_id = entry["ticket"]["ticket_id"]
    return f"{prefix}/{ts:%Y/%m/%d}/{ticket_id}-{ts:%H%M%S%f}.json"


class S3ConversationLog(ConversationLog):
    def __init__(self, bucket: str, prefix: str, region: str) -> None:
        import boto3

        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._s3 = boto3.client("s3", region_name=region)

    def record(self, entry: ConversationRecord) -> str:
        key = _object_key(self._prefix, entry)
        self._s3.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=json.dumps(entry).encode("utf-8"),
            ContentType="application/json",
        )
        logger.info("logged conversation s3://%s/%s", self._bucket, key)
        return f"s3://{self._bucket}/{key}"


class LocalConversationLog(ConversationLog):
    """Same layout on local disk, for development and the eval harness."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def record(self, entry: ConversationRecord) -> str:
        path = self._root / _object_key("conversations", entry)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entry, indent=2))
        return str(path)
