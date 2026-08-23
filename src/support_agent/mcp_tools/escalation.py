"""Escalation sink.

Where escalated tickets go is a deployment concern (SQS in prod, a list in
tests), so the tool talks to a Protocol and the sink is injected.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Protocol

from support_agent.mcp_tools.schemas import EscalationRecord

logger = logging.getLogger(__name__)

TEAM_BY_AREA: dict[str, str] = {
    "billing": "billing-specialists",
    "auth": "security-support",
    "api": "developer-support",
    "data": "data-ops",
    "integrations": "integrations-support",
}
DEFAULT_TEAM = "tier-2-general"


class EscalationSink(Protocol):
    def submit(self, record: EscalationRecord) -> str:
        """Persist the escalation and return an id."""
        ...


@dataclass
class InMemoryEscalationSink(EscalationSink):
    records: list[EscalationRecord] = field(default_factory=list)

    def submit(self, record: EscalationRecord) -> str:
        self.records.append(record)
        return f"esc_{uuid.uuid4().hex[:10]}"


class SQSEscalationSink(EscalationSink):
    def __init__(self, queue_url: str, region: str) -> None:
        import boto3

        self._queue_url = queue_url
        self._sqs = boto3.client("sqs", region_name=region)

    def submit(self, record: EscalationRecord) -> str:
        escalation_id = f"esc_{uuid.uuid4().hex[:10]}"
        self._sqs.send_message(
            QueueUrl=self._queue_url,
            MessageBody=json.dumps({"escalation_id": escalation_id, **record}),
            MessageGroupId=record["ticket_id"],
            MessageDeduplicationId=escalation_id,
        )
        logger.info("escalated %s -> %s", record["ticket_id"], record["suggested_team"])
        return escalation_id


def suggest_team(product_area: str | None) -> str:
    return TEAM_BY_AREA.get(product_area or "", DEFAULT_TEAM)
