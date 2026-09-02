from __future__ import annotations

import json

import pytest

from support_agent.lambda_handler import handler, parse_ticket
from support_agent.types import TicketChannel, TicketPriority


def test_parse_ticket_defaults() -> None:
    t = parse_ticket({"ticket_id": "T-1", "subject": "s", "body": "b"})
    assert t.channel is TicketChannel.EMAIL
    assert t.priority is TicketPriority.NORMAL
    assert t.product_area is None


def test_parse_ticket_missing_fields() -> None:
    with pytest.raises(ValueError, match="subject, body"):
        parse_ticket({"ticket_id": "T-1"})


def test_handler_rejects_bad_json_without_building_runtime() -> None:
    resp = handler({"body": "{not json", "requestContext": {"requestId": "r1"}}, None)
    assert resp["statusCode"] == 400
    assert json.loads(resp["body"])["request_id"] == "r1"


def test_handler_rejects_invalid_enum() -> None:
    body = json.dumps({"ticket_id": "T", "subject": "s", "body": "b", "channel": "fax"})
    resp = handler({"body": body}, None)
    assert resp["statusCode"] == 400
