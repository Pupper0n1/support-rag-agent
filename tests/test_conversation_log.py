from __future__ import annotations

import json
from pathlib import Path

from support_agent.observability.conversation_log import (
    LocalConversationLog,
    build_record,
)
from support_agent.types import AgentAnswer, RouteDecision, Ticket


def test_record_written_under_date_prefix(tmp_path: Path) -> None:
    ticket = Ticket(ticket_id="T-9", subject="s", body="b")
    answer = AgentAnswer(
        ticket_id="T-9",
        route=RouteDecision.AUTO_REPLY,
        reply_text="ok",
        confidence=0.8,
        tool_calls=["search_knowledge_base", "draft_reply"],
    )
    record = build_record(ticket, answer, model="claude-opus-5", latency_ms=1234, request_id="r")
    assert record["ticket"]["channel"] == "email"
    assert record["route"] == "auto_reply"

    location = LocalConversationLog(tmp_path).record(record)
    path = Path(location)
    assert path.exists()
    assert path.parts[-5] == "conversations"
    assert path.name.startswith("T-9-")
    assert json.loads(path.read_text())["tool_calls"] == ["search_knowledge_base", "draft_reply"]
