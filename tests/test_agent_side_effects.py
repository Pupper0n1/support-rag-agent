from __future__ import annotations

import json

from support_agent.agent.runner import SupportAgent
from support_agent.types import AgentAnswer, RouteDecision


def _agent() -> SupportAgent:
    # Bypass __init__: these tests exercise pure state transitions only.
    return SupportAgent.__new__(SupportAgent)


def test_search_then_draft_sets_auto_reply_with_citations() -> None:
    agent = _agent()
    answer = AgentAnswer(ticket_id="T-1", route=RouteDecision.ESCALATE, reply_text="")
    hits: list[dict[str, object]] = []

    search = json.dumps(
        {
            "hits": [
                {"doc_id": "kb-a", "title": "A", "source_url": "u/a", "score": 0.91},
                {"doc_id": "kb-b", "title": "B", "source_url": "u/b", "score": 0.40},
            ],
            "top_score": 0.91,
        }
    )
    agent._apply_side_effects("search_knowledge_base", search, answer, hits)  # type: ignore[arg-type]
    assert answer.confidence == 0.91

    draft = json.dumps({"reply_text": "Here is how.", "citations": ["kb-a"]})
    agent._apply_side_effects("draft_reply", draft, answer, hits)  # type: ignore[arg-type]
    assert answer.route is RouteDecision.AUTO_REPLY
    assert answer.reply_text == "Here is how."
    assert [c["doc_id"] for c in answer.citations] == ["kb-a"]


def test_escalation_sets_reason_and_summary() -> None:
    agent = _agent()
    answer = AgentAnswer(ticket_id="T-1", route=RouteDecision.AUTO_REPLY, reply_text="x")
    payload = json.dumps(
        {"record": {"reason": "gdpr request", "summary": "Erase user data.", "priority": "high"}}
    )
    agent._apply_side_effects("escalate_ticket", payload, answer, [])
    assert answer.route is RouteDecision.ESCALATE
    assert answer.escalation_reason == "gdpr request"
    assert answer.reply_text == "Erase user data."


def test_request_information_sets_needs_info() -> None:
    agent = _agent()
    answer = AgentAnswer(ticket_id="T-1", route=RouteDecision.ESCALATE, reply_text="")
    payload = json.dumps(
        {"questions": ["Which browser?"], "reply_text": "Which browser are you on?"}
    )
    agent._apply_side_effects("request_information", payload, answer, [])
    assert answer.route is RouteDecision.NEEDS_INFO
    assert answer.reply_text == "Which browser are you on?"
    assert answer.escalation_reason is None
