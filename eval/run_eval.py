"""Run the evaluation set through retrieval (and optionally the full agent).

    python eval/run_eval.py --name baseline --retrieval-only
    python eval/run_eval.py --name v2 --limit 30

Retrieval-only mode needs Elasticsearch but no model API key and runs in
seconds; use it for retrieval sweeps. Full mode calls the agent per ticket
and additionally scores routing accuracy.

Writes eval/reports/<name>.json with per-ticket rows and aggregate metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

from support_agent.config import get_settings
from support_agent.mcp_tools.escalation import InMemoryEscalationSink
from support_agent.mcp_tools.server import build_server
from support_agent.retrieval.embeddings import SentenceTransformerEmbeddings
from support_agent.retrieval.es_client import KnowledgeBaseIndex
from support_agent.retrieval.hybrid import HybridRetriever
from support_agent.retrieval.rerank import CrossEncoderReranker, PassthroughReranker
from support_agent.types import Ticket, TicketChannel, TicketPriority

from eval.metrics import (
    RoutingConfusion,
    base_doc_id,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    routing_accuracy,
    routing_confusion,
)

DATASET = Path(__file__).parent / "dataset" / "tickets.jsonl"
GRADES = Path(__file__).parent / "dataset" / "human_grades.jsonl"
REPORTS = Path(__file__).parent / "reports"

logger = logging.getLogger("eval")


class EvalTicket(TypedDict):
    ticket_id: str
    subject: str
    body: str
    channel: str
    priority: str
    customer_tier: str
    product_area: str
    relevant_docs: list[str]
    expected_route: str


class HumanGrade(TypedDict):
    ticket_id: str
    doc_id: str
    grade: int


class TicketRow(TypedDict):
    ticket_id: str
    product_area: str
    retrieved: list[str]
    retrieved_scores: list[float]
    relevant: list[str]
    recall_at_3: float
    recall_at_6: float
    mrr: float
    ndcg_at_6: float
    expected_route: str
    predicted_route: str | None
    confidence: float | None
    tool_calls: list[str]
    latency_ms: int


class RetrievalConfig(TypedDict):
    bm25_size: int
    knn_size: int
    rrf_k: int
    rerank_depth: int
    top_k: int
    reranker: str


class Aggregate(TypedDict):
    tickets: int
    recall_at_3: float
    recall_at_6: float
    mrr: float
    ndcg_at_6: float
    graded_pairs: int
    routing_accuracy: float | None
    routing_confusion: list[RoutingConfusion]
    p50_latency_ms: int
    p95_latency_ms: int


class Report(TypedDict):
    name: str
    created_at: str
    mode: str
    model: str | None
    retrieval: RetrievalConfig
    aggregate: Aggregate
    rows: list[TicketRow]


def load_tickets(limit: int | None) -> list[EvalTicket]:
    rows: list[EvalTicket] = []
    with DATASET.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def load_grades() -> dict[str, dict[str, int]]:
    """ticket_id -> {base doc_id -> grade}"""
    out: dict[str, dict[str, int]] = {}
    if not GRADES.exists():
        return out
    with GRADES.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            g: HumanGrade = json.loads(line)
            out.setdefault(g["ticket_id"], {})[base_doc_id(g["doc_id"])] = g["grade"]
    return out


def to_ticket(row: EvalTicket, use_product_area: bool) -> Ticket:
    return Ticket(
        ticket_id=row["ticket_id"],
        subject=row["subject"],
        body=row["body"],
        channel=TicketChannel(row["channel"]),
        priority=TicketPriority(row["priority"]),
        customer_tier=row["customer_tier"],
        product_area=row["product_area"] if use_product_area else None,
    )


def percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))
    return ordered[idx]


async def run(args: argparse.Namespace) -> Report:
    settings = get_settings()
    reranker = PassthroughReranker() if args.no_rerank else CrossEncoderReranker()
    retriever = HybridRetriever(
        index=KnowledgeBaseIndex(settings.elasticsearch),
        embeddings=SentenceTransformerEmbeddings(),
        settings=settings.retrieval,
        reranker=reranker,
    )
    agent = None
    if not args.retrieval_only:
        from support_agent.agent.runner import SupportAgent

        server = build_server(retriever, InMemoryEscalationSink())
        agent = SupportAgent(server, settings.agent)

    grades = load_grades()
    tickets = load_tickets(args.limit)
    rows: list[TicketRow] = []
    route_pairs: list[tuple[str, str]] = []

    for i, raw in enumerate(tickets, start=1):
        ticket = to_ticket(raw, use_product_area=args.use_product_area)
        started = time.perf_counter()

        result = retriever.retrieve(ticket.as_query(), product_area=ticket.product_area)
        retrieved = [c.document.doc_id for c in result.chunks]
        scores = [round(c.final_score, 4) for c in result.chunks]

        predicted: str | None = None
        confidence: float | None = None
        tool_calls: list[str] = []
        if agent is not None:
            answer = await agent.handle(ticket)
            predicted = answer.route.value
            confidence = round(answer.confidence, 4)
            tool_calls = answer.tool_calls
            route_pairs.append((raw["expected_route"], predicted))

        latency_ms = int((time.perf_counter() - started) * 1000)
        ticket_grades = grades.get(raw["ticket_id"], {})
        rows.append(
            {
                "ticket_id": raw["ticket_id"],
                "product_area": raw["product_area"],
                "retrieved": retrieved,
                "retrieved_scores": scores,
                "relevant": raw["relevant_docs"],
                "recall_at_3": recall_at_k(retrieved, raw["relevant_docs"], 3),
                "recall_at_6": recall_at_k(retrieved, raw["relevant_docs"], 6),
                "mrr": reciprocal_rank(retrieved, raw["relevant_docs"]),
                "ndcg_at_6": ndcg_at_k(retrieved, raw["relevant_docs"], 6, ticket_grades),
                "expected_route": raw["expected_route"],
                "predicted_route": predicted,
                "confidence": confidence,
                "tool_calls": tool_calls,
                "latency_ms": latency_ms,
            }
        )
        if i % 10 == 0:
            logger.info("%d/%d tickets", i, len(tickets))

    n = len(rows)
    latencies = [r["latency_ms"] for r in rows]
    aggregate: Aggregate = {
        "tickets": n,
        "recall_at_3": sum(r["recall_at_3"] for r in rows) / n,
        "recall_at_6": sum(r["recall_at_6"] for r in rows) / n,
        "mrr": sum(r["mrr"] for r in rows) / n,
        "ndcg_at_6": sum(r["ndcg_at_6"] for r in rows) / n,
        "graded_pairs": sum(len(g) for g in grades.values()),
        "routing_accuracy": routing_accuracy(route_pairs) if route_pairs else None,
        "routing_confusion": routing_confusion(route_pairs),
        "p50_latency_ms": percentile(latencies, 0.50),
        "p95_latency_ms": percentile(latencies, 0.95),
    }
    rs = settings.retrieval
    return {
        "name": args.name,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "mode": "retrieval_only" if agent is None else "full",
        "model": None if agent is None else settings.agent.agent_model,
        "retrieval": {
            "bm25_size": rs.bm25_size,
            "knn_size": rs.knn_size,
            "rrf_k": rs.rrf_k,
            "rerank_depth": rs.rerank_depth,
            "top_k": rs.top_k,
            "reranker": type(reranker).__name__,
        },
        "aggregate": aggregate,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument(
        "--use-product-area",
        action="store_true",
        help="pass the ticket's product_area as a retrieval filter",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    report = asyncio.run(run(args))
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"{args.name}.json"
    out.write_text(json.dumps(report, indent=2))

    a = report["aggregate"]
    print(f"\n{report['name']}  ({report['mode']}, {a['tickets']} tickets)")
    print(f"  recall@3 {a['recall_at_3']:.3f}   recall@6 {a['recall_at_6']:.3f}")
    print(f"  MRR      {a['mrr']:.3f}   nDCG@6   {a['ndcg_at_6']:.3f}  ({a['graded_pairs']} graded pairs)")
    if a["routing_accuracy"] is not None:
        print(f"  routing  {a['routing_accuracy']:.3f}")
        for c in a["routing_confusion"]:
            if c["expected"] != c["predicted"]:
                print(f"    {c['expected']:>10} -> {c['predicted']:<10} {c['count']}")
    print(f"  latency  p50 {a['p50_latency_ms']}ms  p95 {a['p95_latency_ms']}ms")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
