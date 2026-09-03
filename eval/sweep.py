"""Retrieval parameter sweep.

    python eval/sweep.py --rrf-k 20 40 60 --rerank-depth 15 25 40 --top-k 5 6 8

Runs the retrieval-only eval once per combination and writes
eval/reports/sweep-<timestamp>.md with one row per combination, sorted by
nDCG@6. The embedding model, reranker, and ES client are built once and
shared across combinations; only the RetrievalSettings differ.
"""

from __future__ import annotations

import argparse
import itertools
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

from eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from eval.run_eval import REPORTS, load_grades, load_tickets, to_ticket
from support_agent.config import RetrievalSettings, get_settings
from support_agent.retrieval.embeddings import SentenceTransformerEmbeddings
from support_agent.retrieval.es_client import KnowledgeBaseIndex
from support_agent.retrieval.hybrid import HybridRetriever
from support_agent.retrieval.rerank import CrossEncoderReranker, PassthroughReranker
from support_agent.types import Reranker

logger = logging.getLogger("sweep")


class SweepRow(TypedDict):
    rrf_k: int
    rerank_depth: int
    top_k: int
    reranker: str
    recall_at_3: float
    recall_at_6: float
    mrr: float
    ndcg_at_6: float


def evaluate(
    index: KnowledgeBaseIndex,
    embeddings: SentenceTransformerEmbeddings,
    reranker: Reranker,
    settings: RetrievalSettings,
    limit: int | None,
) -> SweepRow:
    retriever = HybridRetriever(index, embeddings, settings, reranker)
    grades = load_grades()
    tickets = load_tickets(limit)
    r3 = r6 = mrr = ndcg = 0.0
    for raw in tickets:
        ticket = to_ticket(raw, use_product_area=False)
        result = retriever.retrieve(ticket.as_query())
        ids = [c.document.doc_id for c in result.chunks]
        r3 += recall_at_k(ids, raw["relevant_docs"], 3)
        r6 += recall_at_k(ids, raw["relevant_docs"], 6)
        mrr += reciprocal_rank(ids, raw["relevant_docs"])
        ndcg += ndcg_at_k(ids, raw["relevant_docs"], 6, grades.get(raw["ticket_id"], {}))
    n = len(tickets)
    return {
        "rrf_k": settings.rrf_k,
        "rerank_depth": settings.rerank_depth,
        "top_k": settings.top_k,
        "reranker": type(reranker).__name__,
        "recall_at_3": r3 / n,
        "recall_at_6": r6 / n,
        "mrr": mrr / n,
        "ndcg_at_6": ndcg / n,
    }


def render(rows: list[SweepRow]) -> str:
    header = "| rrf_k | rerank_depth | top_k | reranker | recall@3 | recall@6 | MRR | nDCG@6 |"
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in sorted(rows, key=lambda r: -r["ndcg_at_6"]):
        lines.append(
            f"| {r['rrf_k']} | {r['rerank_depth']} | {r['top_k']} | {r['reranker']} "
            f"| {r['recall_at_3']:.3f} | {r['recall_at_6']:.3f} | {r['mrr']:.3f} "
            f"| {r['ndcg_at_6']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rrf-k", type=int, nargs="+", default=[60])
    parser.add_argument("--rerank-depth", type=int, nargs="+", default=[25])
    parser.add_argument("--top-k", type=int, nargs="+", default=[6])
    parser.add_argument("--include-no-rerank", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    base = get_settings()
    index = KnowledgeBaseIndex(base.elasticsearch)
    embeddings = SentenceTransformerEmbeddings()
    rerankers: list[Reranker] = [CrossEncoderReranker()]
    if args.include_no_rerank:
        rerankers.append(PassthroughReranker())

    rows: list[SweepRow] = []
    combos = list(itertools.product(args.rrf_k, args.rerank_depth, args.top_k, rerankers))
    for i, (rrf_k, depth, top_k, reranker) in enumerate(combos, start=1):
        settings = base.retrieval.model_copy(
            update={"rrf_k": rrf_k, "rerank_depth": depth, "top_k": top_k}
        )
        row = evaluate(index, embeddings, reranker, settings, args.limit)
        rows.append(row)
        logger.info(
            "%d/%d k=%d depth=%d top=%d %s nDCG@6=%.3f",
            i,
            len(combos),
            rrf_k,
            depth,
            top_k,
            row["reranker"],
            row["ndcg_at_6"],
        )

    REPORTS.mkdir(exist_ok=True)
    out: Path = REPORTS / f"sweep-{datetime.now(UTC):%Y%m%d-%H%M%S}.md"
    out.write_text(render(rows))
    print(render(rows))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
