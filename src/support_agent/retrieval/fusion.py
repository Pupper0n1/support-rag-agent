"""Reciprocal rank fusion of the BM25 and kNN result lists.

RRF is used instead of score interpolation because the two arms produce
scores on unrelated scales (BM25 is unbounded, cosine is [-1, 1]) and any
linear blend needs per-query normalisation that is brittle on short queries.
Rank-only fusion sidesteps that entirely.
"""

from __future__ import annotations

from collections.abc import Sequence

from support_agent.types import ScoredChunk


def reciprocal_rank_fusion(
    bm25: Sequence[ScoredChunk],
    knn: Sequence[ScoredChunk],
    k: int = 60,
    bm25_weight: float = 1.0,
    knn_weight: float = 1.0,
) -> list[ScoredChunk]:
    """Merge two ranked lists into one, keyed by document id.

    score(d) = sum over arms of  weight / (k + rank_in_arm(d))

    A document that appears in only one arm keeps that arm's contribution and
    gets nothing from the other, which is the standard RRF behaviour. The
    per-arm scores and ranks on the incoming chunks are preserved on the
    merged chunk so downstream stages (and the eval harness) can still see
    where a hit came from.
    """
    if k < 1:
        raise ValueError("rrf k must be >= 1")

    merged: dict[str, ScoredChunk] = {}

    for chunk in bm25:
        rank = chunk.bm25_rank
        if rank is None:
            raise ValueError(f"bm25 chunk {chunk.document.doc_id} missing rank")
        fused = merged.setdefault(chunk.document.doc_id, ScoredChunk(document=chunk.document))
        fused.bm25_score = chunk.bm25_score
        fused.bm25_rank = rank
        fused.fused_score += bm25_weight / (k + rank)

    for chunk in knn:
        rank = chunk.knn_rank
        if rank is None:
            raise ValueError(f"knn chunk {chunk.document.doc_id} missing rank")
        fused = merged.setdefault(chunk.document.doc_id, ScoredChunk(document=chunk.document))
        fused.knn_score = chunk.knn_score
        fused.knn_rank = rank
        fused.fused_score += knn_weight / (k + rank)

    # Tie-break on doc_id so fusion is deterministic across runs; without it
    # two chunks with identical fused scores could swap order between eval
    # runs and show up as phantom regressions.
    return sorted(merged.values(), key=lambda c: (-c.fused_score, c.document.doc_id))
