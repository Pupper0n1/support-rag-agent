"""Retrieval and routing metrics.

All functions are pure so they can be unit-tested without the pipeline.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TypedDict


class RoutingConfusion(TypedDict):
    expected: str
    predicted: str
    count: int


def base_doc_id(doc_id: str) -> str:
    """Strip the chunk suffix so any chunk of an article counts as the article."""
    return doc_id.split("#", 1)[0]


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    head = {base_doc_id(d) for d in retrieved[:k]}
    hit = sum(1 for r in relevant if r in head)
    return hit / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    wanted = set(relevant)
    for rank, doc in enumerate(retrieved, start=1):
        if base_doc_id(doc) in wanted:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved: Sequence[str],
    relevant: Sequence[str],
    k: int,
    grades: Mapping[str, int] | None = None,
) -> float:
    """nDCG@k using human grades where present, binary gold labels otherwise.

    `grades` maps base doc_id -> 0..3. An ungraded doc in the gold set
    scores 3 (the grader's 'answers' level); an ungraded doc outside it
    scores 0. Mixing scales this way is deliberate: it lets partial grading
    of the surfaced results sharpen the metric without requiring every
    (ticket, doc) pair to be graded first.
    """
    grades = grades or {}
    gold = set(relevant)

    def gain(doc: str) -> int:
        base = base_doc_id(doc)
        if base in grades:
            return grades[base]
        return 3 if base in gold else 0

    dcg = 0.0
    seen: set[str] = set()
    position = 0
    for doc in retrieved:
        base = base_doc_id(doc)
        if base in seen:
            continue
        seen.add(base)
        position += 1
        if position > k:
            break
        dcg += gain(doc) / math.log2(position + 1)

    ideal_gains = sorted(
        (grades.get(d, 3) for d in gold) if gold else [], reverse=True
    )
    # Graded docs outside the gold set with a positive grade also belong in
    # the ideal ordering; otherwise a grader upgrading a non-gold doc would
    # make nDCG exceed 1.
    extra = sorted((g for d, g in grades.items() if d not in gold and g > 0), reverse=True)
    ideal = sorted(ideal_gains + extra, reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def routing_accuracy(pairs: Sequence[tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    return sum(1 for exp, pred in pairs if exp == pred) / len(pairs)


def routing_confusion(pairs: Sequence[tuple[str, str]]) -> list[RoutingConfusion]:
    counts: dict[tuple[str, str], int] = {}
    for exp, pred in pairs:
        counts[(exp, pred)] = counts.get((exp, pred), 0) + 1
    return [
        {"expected": e, "predicted": p, "count": c}
        for (e, p), c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
