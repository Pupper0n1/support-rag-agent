from __future__ import annotations

import pytest

from eval.metrics import (
    base_doc_id,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    routing_accuracy,
    routing_confusion,
)


def test_base_doc_id_strips_chunk_suffix() -> None:
    assert base_doc_id("kb-x#3") == "kb-x"
    assert base_doc_id("kb-x") == "kb-x"


def test_recall_counts_any_chunk_of_relevant_article() -> None:
    assert recall_at_k(["kb-a#1", "kb-z", "kb-b#0"], ["kb-a", "kb-b"], k=3) == 1.0
    assert recall_at_k(["kb-a#1", "kb-z", "kb-b#0"], ["kb-a", "kb-b"], k=2) == 0.5
    assert recall_at_k([], ["kb-a"], k=3) == 0.0
    assert recall_at_k(["kb-a"], [], k=3) == 0.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["x", "y", "kb-a"], ["kb-a"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x"], ["kb-a"]) == 0.0


def test_ndcg_perfect_binary() -> None:
    assert ndcg_at_k(["a", "b"], ["a", "b"], k=6) == pytest.approx(1.0)


def test_ndcg_worse_when_relevant_doc_is_lower() -> None:
    top = ndcg_at_k(["a", "x", "y"], ["a"], k=3)
    low = ndcg_at_k(["x", "y", "a"], ["a"], k=3)
    assert top > low > 0


def test_ndcg_uses_human_grades_and_stays_bounded() -> None:
    # Grader says the gold doc only partially answers, and a non-gold doc answers.
    grades = {"a": 2, "x": 3}
    score = ndcg_at_k(["x", "a"], ["a"], k=6, grades=grades)
    assert 0 < score <= 1.0
    assert score == pytest.approx(1.0)  # ideal order is exactly x, a


def test_ndcg_dedupes_chunks_of_same_article() -> None:
    assert ndcg_at_k(["a#0", "a#1", "b"], ["a", "b"], k=2) == ndcg_at_k(["a", "b"], ["a", "b"], k=2)


def test_routing_metrics() -> None:
    pairs = [("auto_reply", "auto_reply"), ("escalate", "auto_reply"), ("escalate", "escalate")]
    assert routing_accuracy(pairs) == pytest.approx(2 / 3)
    confusion = routing_confusion(pairs)
    assert confusion[0]["count"] == 1
    assert {(c["expected"], c["predicted"]) for c in confusion} == {
        ("auto_reply", "auto_reply"),
        ("escalate", "auto_reply"),
        ("escalate", "escalate"),
    }
    assert routing_accuracy([]) == 0.0
