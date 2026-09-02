from __future__ import annotations

import pytest

from support_agent.retrieval.fusion import reciprocal_rank_fusion
from support_agent.types import ScoredChunk
from tests.conftest import bm25_list, knn_list, make_doc


def test_doc_in_both_arms_outranks_doc_in_one() -> None:
    fused = reciprocal_rank_fusion(bm25_list("a", "b"), knn_list("b", "c"), k=60)
    assert [c.document.doc_id for c in fused][0] == "b"
    b = fused[0]
    assert b.bm25_rank == 2 and b.knn_rank == 1
    assert b.fused_score == pytest.approx(1 / 62 + 1 / 61)


def test_single_arm_scores_are_preserved() -> None:
    fused = reciprocal_rank_fusion(bm25_list("a"), knn_list("c"), k=60)
    by_id = {c.document.doc_id: c for c in fused}
    assert by_id["a"].bm25_score == 10.0 and by_id["a"].knn_score is None
    assert by_id["c"].knn_score == pytest.approx(0.9) and by_id["c"].bm25_score is None


def test_tie_breaks_on_doc_id_for_determinism() -> None:
    fused = reciprocal_rank_fusion(bm25_list("z"), knn_list("m"), k=60)
    assert [c.document.doc_id for c in fused] == ["m", "z"]


def test_weights_shift_ranking() -> None:
    bm25 = bm25_list("a", "b")
    knn = knn_list("b", "a")
    neutral = reciprocal_rank_fusion(bm25, knn, k=60)
    assert neutral[0].fused_score == pytest.approx(neutral[1].fused_score)
    knn_heavy = reciprocal_rank_fusion(bm25, knn, k=60, knn_weight=2.0)
    assert knn_heavy[0].document.doc_id == "b"


def test_missing_rank_is_an_error() -> None:
    broken = [ScoredChunk(document=make_doc("a"), bm25_score=1.0)]
    with pytest.raises(ValueError, match="missing rank"):
        reciprocal_rank_fusion(broken, [], k=60)


def test_invalid_k() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], [], k=0)


def test_empty_inputs() -> None:
    assert reciprocal_rank_fusion([], [], k=60) == []
