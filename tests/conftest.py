from __future__ import annotations

from datetime import UTC, datetime

import pytest

from support_agent.types import KBDocument, ScoredChunk


def make_doc(doc_id: str, area: str = "billing") -> KBDocument:
    return KBDocument(
        doc_id=doc_id,
        title=f"Title {doc_id}",
        text=f"Body text for {doc_id}.",
        source_url=f"https://help.example.com/{doc_id}",
        product_area=area,
        updated_at=datetime(2026, 8, 1, tzinfo=UTC),
    )


def bm25_list(*doc_ids: str) -> list[ScoredChunk]:
    return [
        ScoredChunk(document=make_doc(d), bm25_score=10.0 - i, bm25_rank=i + 1)
        for i, d in enumerate(doc_ids)
    ]


def knn_list(*doc_ids: str) -> list[ScoredChunk]:
    return [
        ScoredChunk(document=make_doc(d), knn_score=0.9 - i * 0.05, knn_rank=i + 1)
        for i, d in enumerate(doc_ids)
    ]


@pytest.fixture
def docs() -> dict[str, KBDocument]:
    return {d: make_doc(d) for d in ("a", "b", "c", "d", "e")}
