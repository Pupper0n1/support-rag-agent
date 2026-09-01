"""Thin wrapper around the Elasticsearch client.

The wrapper exists so the rest of the codebase never touches raw ES response
dicts - everything comes back as ScoredChunk.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import TypedDict

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from support_agent.config import ElasticsearchSettings
from support_agent.retrieval.index_mapping import build_kb_mapping
from support_agent.types import KBDocument, ScoredChunk

logger = logging.getLogger(__name__)


class _SourceDoc(TypedDict):
    doc_id: str
    title: str
    text: str
    source_url: str
    product_area: str
    updated_at: str
    embedding: list[float]


class _Hit(TypedDict):
    _id: str
    _score: float
    _source: _SourceDoc


def _to_document(source: _SourceDoc) -> KBDocument:
    return KBDocument(
        doc_id=source["doc_id"],
        title=source["title"],
        text=source["text"],
        source_url=source["source_url"],
        product_area=source["product_area"],
        updated_at=datetime.fromisoformat(source["updated_at"]),
    )


class KnowledgeBaseIndex:
    def __init__(
        self, settings: ElasticsearchSettings, client: Elasticsearch | None = None
    ) -> None:
        self._settings = settings
        self._client = client or self._build_client(settings)

    @staticmethod
    def _build_client(settings: ElasticsearchSettings) -> Elasticsearch:
        api_key = settings.api_key.get_secret_value() if settings.api_key else None
        return Elasticsearch(
            settings.url,
            api_key=api_key,
            request_timeout=settings.request_timeout_s,
        )

    @property
    def index_name(self) -> str:
        return self._settings.kb_index

    def ensure_index(self, embedding_dims: int) -> bool:
        """Create the index if absent. Returns True when it was created."""
        if self._client.indices.exists(index=self.index_name):
            return False
        mapping = build_kb_mapping(embedding_dims)
        self._client.indices.create(
            index=self.index_name,
            mappings=mapping["mappings"],
            settings=mapping["settings"],
        )
        logger.info("created index %s (dims=%d)", self.index_name, embedding_dims)
        return True

    def index_documents(
        self, documents: Sequence[KBDocument], embeddings: Sequence[list[float]]
    ) -> int:
        if len(documents) != len(embeddings):
            raise ValueError(
                f"documents/embeddings length mismatch: {len(documents)} vs {len(embeddings)}"
            )

        def actions() -> Iterable[dict[str, object]]:
            for doc, vector in zip(documents, embeddings, strict=True):
                yield {
                    "_op_type": "index",
                    "_index": self.index_name,
                    "_id": doc.doc_id,
                    "_source": {
                        "doc_id": doc.doc_id,
                        "title": doc.title,
                        "text": doc.text,
                        "source_url": doc.source_url,
                        "product_area": doc.product_area,
                        "updated_at": doc.updated_at.isoformat(),
                        "embedding": vector,
                    },
                }

        indexed, _errors = bulk(self._client, actions(), refresh="wait_for")
        return int(indexed)

    def bm25_search(
        self, query: str, size: int, product_area: str | None = None
    ) -> list[ScoredChunk]:
        must: list[dict[str, object]] = [
            {"multi_match": {"query": query, "fields": ["title^2", "text"], "type": "best_fields"}}
        ]
        query_body: dict[str, object] = {"bool": {"must": must}}
        if product_area:
            query_body["bool"]["filter"] = [{"term": {"product_area": product_area}}]  # type: ignore[index]

        response = self._client.search(index=self.index_name, query=query_body, size=size)
        hits: list[_Hit] = response["hits"]["hits"]
        return [
            ScoredChunk(
                document=_to_document(hit["_source"]),
                bm25_score=hit["_score"],
                bm25_rank=rank,
            )
            for rank, hit in enumerate(hits, start=1)
        ]

    def knn_search(
        self,
        vector: list[float],
        size: int,
        num_candidates: int,
        product_area: str | None = None,
    ) -> list[ScoredChunk]:
        knn: dict[str, object] = {
            "field": "embedding",
            "query_vector": vector,
            "k": size,
            "num_candidates": num_candidates,
        }
        if product_area:
            knn["filter"] = {"term": {"product_area": product_area}}

        response = self._client.search(index=self.index_name, knn=knn, size=size)
        hits: list[_Hit] = response["hits"]["hits"]
        return [
            ScoredChunk(
                document=_to_document(hit["_source"]),
                knn_score=hit["_score"],
                knn_rank=rank,
            )
            for rank, hit in enumerate(hits, start=1)
        ]
