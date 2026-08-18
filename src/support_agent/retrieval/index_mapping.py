"""Elasticsearch mapping for the support knowledge base index.

One index serves both retrieval arms: `text` is analysed for BM25 and
`embedding` is a dense_vector for kNN. Keeping them on the same document
means a fused hit always resolves to a single chunk, so the two arms can be
joined by document id rather than by re-fetching.
"""

from __future__ import annotations

from typing import TypedDict


class IndexMapping(TypedDict):
    mappings: dict[str, object]
    settings: dict[str, object]


def build_kb_mapping(embedding_dims: int) -> IndexMapping:
    return {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 1,
            },
            "analysis": {
                "analyzer": {
                    "kb_text": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "asciifolding", "kb_stopwords", "porter_stem"],
                    }
                },
                "filter": {
                    "kb_stopwords": {"type": "stop", "stopwords": "_english_"},
                },
            },
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "doc_id": {"type": "keyword"},
                "title": {
                    "type": "text",
                    "analyzer": "kb_text",
                    # Titles carry a lot of signal for short "how do I X" tickets.
                    "boost": 2.0,
                },
                "text": {"type": "text", "analyzer": "kb_text"},
                "source_url": {"type": "keyword", "index": False},
                "product_area": {"type": "keyword"},
                "updated_at": {"type": "date"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": embedding_dims,
                    "index": True,
                    "similarity": "cosine",
                },
            },
        },
    }
