"""Ingest the knowledge base into Elasticsearch.

    python scripts/ingest_kb.py data/kb/knowledge_base.jsonl [--recreate]

Each line is one KB article. Articles longer than CHUNK_CHARS are split on
paragraph boundaries into overlapping chunks so a single long article cannot
dominate the kNN arm with one oversized vector.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from support_agent.config import get_settings
from support_agent.retrieval.embeddings import SentenceTransformerEmbeddings
from support_agent.retrieval.es_client import KnowledgeBaseIndex
from support_agent.types import KBDocument

CHUNK_CHARS = 1200
CHUNK_OVERLAP = 150

logger = logging.getLogger("ingest")


class RawArticle(TypedDict):
    doc_id: str
    title: str
    product_area: str
    text: str
    source_url: str
    updated_at: str


def _chunk(text: str) -> list[str]:
    if len(text) <= CHUNK_CHARS:
        return [text]
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 > CHUNK_CHARS and current:
            chunks.append(current)
            current = current[-CHUNK_OVERLAP:] + "\n\n" + para
        else:
            current = f"{current}\n\n{para}".strip()
    if current:
        chunks.append(current)
    return chunks


def load_articles(path: Path) -> list[KBDocument]:
    documents: list[KBDocument] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            raw: RawArticle = json.loads(line)
            pieces = _chunk(raw["text"])
            for n, piece in enumerate(pieces):
                doc_id = raw["doc_id"] if len(pieces) == 1 else f"{raw['doc_id']}#{n}"
                documents.append(
                    KBDocument(
                        doc_id=doc_id,
                        title=raw["title"],
                        text=piece,
                        source_url=raw["source_url"],
                        product_area=raw["product_area"],
                        updated_at=datetime.fromisoformat(raw["updated_at"]),
                    )
                )
    return documents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--recreate", action="store_true", help="drop and rebuild the index")
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    embeddings = SentenceTransformerEmbeddings()
    index = KnowledgeBaseIndex(settings.elasticsearch)

    if args.recreate:
        index._client.indices.delete(index=index.index_name, ignore_unavailable=True)
    index.ensure_index(embeddings.dimensions)

    documents = load_articles(args.source)
    logger.info("loaded %d chunks from %s", len(documents), args.source)

    total = 0
    for start in range(0, len(documents), args.batch):
        batch = documents[start : start + args.batch]
        vectors = embeddings.embed_documents([f"{d.title}\n{d.text}" for d in batch])
        total += index.index_documents(batch, vectors)
        logger.info("indexed %d/%d", total, len(documents))


if __name__ == "__main__":
    main()
