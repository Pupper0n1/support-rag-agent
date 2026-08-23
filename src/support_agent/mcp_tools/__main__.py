"""Run the MCP server over stdio for local development and MCP inspectors.

    python -m support_agent.mcp_tools
"""

from __future__ import annotations

import logging

from support_agent.config import get_settings
from support_agent.mcp_tools.escalation import InMemoryEscalationSink
from support_agent.mcp_tools.server import build_server
from support_agent.retrieval.embeddings import SentenceTransformerEmbeddings
from support_agent.retrieval.es_client import KnowledgeBaseIndex
from support_agent.retrieval.hybrid import HybridRetriever
from support_agent.retrieval.rerank import CrossEncoderReranker


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    retriever = HybridRetriever(
        index=KnowledgeBaseIndex(settings.elasticsearch),
        embeddings=SentenceTransformerEmbeddings(),
        settings=settings.retrieval,
        reranker=CrossEncoderReranker(),
    )
    build_server(retriever, InMemoryEscalationSink()).run(transport="stdio")


if __name__ == "__main__":
    main()
