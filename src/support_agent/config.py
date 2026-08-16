"""Typed configuration loaded from the environment."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrievalSettings(BaseSettings):
    """Knobs for the hybrid retrieval pipeline.

    Defaults are the starting point; the values that ship were picked from the
    eval sweep in eval/reports.
    """

    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_", env_file=".env", extra="ignore")

    bm25_size: int = Field(default=50, ge=1, le=500)
    knn_size: int = Field(default=50, ge=1, le=500)
    knn_num_candidates: int = Field(default=200, ge=1, le=10_000)
    rrf_k: int = Field(default=60, ge=1)
    rerank_depth: int = Field(default=25, ge=1)
    top_k: int = Field(default=6, ge=1, le=50)


class ElasticsearchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ES_", env_file=".env", extra="ignore")

    url: str = "https://localhost:9200"
    api_key: SecretStr | None = None
    kb_index: str = "support-kb"
    request_timeout_s: float = 10.0


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: SecretStr | None = None
    agent_model: str = "claude-opus-5"
    max_tokens: int = 4096
    escalation_confidence_floor: float = Field(default=0.45, ge=0.0, le=1.0)
    max_tool_iterations: int = Field(default=8, ge=1, le=32)


class AWSSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    aws_region: str = "us-east-1"
    conversation_log_bucket: str = "support-agent-conversation-logs"
    conversation_log_prefix: str = "conversations"


class Settings(BaseSettings):
    """Aggregate settings object handed to every constructor."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    elasticsearch: ElasticsearchSettings = Field(default_factory=ElasticsearchSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    aws: AWSSettings = Field(default_factory=AWSSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, cached so a warm Lambda parses the env once."""
    return Settings()
