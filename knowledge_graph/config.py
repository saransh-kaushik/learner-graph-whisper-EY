"""
knowledge_graph/config.py
─────────────────────────
Typed configuration loaded from environment variables / .env file.

Usage
-----
    from knowledge_graph.config import settings

    print(settings.neo4j_uri)
    print(settings.openai_model)
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = Field("", alias="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o", alias="OPENAI_MODEL")
    openai_embedding_model: str = Field(
        "text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL"
    )

    # ── Neo4j ─────────────────────────────────────────────────────────────────
    neo4j_uri: str = Field("bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field("neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field("", alias="NEO4J_PASSWORD")
    neo4j_database: str = Field("neo4j", alias="NEO4J_DATABASE")

    # ── App ───────────────────────────────────────────────────────────────────
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    cefr_reference_path: Path = Field(
        Path("./knowledge_graph/data/cefr_wordlist.json"),
        alias="CEFR_REFERENCE_PATH",
    )
    rolling_baseline_sessions: int = Field(5, alias="ROLLING_BASELINE_SESSIONS")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Convenience singleton
settings: Settings = get_settings()
