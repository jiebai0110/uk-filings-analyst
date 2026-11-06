"""Central configuration, read from environment variables / .env."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    chat_model: str = "anthropic/claude-sonnet-5-5"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    database_url: str = "postgresql://ufa:ufa@localhost:5432/ufa"

    chunk_tokens: int = 512
    chunk_overlap: int = 64
    top_k: int = 6


@lru_cache
def get_settings() -> Settings:
    return Settings()
