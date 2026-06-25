"""
app/core/config.py
──────────────────
Centralised settings loaded from environment variables / .env file.
Uses pydantic-settings so every field is typed and validated at startup.

Resolution order:
  1. Real environment variables
  2. .env file (auto-loaded)
  3. Field defaults

To add a new LLM provider:
  1. Add its API key + model fields here
  2. Create the handler in app/chain/
  3. Register it in app/chain/registry.py
  4. Set LLM_PROVIDER=<name> in .env  ← only change needed at runtime
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    app_title: str = "Agentic Travel Planner"
    app_description: str = (
        "A production-ready AI Travel Planning API — Phase 1: LangChain chains. "
        "Describe your trip and get a detailed itinerary powered by Gemini."
    )
    app_version: str = "1.0.0"

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    log_level: str = "info"

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_origins: str = "*"  # comma-separated list or wildcard

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    # ── LLM Provider ──────────────────────────────────────────────────────────
    # Options: "gemini" | "openai"
    # To switch providers: change this in .env — no code changes needed.
    llm_provider: str = Field(default="gemini", description="gemini | openai")

    @field_validator("llm_provider", mode="before")
    @classmethod
    def normalise_llm_provider(cls, v: str) -> str:
        return v.strip().lower()

    # ── Google Gemini ─────────────────────────────────────────────────────────
    gemini_api_key: str = ""   # never hardcode — set in .env
    gemini_model: str = "gemini-2.5-flash-lite"

    # ── OpenAI (Phase 2 ready — stub for future) ──────────────────────────────
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # ── Chain / Generation ────────────────────────────────────────────────────
    temperature: float = 0.4   # slightly higher than RAG — travel is creative
    max_tokens: int = 2048     # itineraries can be long


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    return Settings()
