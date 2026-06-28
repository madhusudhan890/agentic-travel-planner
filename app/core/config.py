"""
app/core/config.py
──────────────────
Centralised settings — every config field is typed, validated, and documented.

DESIGN PRINCIPLE: All configuration lives here. No magic strings elsewhere.
PRODUCTION RULE: All secrets come from environment variables. Never hardcode.

Resolution order (pydantic-settings):
  1. Real environment variables  ← highest priority
  2. .env file
  3. Field defaults              ← lowest priority
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
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
    app_title: str = "AI Travel Planner Enterprise"
    app_description: str = (
        "Enterprise-grade AI Travel Planner — Phase 3: LangGraph Multi-Agent System. "
        "Real-time streaming, specialist agents, RAG, memory, and human-in-the-loop."
    )
    app_version: str = "3.0.0"

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    log_level: str = "info"

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_origins: str = "*"

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    # ── LLM Provider ──────────────────────────────────────────────────────────
    llm_provider: str = Field(default="gemini", description="gemini | openai")

    @field_validator("llm_provider", mode="before")
    @classmethod
    def normalise_llm_provider(cls, v: str) -> str:
        return v.strip().lower()

    # ── Google Gemini ─────────────────────────────────────────────────────────
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash-lite"
    # WHY flash-lite: fastest + cheapest Gemini. For production with complex plans,
    # upgrade to "gemini-1.5-pro" — 10x more capable but 10x more expensive.

    # ── OpenAI (Phase 2 ready) ────────────────────────────────────────────────
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # ── Generation ────────────────────────────────────────────────────────────
    temperature: float = 0.4
    max_tokens: int = 4096       # Increased for detailed itineraries
    agent_timeout_sec: int = 120  # Per-agent timeout (Ctrl+C safety net)
    max_retries: int = 2          # Tenacity: 1 original + 1 retry

    # ── LangGraph ─────────────────────────────────────────────────────────────
    # WHY max_revisions: without a limit, writer↔reviewer can loop forever.
    # Production: 2-3 is optimal. Beyond that, the LLM rarely improves quality.
    max_revisions: int = 2
    max_agent_iterations: int = 6   # Per-agent ReAct iterations

    # ── LangSmith Observability ───────────────────────────────────────────────
    # WHY LangSmith: production AI systems NEED observability.
    # Without it you're flying blind — no visibility into which prompt caused
    # a bad output, how many tokens were used, or why an agent loop ran 6 times.
    # FREE tier: smith.langchain.com — up to 5k traces/month free.
    # Set LANGSMITH_API_KEY in .env to enable. Leave empty to disable.
    langsmith_api_key: str = ""
    langsmith_project: str = "agentic-travel-planner"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    @property
    def langsmith_enabled(self) -> bool:
        return bool(self.langsmith_api_key)

    # ── ChromaDB (Vector Database for RAG) ───────────────────────────────────
    # WHY local ChromaDB: zero setup, no API key, runs embedded in process.
    # MIGRATION PATH to production:
    #   1. Qdrant: docker run -p 6333:6333 qdrant/qdrant
    #      Then set CHROMA_MODE=qdrant + QDRANT_URL=http://localhost:6333
    #   2. pgvector: ALTER TABLE add COLUMN embedding vector(768);
    #      Then set CHROMA_MODE=pgvector + POSTGRES_URL=postgresql://...
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection_name: str = "travel_knowledge"
    rag_top_k: int = 5              # Number of chunks to retrieve
    rag_chunk_size: int = 800       # Characters per chunk
    rag_chunk_overlap: int = 200    # Overlap between chunks

    # ── Memory ────────────────────────────────────────────────────────────────
    # WHY window memory: keeps last N messages, not the full history.
    # Token budget: 4K context × $0.002/1K = $0.008 per "memory load"
    # At k=10 messages you pay ~$0.005 in context every time. Balance this.
    memory_window_k: int = 10      # Short-term: last 10 messages
    summary_threshold: int = 20    # When to trigger summarization
    longterm_db_path: str = "./data/user_preferences.json"  # Simple JSON for dev

    # ── PDF Output ────────────────────────────────────────────────────────────
    pdf_output_dir: str = "./data/pdfs"

    # ── Email ─────────────────────────────────────────────────────────────────
    # WHY SMTP: free, uses your own email server. For production, use:
    # SendGrid / AWS SES / Mailgun — they have free tiers too.
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = ""

    @property
    def email_configured(self) -> bool:
        return bool(self.smtp_host and self.smtp_user)

    # ── Security (Guardrails) ─────────────────────────────────────────────────
    # WHY API keys: prevent abuse. Even internal tools need auth in production.
    api_key: str = ""   # Optional: set API_KEY in .env to require X-API-Key header

    @property
    def auth_required(self) -> bool:
        return bool(self.api_key)

    # ── Data Paths ────────────────────────────────────────────────────────────
    @property
    def knowledge_dir(self) -> Path:
        return Path("./data/travel_knowledge")

    @property
    def pdf_dir(self) -> Path:
        p = Path(self.pdf_output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    return Settings()
