"""
app/server.py
──────────────
FastAPI application factory — Phase 3: LangGraph Multi-Agent System.

Startup sequence:
  1. Configure logging
  2. Init LangSmith observability (if configured)
  3. Build the LangGraph travel workflow
  4. Ingest RAG knowledge base (if empty)
  5. Build Phase 2 chain (backward compat)
  6. Mount routers + static files
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.chain.registry import build_travel_chain
from app.api.routes import chat
from app.api.routes import graph as graph_route

logger = get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown lifecycle."""
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("Starting %s v%s", settings.app_title, settings.app_version)

    # ── 1. LangSmith observability ─────────────────────────────────────────
    if settings.langsmith_enabled:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
        logger.info("LangSmith tracing ENABLED | project=%s", settings.langsmith_project)
    else:
        logger.info("LangSmith tracing DISABLED (set LANGSMITH_API_KEY in .env to enable)")

    # ── 2. LangGraph travel graph ──────────────────────────────────────────
    try:
        from app.graph.builder import get_travel_graph
        app.state.travel_graph = get_travel_graph()
        logger.info("LangGraph multi-agent graph compiled and ready")
    except Exception as e:
        logger.error("LangGraph graph failed to build: %s", e)
        app.state.travel_graph = None

    # ── 3. RAG knowledge base ingestion ───────────────────────────────────
    try:
        from app.rag.ingestor import ingest_knowledge_base
        rag_result = ingest_knowledge_base(force_reingest=False)
        logger.info("RAG knowledge base: %s", rag_result)
    except Exception as e:
        logger.warning("RAG ingestion skipped: %s", e)

    # ── 4. Phase 2 chain (backward compat) ────────────────────────────────
    try:
        app.state.travel_chain = build_travel_chain()
        logger.info(
            "Phase2 TravelChain ready | provider=%s | model=%s",
            app.state.travel_chain.name,
            app.state.travel_chain.model,
        )
    except Exception as e:
        logger.error("Phase2 chain failed: %s", e)
        app.state.travel_chain = None

    logger.info("🚀 Application startup complete")

    yield  # ── Application running ─────────────────────────────────────────

    logger.info("%s shutting down.", settings.app_title)


def create_app() -> FastAPI:
    """Construct and return the fully configured FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_title,
        description=settings.app_description,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=_lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Static Files ──────────────────────────────────────────────────────────
    static_dir = Path("static")
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory="static"), name="static")

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(chat.router)         # Phase 2 backward compat
    app.include_router(graph_route.router)  # Phase 3 LangGraph

    # ── Root ──────────────────────────────────────────────────────────────────
    @app.get("/", tags=["Root"], response_class=HTMLResponse)
    def root():
        index = Path("static/index.html")
        if index.exists():
            return HTMLResponse(content=index.read_text())
        return HTMLResponse(content="<h1>AI Travel Planner Enterprise</h1><p>See <a href='/docs'>/docs</a></p>")

    # ── Health ────────────────────────────────────────────────────────────────
    @app.get("/health", tags=["Health"])
    def health():
        chain = getattr(app.state, "travel_chain", None)
        graph = getattr(app.state, "travel_graph", None)
        return {
            "status": "ok",
            "phase": "3 — LangGraph Multi-Agent",
            "llm_provider": chain.name if chain else "unknown",
            "llm_model": chain.model if chain else "unknown",
            "llm_configured": chain.is_configured if chain else False,
            "graph_ready": graph is not None,
            "langsmith_enabled": settings.langsmith_enabled,
        }

    return app
