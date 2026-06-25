"""
app/server.py
──────────────
FastAPI application factory — identical pattern to ChatPDF's server.py.

`create_app()` is the single source of truth for the application.
It wires together:
  - Settings & Logging
  - CORS Middleware
  - TravelChain (stored on app.state via lifespan)
  - Routers
  - Static UI files

Phase upgrade path:
  - Phase 2: add `app.state.tools = build_tools()` in lifespan
  - Phase 3: add `app.state.agent = build_agent(chain, tools)` in lifespan
  Route code and DI code stay the same.
"""

from __future__ import annotations

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

logger = get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Startup / shutdown lifecycle manager.
    Everything before `yield` runs on startup.
    Everything after `yield` runs on shutdown.
    """
    settings = get_settings()

    # 1. Configure logging first — every subsequent log is formatted correctly
    configure_logging(settings.log_level)
    logger.info("Starting %s v%s", settings.app_title, settings.app_version)

    # 2. Build and store the travel chain on app.state
    #    This is the only place the chain is instantiated — once, at startup.
    #    Routes receive it via Depends(get_travel_chain).
    app.state.travel_chain = build_travel_chain()
    logger.info(
        "TravelChain ready | provider=%s | model=%s | configured=%s",
        app.state.travel_chain.name,
        app.state.travel_chain.model,
        app.state.travel_chain.is_configured,
    )

    yield  # ── Application is running ──────────────────────────────────────────

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

    # ── CORS Middleware ────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Static Files (UI) ─────────────────────────────────────────────────────
    static_dir = Path("static")
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory="static"), name="static")

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(chat.router)

    # ── Root & Health ─────────────────────────────────────────────────────────
    @app.get("/", tags=["Root"], summary="Travel Planner UI", response_class=HTMLResponse)
    def root():
        index = Path("static/index.html")
        if index.exists():
            return HTMLResponse(content=index.read_text())
        return HTMLResponse(
            content="<h1>Agentic Travel Planner</h1><p>See <a href='/docs'>/docs</a></p>"
        )

    @app.get("/health", tags=["Health"], summary="Health & provider status")
    def health():
        return {
            "status": "ok",
            "phase": "1 — LangChain Chain",
            "llm_provider": app.state.travel_chain.name,
            "llm_model": app.state.travel_chain.model,
            "llm_configured": app.state.travel_chain.is_configured,
        }

    return app
