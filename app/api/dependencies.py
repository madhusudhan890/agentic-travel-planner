"""
app/api/dependencies.py
────────────────────────
FastAPI dependency injection for the graph and chain layers.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from app.chain.base import TravelChain
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(_api_key_header)) -> bool:
    """Optional API key authentication guard."""
    if not settings.auth_required:
        return True
    if api_key == settings.api_key:
        return True
    raise HTTPException(status_code=403, detail="Invalid or missing API key")


def get_travel_chain(request: Request) -> TravelChain:
    """Inject the Phase 2 TravelChain from app.state."""
    chain = getattr(request.app.state, "travel_chain", None)
    if chain is None:
        raise HTTPException(status_code=503, detail="Travel chain not initialized")
    return chain
