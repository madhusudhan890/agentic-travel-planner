"""
app/api/dependencies.py
────────────────────────
FastAPI dependency functions shared across all routes.

The travel chain is resolved ONCE at startup, stored on `app.state`,
and injected into routes via `Depends(get_travel_chain)`.

This is identical to ChatPDF's dependency pattern — it keeps routes
completely decoupled from the chain implementation.
"""

from __future__ import annotations

from fastapi import HTTPException, Request

from app.chain.base import TravelChain


async def get_travel_chain(request: Request) -> TravelChain:
    """
    Return the active TravelChain from app.state.
    Raises HTTP 503 if the chain's API key is missing.
    """
    chain: TravelChain = request.app.state.travel_chain
    if not chain.is_configured:
        raise HTTPException(
            status_code=503,
            detail=(
                f"LLM provider '{chain.name}' is not configured. "
                "Set the corresponding API key in your .env file."
            ),
        )
    return chain
