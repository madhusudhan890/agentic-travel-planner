"""
app/chain/registry.py
──────────────────────
Chain registry + factory — mirrors ChatPDF's handlers/registry.py exactly.

To add a new LLM provider (e.g. OpenAI in Phase 2):
  1. Create app/chain/openai_chain.py implementing TravelChain
  2. Import it here and add to _REGISTRY
  3. Set LLM_PROVIDER=openai in .env
  ← That is the ONLY change needed. No route or server code changes.
"""

from __future__ import annotations

from typing import Dict, Type

from app.chain.base import TravelChain
from app.chain.gemini_chain import GeminiTravelChain
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Registry ───────────────────────────────────────────────────────────────────
# Maps the LLM_PROVIDER env value → concrete chain class.
_REGISTRY: Dict[str, Type[TravelChain]] = {
    GeminiTravelChain.name: GeminiTravelChain,
    # Phase 2: OpenAITravelChain.name: OpenAITravelChain,
}


def build_travel_chain() -> TravelChain:
    """
    Instantiate and return the chain selected by LLM_PROVIDER in .env.
    Called ONCE at startup inside the lifespan — result stored on app.state.

    Raises
    ------
    ValueError — if LLM_PROVIDER names an unknown provider.
    """
    settings = get_settings()
    provider_name = settings.llm_provider

    chain_cls = _REGISTRY.get(provider_name)
    if chain_cls is None:
        supported = ", ".join(_REGISTRY.keys())
        raise ValueError(
            f"Unknown LLM_PROVIDER='{provider_name}'. "
            f"Supported providers: {supported}"
        )

    logger.info("Building travel chain: '%s'", provider_name)
    return chain_cls()


def get_supported_providers() -> list[str]:
    """Return all registered provider names (used by /health endpoint)."""
    return list(_REGISTRY.keys())
