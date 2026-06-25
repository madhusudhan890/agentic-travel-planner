"""
app/chain/base.py
──────────────────
Abstract base class that every LangChain travel chain must implement.

This mirrors the LLMHandler pattern from ChatPDF — but adapted for LangChain.
Instead of `generate_answer(question, context_chunks)`, travel chains expose
`plan_trip(request)` which takes a structured TripRequest.

Adding a new chain variant:
  1. Subclass TravelChain in a new file under app/chain/
  2. Implement `is_configured` and `plan_trip`
  3. Register it in app/chain/registry.py
  4. Set LLM_PROVIDER=<name> in .env  ← only change needed
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.request import TripRequest


class TravelChain(ABC):
    """
    Abstract interface for a travel planning chain.

    Each concrete implementation:
      - Wraps a specific LLM via LangChain (Gemini, OpenAI, etc.)
      - Owns the prompt template and output parsing
      - Is completely transparent to the route — route only calls `plan_trip`
    """

    #: Short identifier used by the registry and returned in responses.
    name: str = "base"

    #: The underlying LLM model name (for logging/health endpoint).
    model: str = ""

    # ── Abstract interface ─────────────────────────────────────────────────────

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if all required API keys/config are present."""

    @abstractmethod
    async def plan_trip(self, request: TripRequest) -> str:
        """
        Run the LangChain chain and return the full itinerary as a string.

        Parameters
        ----------
        request: TripRequest — the user's trip parameters (destination, days, etc.)
        """
