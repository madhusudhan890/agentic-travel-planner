"""
app/chain/gemini_chain.py
──────────────────────────
Gemini-powered travel planning chain using LangChain.

Phase 1 architecture:
  ChatPromptTemplate → ChatGoogleGenerativeAI → StrOutputParser

The `|` pipe operator is LangChain's LCEL (LangChain Expression Language).
It composes components into a single runnable pipeline that you invoke once.

Phase 2 upgrade path (no base class changes needed):
  - Override plan_trip to use llm.bind_tools([...])
  - Add manual tool execution before returning

Phase 3 upgrade path (no base class changes needed):
  - Replace the chain with a LangGraph create_react_agent
  - plan_trip becomes agent.invoke(...)
"""

from __future__ import annotations

import asyncio

from app.chain.base import TravelChain
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.request import TripRequest

logger = get_logger(__name__)

# Retry config for transient Gemini 503/429 errors
_MAX_RETRIES = 2      # total attempts: 1 initial + 1 retry = 2 tries, max 1s wait
_BASE_DELAY = 1.0    # seconds for the single retry pause

# ── System prompt (shared, easy to improve without code changes) ───────────────
_SYSTEM_PROMPT = """You are an expert travel planner with deep knowledge of destinations worldwide.
Your itineraries are detailed, practical, and tailored to the traveler's preferences.

For every itinerary you create:
- Organise by day (Day 1, Day 2, etc.)
- Include morning, afternoon, and evening suggestions
- Recommend specific restaurants, attractions, and experiences
- Provide practical tips (best time to visit, transport, local customs)
- Keep the budget in mind throughout
- End with a packing list and key travel tips

Be specific, enthusiastic, and genuinely helpful."""

_HUMAN_PROMPT = """Plan a {days}-day trip to {destination} in {month}.

Budget: {budget}
Interests: {interests}
Travel style: {travel_style}

Please create a detailed day-by-day itinerary."""


class GeminiTravelChain(TravelChain):
    name = "gemini"

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key: str = settings.gemini_api_key
        self.model: str = settings.gemini_model
        self._temperature: float = settings.temperature
        self._max_tokens: int = settings.max_tokens
        logger.info(
            "GeminiTravelChain initialised | configured=%s | model=%s",
            self.is_configured,
            self.model,
        )

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    async def plan_trip(self, request: TripRequest) -> str:
        """
        Run the LangChain LCEL chain:
          ChatPromptTemplate | ChatGoogleGenerativeAI | StrOutputParser

        Retries up to 3 times on transient 503/429 errors with exponential backoff.
        """
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
        except ImportError as exc:
            raise RuntimeError(
                "langchain-google-genai is not installed. Run: uv add langchain-google-genai"
            ) from exc

        # ── Build the chain (prompt | llm | parser) ────────────────────────────
        # This is the core LangChain LCEL pattern.
        # Each | connects a Runnable — data flows left to right.
        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human", _HUMAN_PROMPT),
        ])

        llm = ChatGoogleGenerativeAI(
            model=self.model,
            google_api_key=self._api_key,
            temperature=self._temperature,
            max_output_tokens=self._max_tokens,
        )

        # StrOutputParser extracts the plain string from the AIMessage object
        chain = prompt | llm | StrOutputParser()

        logger.info(
            "Invoking GeminiTravelChain | destination=%s | days=%d | model=%s",
            request.destination,
            request.days,
            self.model,
        )

        # ── Retry loop with exponential backoff ────────────────────────────────
        # CancelledError is a BaseException (Python 3.8+), not Exception.
        # We catch it explicitly in BOTH the API call AND the sleep so that
        # Ctrl+C propagates immediately at either point.
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                result: str = await chain.ainvoke({
                    "destination": request.destination,
                    "days": request.days,
                    "month": request.month,
                    "budget": request.budget,
                    "interests": request.interests or "general sightseeing",
                    "travel_style": request.travel_style or "balanced",
                })
                return result.strip()

            except asyncio.CancelledError:
                # Ctrl+C or server shutdown during the API call — stop immediately.
                logger.info("Request cancelled during Gemini call (shutdown signal).")
                raise

            except Exception as exc:
                err_str = str(exc)
                is_transient = any(
                    code in err_str
                    for code in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED")
                )

                if is_transient and attempt < _MAX_RETRIES:
                    delay = _BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        "Gemini transient error (attempt %d/%d) — retrying in %.0fs | %s",
                        attempt, _MAX_RETRIES, delay, err_str[:120],
                    )
                    try:
                        await asyncio.sleep(delay)  # cancellable pause
                    except asyncio.CancelledError:
                        # Ctrl+C during the retry wait — also stop immediately.
                        logger.info("Request cancelled during retry sleep (shutdown signal).")
                        raise

                elif is_transient:
                    # All retries exhausted — raise a clean user-facing message.
                    logger.error(
                        "Gemini unavailable after %d attempt(s). Giving up. | %s",
                        _MAX_RETRIES, err_str[:200],
                    )
                    raise RuntimeError(
                        "Gemini is currently experiencing high demand. "
                        "Please try again in a few minutes."
                    ) from exc

                else:
                    # Non-transient error (bad API key, invalid request) — fail fast.
                    raise
