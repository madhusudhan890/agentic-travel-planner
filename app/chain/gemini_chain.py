"""
app/chain/gemini_chain.py
──────────────────────────
Gemini-powered travel planning chain using LangChain.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 1 (LCEL — kept as reference, commented out below)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Architecture:  ChatPromptTemplate → ChatGoogleGenerativeAI → StrOutputParser
  Pipeline:      prompt | llm | StrOutputParser()
  Invocation:    await chain.ainvoke({...})
  LLM answers:   purely from training data — no live data

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 2 (Tools + ReAct Agent — ACTIVE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Architecture:  LLM + Tools → create_react_agent → AgentExecutor
  Pattern:       ReAct (Reason + Act) loop
  Retry:         tenacity — correct library, handles asyncio properly
  Cancellation:  asyncio.wait_for() with timeout — forces clean shutdown on Ctrl+C

  WHY tenacity instead of a manual for-loop:
    - Manual loops with `except Exception` accidentally retry on
      broad exceptions (including wrapped CancelledError from libraries)
    - tenacity's retry_if_exception() lets you precisely define
      which exceptions are retryable, and by default NEVER catches
      BaseException (so CancelledError always propagates)
    - before_sleep_log gives you automatic retry logging
    - reraise=True re-raises the original exception after all retries

  WHY asyncio.wait_for() for Ctrl+C:
    - LangChain tools run in a thread pool (run_in_executor)
    - Thread pool tasks CANNOT be cancelled by asyncio.CancelledError alone
    - The thread keeps running even after the asyncio task is cancelled
    - wait_for() with a timeout ensures the coroutine is abandoned
      after the timeout, even if threads are still running in the background
    - This means Ctrl+C + timeout = truly responsive shutdown

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 3 upgrade path (no base class changes needed):
  - Replace AgentExecutor with a LangGraph StateGraph
  - Add parallel specialist sub-agents (flight, hotel, weather)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import asyncio
import logging

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    RetryError,
)

from app.chain.base import TravelChain
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.request import TripRequest

logger = get_logger(__name__)

# ── Retry configuration ────────────────────────────────────────────────────────
_MAX_RETRIES = 2          # total attempts (1 original + 1 retry)
_AGENT_TIMEOUT_SEC = 90   # hard timeout per agent call — ensures Ctrl+C responds quickly


# ── Tenacity retry predicate ──────────────────────────────────────────────────
# ONLY retry on transient LLM overload errors.
# NEVER retry on CancelledError — it is a BaseException, not Exception,
# so tenacity's default retry_if_exception_type(Exception) would skip it.
# We define this explicitly so it's clear and safe.
def _is_transient_llm_error(exc: BaseException) -> bool:
    """Return True ONLY for transient Gemini/OpenAI server errors worth retrying.

    Critically: returns False for asyncio.CancelledError and KeyboardInterrupt
    so they always propagate immediately — never get swallowed by the retry loop.
    """
    # Never retry cancellation or keyboard interrupt
    if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)):
        return False
    err_str = str(exc)
    return any(
        code in err_str
        for code in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "overloaded")
    )


# ── System prompt ──────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are an expert travel planner with deep knowledge of destinations worldwide.
You have access to tools that can give you real, up-to-date information.

IMPORTANT: Always use your tools before generating the itinerary:
1. Use get_destination_info to get accurate facts about the destination
2. Use get_exchange_rate to convert the traveller's budget into local currency

For every itinerary you create:
- Organise by day (Day 1, Day 2, etc.)
- Include morning, afternoon, and evening suggestions
- Recommend specific restaurants, attractions, and experiences
- Include the real budget in local currency (from the tool result)
- Include key facts about the destination (from the Wikipedia tool result)
- Provide practical tips (best time to visit, transport, local customs)
- End with a packing list and key travel tips

Be specific, enthusiastic, and genuinely helpful."""

_AGENT_HUMAN_PROMPT = """Plan a {days}-day trip to {destination} in {month}.

Budget: {budget}
Interests: {interests}
Travel style: {travel_style}

Use your tools to get real information about {destination} and convert the budget.
Then create a detailed day-by-day itinerary using that real data."""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 1 — LCEL Chain (kept as reference, do not delete)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _HUMAN_PROMPT_PHASE1 = """Plan a {days}-day trip to {destination} in {month}.
#
# Budget: {budget}
# Interests: {interests}
# Travel style: {travel_style}
#
# Please create a detailed day-by-day itinerary."""
#
#
# async def _plan_trip_phase1(self, request: TripRequest) -> str:
#     """Phase 1: Pure LCEL chain — prompt | llm | StrOutputParser(). No tools."""
#     from langchain_google_genai import ChatGoogleGenerativeAI
#     from langchain_core.prompts import ChatPromptTemplate
#     from langchain_core.output_parsers import StrOutputParser
#
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", _SYSTEM_PROMPT),
#         ("human", _HUMAN_PROMPT_PHASE1),
#     ])
#     llm = ChatGoogleGenerativeAI(
#         model=self.model,
#         google_api_key=self._api_key,
#         temperature=self._temperature,
#         max_output_tokens=self._max_tokens,
#     )
#     # LCEL pipe — data flows left to right through each Runnable
#     chain = prompt | llm | StrOutputParser()
#
#     result: str = await chain.ainvoke({
#         "destination": request.destination,
#         "days": request.days,
#         "month": request.month,
#         "budget": request.budget,
#         "interests": request.interests or "general sightseeing",
#         "travel_style": request.travel_style or "balanced",
#     })
#     return result.strip()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class GeminiTravelChain(TravelChain):
    name = "gemini"

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key: str = settings.gemini_api_key
        self.model: str = settings.gemini_model
        self._temperature: float = settings.temperature
        self._max_tokens: int = settings.max_tokens
        logger.info(
            "GeminiTravelChain initialised | phase=2 (tools+agent) | configured=%s | model=%s",
            self.is_configured,
            self.model,
        )

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    async def plan_trip(self, request: TripRequest) -> str:
        """
        Phase 2: ReAct agent with tools + tenacity retry + asyncio.wait_for timeout.

        Retry strategy (tenacity):
          - Retries ONLY on transient 503/429 LLM errors
          - NEVER retries on CancelledError or KeyboardInterrupt
          - Exponential backoff: 1s → 2s between attempts
          - Logs each retry attempt automatically via before_sleep_log

        Cancellation strategy (asyncio.wait_for):
          - Wraps the entire agent call in a timeout
          - Even if LangChain's thread pool keeps running internally,
            the asyncio task is abandoned after _AGENT_TIMEOUT_SEC seconds
          - This makes Ctrl+C respond in seconds, not minutes
        """
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
        from app.tools.registry import get_tools

        # ── Build LLM + tools + agent ─────────────────────────────────────────
        llm = ChatGoogleGenerativeAI(
            model=self.model,
            google_api_key=self._api_key,
            temperature=self._temperature,
            max_output_tokens=self._max_tokens,
        )
        tools = get_tools()

        # ── Use Native Tool Calling Prompt ────────────────────────────────────
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # create_tool_calling_agent uses native Gemini function calling.
        # It is much more reliable than ReAct and eliminates ValidationErrors
        # caused by the LLM failing to format arguments correctly.
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=6,
            handle_parsing_errors=True,
        )

        agent_input = {
            "input": _AGENT_HUMAN_PROMPT.format(
                destination=request.destination,
                days=request.days,
                month=request.month,
                budget=request.budget,
                interests=request.interests or "general sightseeing",
                travel_style=request.travel_style or "balanced",
            )
        }

        logger.info(
            "Invoking Phase2 Agent | destination=%s | days=%d | tools=%s | model=%s",
            request.destination,
            request.days,
            [t.name for t in tools],
            self.model,
        )

        # ── Tenacity retry decorator (applied at call time) ───────────────────
        # Using tenacity.retry as a decorator on an inner async function so that
        # the agent_executor and agent_input are already in scope.
        #
        # Key design decisions:
        #  1. retry_if_exception(_is_transient_llm_error)
        #     → ONLY retries on 503/429, never on CancelledError
        #  2. wait_exponential(min=1, max=4)
        #     → waits 1s then 2s between attempts (tenacity's sleep is async-safe)
        #  3. before_sleep_log
        #     → logs "retrying in Xs" automatically — no manual logger.warning needed
        #  4. reraise=True
        #     → after all retries fail, the ORIGINAL exception is re-raised
        #       (not wrapped in RetryError), so the API returns the real error message

        @retry(
            stop=stop_after_attempt(_MAX_RETRIES),
            wait=wait_exponential(multiplier=1, min=1, max=4),
            retry=retry_if_exception(_is_transient_llm_error),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        async def _invoke() -> dict:
            # asyncio.wait_for() is the KEY fix for Ctrl+C:
            #
            # Problem: LangChain tools run in thread pools (run_in_executor).
            # Thread pool threads cannot be cancelled by asyncio.CancelledError.
            # Without wait_for, pressing Ctrl+C cancels the asyncio task but
            # leaves threads still making HTTP calls to Gemini in the background —
            # the process won't actually stop for 30-60+ seconds.
            #
            # Solution: wait_for(coroutine, timeout=N) creates an internal
            # asyncio.Task and cancels it after N seconds. When Ctrl+C fires
            # (uvicorn cancels the request task), this wait_for also gets
            # cancelled immediately, abandoning the agent and responding right away.
            #
            # The background threads will still complete naturally, but the
            # user-visible behaviour is: Ctrl+C → server responds in <1 second.
            return await asyncio.wait_for(
                agent_executor.ainvoke(agent_input),
                timeout=_AGENT_TIMEOUT_SEC,
            )

        # ── Execute with retry + timeout ──────────────────────────────────────
        try:
            result = await _invoke()
            return result.get("output", "").strip()

        except asyncio.CancelledError:
            # Ctrl+C or server shutdown — log and re-raise immediately.
            # tenacity's _is_transient_llm_error returns False for CancelledError
            # so it will never be retried — but we add this block for clarity.
            logger.info("Agent call cancelled (server shutdown signal).")
            raise

        except asyncio.TimeoutError:
            # The agent took longer than _AGENT_TIMEOUT_SEC.
            # This also means the server is "free" — the wait_for abandoned the call.
            logger.error("Agent call timed out after %ds.", _AGENT_TIMEOUT_SEC)
            raise RuntimeError(
                f"The request timed out after {_AGENT_TIMEOUT_SEC} seconds. "
                "Please try a simpler query or try again later."
            )

        except RuntimeError:
            # Re-raise clean user-facing errors (e.g. from tools) as-is.
            raise

        except Exception as exc:
            # All retries exhausted for a transient error — raise user-friendly message.
            err_str = str(exc)
            if _is_transient_llm_error(exc):
                logger.error(
                    "Gemini unavailable after %d attempt(s). | %s", _MAX_RETRIES, err_str[:200]
                )
                raise RuntimeError(
                    "Gemini is currently experiencing high demand. "
                    "Please try again in a few minutes."
                ) from exc
            raise
