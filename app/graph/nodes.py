"""
app/graph/nodes.py
───────────────────
All LangGraph node functions — the core of the multi-agent workflow.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT A NODE IS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A LangGraph node is a Python function that:
  INPUT:  receives the full state dict (read-only)
  OUTPUT: returns a PARTIAL state dict (only fields it changed)

LangGraph MERGES the return value into the existing state.
Unreturned fields stay unchanged.

Example:
  state = {"destination": "Tokyo", "weather_data": None, "flight_data": None}

  def weather_node(state) → {"weather_data": "Sunny, 25°C"}

  After node: {"destination": "Tokyo", "weather_data": "Sunny, 25°C", "flight_data": None}
  ↑ Only weather_data changed. destination and flight_data are unchanged.

WHY PARTIAL RETURNS:
  In parallel fan-out (multiple agents running simultaneously),
  each agent writes ONLY its own field. No conflicts.
  If they all returned the full state, the last writer would win,
  losing all other agents' work.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CANCELLATION SAFETY (Ctrl+C)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Every LLM call is wrapped in asyncio.wait_for() with a timeout.
CancelledError is NEVER caught — always re-raised.
This ensures Ctrl+C immediately stops execution.

Pattern (must be used in EVERY async LLM call):
  try:
      result = await asyncio.wait_for(llm_call(), timeout=TIMEOUT)
  except asyncio.CancelledError:
      raise  # Never swallow this
  except asyncio.TimeoutError:
      return {"errors": [f"{node}: timed out"]}
  except Exception as exc:
      if _is_transient_llm_error(exc):
          # tenacity handles retry
      return {"errors": [str(exc)]}
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from app.core.config import get_settings
from app.core.logging import get_logger
from app.graph.state import AgentStatusInfo

logger = get_logger(__name__)
settings = get_settings()

# ── Retry predicate (same as Phase 2 — never swallow CancelledError) ──────────
def _is_transient_llm_error(exc: BaseException) -> bool:
    """Return True ONLY for transient LLM errors worth retrying."""
    if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)):
        return False
    err_str = str(exc)
    return any(code in err_str for code in (
        "503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "overloaded", "rate limit"
    ))


def _build_llm():
    """Build the Gemini LLM for node use."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=settings.temperature,
        max_output_tokens=settings.max_tokens,
    )


def _build_agent_executor(system_prompt: str, tools: list, llm=None):
    """Build a tool-calling AgentExecutor for a specialist agent."""
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.agents import create_tool_calling_agent, AgentExecutor

    if llm is None:
        llm = _build_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt) if tools else None

    if agent:
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            max_iterations=settings.max_agent_iterations,
            handle_parsing_errors=True,
        )
    return None


def _update_node_status(node_name: str, status: str, output: str = "") -> dict:
    """Helper to build node status update for UI streaming."""
    return {
        "current_node": node_name,
        "node_statuses": {node_name: status},
        "node_outputs": {node_name: output[:500] if output else ""},
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 1: PLANNER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract structured trip intent and decide which agents to activate.

    TEACHES:
    - Structured Outputs (LLM forced to return JSON)
    - Intent Extraction (converting free-form to structured data)
    - Router logic (which downstream agents are needed)

    WHY THIS NODE EXISTS:
    The user provides free-form input. Every downstream agent needs
    structured data (num_days as int, budget as parsed number, etc.).
    The planner is the "parser" that creates a consistent structure
    that all other nodes can reliably access.
    """
    logger.info("=== PLANNER NODE START === destination=%s", state["destination"])
    start_time = time.time()

    # Determine which agents to run based on input
    destination = state["destination"]
    has_departure = bool(state.get("departure_city", "").strip())
    interests = (state.get("interests") or "").lower()
    days = state.get("days", 1)

    agents_to_run = ["weather", "hotel", "budget", "restaurant", "currency"]

    # Flight agent: only if departure city is provided
    if has_departure:
        agents_to_run.insert(0, "flight")

    # Visa agent: for international trips (simplified heuristic)
    # Production: use proper country detection
    visa_hints = ["international", "abroad", "overseas"]
    always_visa_countries = ["japan", "china", "india", "usa", "uk", "france", "thailand", "australia"]
    if any(h in destination.lower() for h in always_visa_countries) or days > 3:
        agents_to_run.append("visa")

    # Build structured trip plan
    trip_plan = {
        "destination": destination,
        "days": days,
        "month": state.get("month", "October"),
        "budget": state.get("budget", "$2000"),
        "interests": state.get("interests", "general sightseeing"),
        "travel_style": state.get("travel_style", "balanced"),
        "num_travelers": state.get("num_travelers", 1),
        "departure_city": state.get("departure_city", ""),
        "agents_activated": agents_to_run,
    }

    elapsed = time.time() - start_time
    output_summary = f"Activated {len(agents_to_run)} agents: {', '.join(agents_to_run)}"
    logger.info("PLANNER complete in %.2fs | %s", elapsed, output_summary)

    return {
        "trip_plan": trip_plan,
        "agents_to_run": agents_to_run,
        "current_node": "planner",
        "node_statuses": {**state.get("node_statuses", {}), "planner": AgentStatusInfo.COMPLETED},
        "node_outputs": {**state.get("node_outputs", {}), "planner": output_summary},
        "progress_pct": 10,
        "node_timings": {**state.get("node_timings", {}), "planner": elapsed},
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 2: RETRIEVER (RAG)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async def retriever_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve relevant context from ChromaDB knowledge base.

    TEACHES:
    - RAG retrieval (semantic search over pre-indexed documents)
    - Context compression (selecting most relevant chunks)
    - Grounding (giving LLMs factual context before generating)

    WHY RAG HERE:
    Without RAG, the writer relies purely on LLM training data which may be:
    - Outdated (knowledge cutoff)
    - Hallucinated (confident but wrong)
    - Generic (not specific to user's budget/style)

    With RAG, we retrieve specific, verified facts from our knowledge base
    about this destination before the writer generates content.
    """
    logger.info("=== RETRIEVER NODE START ===")
    start_time = time.time()

    destination = state["destination"]
    interests = state.get("interests", "")
    budget = state.get("budget", "")

    try:
        from app.rag.vectorstore import similarity_search_with_score

        # Build multiple queries to retrieve different types of context
        queries = [
            f"{destination} travel tips and attractions",
            f"{destination} budget travel {budget}",
            f"{destination} {interests} activities",
            f"{destination} visa requirements entry",
        ]

        all_chunks = []
        seen_content = set()

        for query in queries:
            results = similarity_search_with_score(query=query, k=3, score_threshold=0.2)
            for doc, score in results:
                content_key = doc.page_content[:100]
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    all_chunks.append(f"[Score: {score:.2f}] {doc.page_content}")

        if all_chunks:
            rag_context = "\n\n---\n\n".join(all_chunks[:8])  # Max 8 chunks
            output_summary = f"Retrieved {len(all_chunks)} relevant chunks from knowledge base"
        else:
            rag_context = f"No specific knowledge base content found for {destination}. Using general travel knowledge."
            output_summary = "No RAG matches — using general knowledge"

    except Exception as e:
        logger.warning("RAG retrieval failed: %s", e)
        rag_context = f"Knowledge base unavailable: {e}"
        output_summary = f"RAG error: {e}"

    elapsed = time.time() - start_time
    logger.info("RETRIEVER complete in %.2fs | %s", elapsed, output_summary)

    return {
        "rag_context": rag_context,
        "current_node": "retriever",
        "node_statuses": {**state.get("node_statuses", {}), "retriever": AgentStatusInfo.COMPLETED},
        "node_outputs": {**state.get("node_outputs", {}), "retriever": output_summary},
        "progress_pct": 20,
        "node_timings": {**state.get("node_timings", {}), "retriever": elapsed},
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 3: PARALLEL SPECIALIST AGENTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async def run_agents_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run all specialist agents IN PARALLEL using asyncio.gather.

    TEACHES:
    - Parallel execution (agents run concurrently, not sequentially)
    - Async/await patterns for concurrent LLM calls
    - Error isolation (one agent failure doesn't kill others)
    - Timeout handling per agent (Ctrl+C safety)

    WHY PARALLEL:
    Sequential agents: 7 agents × 5s each = 35s total
    Parallel agents:  7 agents × 5s each = ~5s total (limited by slowest)

    WHY asyncio.gather over threading:
    LLM calls are I/O-bound (waiting for API responses), not CPU-bound.
    asyncio.gather handles I/O concurrency perfectly without OS threads.
    ThreadPoolExecutor would be needed for CPU-bound tasks.

    LANGGRAPH SEND API (alternative for true parallel branches):
    LangGraph's Send() API can dispatch to multiple nodes simultaneously.
    We use asyncio.gather within a single node for simplicity — same result
    but all parallel work happens in one logical node.

    Production with LangGraph Send:
        from langgraph.constants import Send
        def router_to_agents(state):
            return [Send(agent, state) for agent in state["agents_to_run"]]
    """
    logger.info("=== PARALLEL AGENTS START === agents=%s", state.get("agents_to_run", []))
    start_time = time.time()

    agents_to_run = state.get("agents_to_run", [])
    destination = state["destination"]
    days = state.get("days", 5)
    month = state.get("month", "October")
    budget = state.get("budget", "$2000")
    interests = state.get("interests", "general sightseeing")
    travel_style = state.get("travel_style", "balanced")
    num_travelers = state.get("num_travelers", 1)
    departure_city = state.get("departure_city", "")

    # Build coroutines for each requested agent
    agent_coroutines = {}

    if "flight" in agents_to_run and departure_city:
        agent_coroutines["flight"] = _run_flight_agent(departure_city, destination, month, num_travelers)
    if "hotel" in agents_to_run:
        agent_coroutines["hotel"] = _run_hotel_agent(destination, month, days, num_travelers, travel_style)
    if "weather" in agents_to_run:
        agent_coroutines["weather"] = _run_weather_agent(destination, days)
    if "budget" in agents_to_run:
        agent_coroutines["budget"] = _run_budget_agent(destination, budget, days, num_travelers)
    if "restaurant" in agents_to_run:
        agent_coroutines["restaurant"] = _run_restaurant_agent(destination, interests, days)
    if "visa" in agents_to_run:
        agent_coroutines["visa"] = _run_visa_agent(destination)
    if "currency" in agents_to_run:
        agent_coroutines["currency"] = _run_currency_agent(destination, budget, num_travelers, days)

    if not agent_coroutines:
        return {
            "current_node": "agents",
            "progress_pct": 60,
        }

    # Run ALL agents concurrently
    # return_exceptions=True: a failed agent returns Exception, not crash
    agent_names = list(agent_coroutines.keys())
    results = await asyncio.gather(
        *agent_coroutines.values(),
        return_exceptions=True,
    )

    # Collect results into state fields
    updates: Dict[str, Any] = {}
    errors = list(state.get("errors", []))
    node_statuses = dict(state.get("node_statuses", {}))
    node_outputs = dict(state.get("node_outputs", {}))

    for agent_name, result in zip(agent_names, results):
        if isinstance(result, asyncio.CancelledError):
            raise result  # Always propagate cancellation
        elif isinstance(result, Exception):
            error_msg = f"{agent_name}_agent: {result}"
            errors.append(error_msg)
            logger.error("Agent %s failed: %s", agent_name, result)
            updates[f"{agent_name}_data"] = f"⚠️ {agent_name.title()} agent failed: {result}"
            node_statuses[f"{agent_name}_agent"] = AgentStatusInfo.ERROR
        else:
            updates[f"{agent_name}_data"] = result
            node_statuses[f"{agent_name}_agent"] = AgentStatusInfo.COMPLETED
            node_outputs[f"{agent_name}_agent"] = (result or "")[:200]
            logger.info("Agent %s: OK (%d chars)", agent_name, len(result or ""))

    elapsed = time.time() - start_time
    logger.info(
        "PARALLEL AGENTS complete in %.2fs | %d agents ran",
        elapsed, len(agent_names),
    )

    return {
        **updates,
        "errors": errors,
        "current_node": "agents",
        "node_statuses": node_statuses,
        "node_outputs": node_outputs,
        "progress_pct": 60,
        "node_timings": {**state.get("node_timings", {}), "agents": elapsed},
    }


# ── Individual agent runners ───────────────────────────────────────────────────
async def _run_flight_agent(origin: str, destination: str, month: str, num_travelers: int) -> str:
    from app.tools.flight_tool import search_flights
    result = await asyncio.wait_for(
        asyncio.get_event_loop().run_in_executor(
            None,
            lambda: search_flights.invoke({
                "origin_city": origin, "destination_city": destination,
                "travel_month": month, "num_travelers": num_travelers,
            }),
        ),
        timeout=settings.agent_timeout_sec,
    )
    return result


async def _run_hotel_agent(destination: str, month: str, days: int, num_travelers: int, style: str) -> str:
    from app.tools.hotel_tool import search_hotels
    budget_level = "luxury" if "luxury" in style.lower() else ("budget" if "budget" in style.lower() else "moderate")
    result = await asyncio.wait_for(
        asyncio.get_event_loop().run_in_executor(
            None,
            lambda: search_hotels.invoke({
                "destination": destination, "check_in_month": month,
                "num_nights": days, "num_travelers": num_travelers,
                "budget_level": budget_level,
            }),
        ),
        timeout=settings.agent_timeout_sec,
    )
    return result


async def _run_weather_agent(destination: str, days: int) -> str:
    from app.tools.weather_tool import get_weather_forecast
    result = await asyncio.wait_for(
        asyncio.get_event_loop().run_in_executor(
            None,
            lambda: get_weather_forecast.invoke({"destination": destination, "days": days}),
        ),
        timeout=settings.agent_timeout_sec,
    )
    return result


async def _run_budget_agent(destination: str, budget: str, days: int, num_travelers: int) -> str:
    from app.tools.calculator_tool import calculate_trip_budget
    # Parse budget amount from string like "$2000" or "₹80,000"
    import re
    nums = re.findall(r"[\d,]+", budget.replace(",", ""))
    budget_amount = float(nums[0]) if nums else 2000.0

    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: calculate_trip_budget.invoke({
            "total_budget": budget_amount,
            "num_travelers": num_travelers,
            "num_days": days,
            "hotel_cost_per_night": 80.0,
            "food_cost_per_day": 40.0,
            "activities_per_day": 30.0,
            "transport_per_day": 20.0,
        }),
    )
    return result


async def _run_restaurant_agent(destination: str, interests: str, days: int) -> str:
    from app.tools.restaurant_tool import search_restaurants
    result = await asyncio.wait_for(
        asyncio.get_event_loop().run_in_executor(
            None,
            lambda: search_restaurants.invoke({
                "destination": destination, "interests": interests, "days": days,
            }),
        ),
        timeout=settings.agent_timeout_sec,
    )
    return result


async def _run_visa_agent(destination: str) -> str:
    from app.tools.visa_tool import get_visa_requirements
    result = await asyncio.wait_for(
        asyncio.get_event_loop().run_in_executor(
            None,
            lambda: get_visa_requirements.invoke({"destination": destination}),
        ),
        timeout=settings.agent_timeout_sec,
    )
    return result


async def _run_currency_agent(destination: str, budget: str, num_travelers: int, days: int) -> str:
    from app.tools.currency_tool import get_exchange_rate
    # Get exchange rate for destination currency
    currency_map = {
        "japan": ("USD", "JPY"), "thailand": ("USD", "THB"), "india": ("USD", "INR"),
        "france": ("USD", "EUR"), "uk": ("USD", "GBP"), "australia": ("USD", "AUD"),
        "canada": ("USD", "CAD"), "mexico": ("USD", "MXN"), "brazil": ("USD", "BRL"),
    }
    dest_lower = destination.lower()
    currencies = ("USD", "USD")
    for country, pair in currency_map.items():
        if country in dest_lower:
            currencies = pair
            break

    if currencies[0] == currencies[1]:
        return f"💱 Currency: {destination} uses the same currency (likely USD/local)."

    import re
    nums = re.findall(r"[\d,]+", budget.replace(",", ""))
    amount = float(nums[0]) if nums else 2000.0

    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: get_exchange_rate.invoke({
            "base_currency": currencies[0],
            "target_currency": currencies[1],
            "amount": amount,
        }),
    )
    return f"## 💱 Currency Exchange\n{result}\n\nTotal budget per person: {amount/num_travelers:.0f} {currencies[0]}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 4: WRITER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async def writer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synthesize all agent outputs into a comprehensive itinerary.

    TEACHES:
    - Context assembly (stuffing all context into one prompt)
    - Hallucination prevention (ground truth from agents)
    - Long-form generation (multi-day itineraries)
    - Prompt construction from multiple data sources

    WHY WRITER IS SEPARATE FROM AGENTS:
    Agents gather data. Writer synthesizes data.
    Mixing these two jobs in one prompt produces worse output.
    A writer that also calls tools gets confused about
    "when should I call tools vs when should I write."
    """
    logger.info("=== WRITER NODE START === revision=%d", state.get("revision_count", 0))
    start_time = time.time()

    from app.prompts.writer_prompt import WRITER_SYSTEM_PROMPT, WRITER_HUMAN_PROMPT
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    llm = _build_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", WRITER_SYSTEM_PROMPT),
        ("human", WRITER_HUMAN_PROMPT),
    ])
    chain = prompt | llm | StrOutputParser()

    human_input = {
        "destination": state["destination"],
        "days": state.get("days", 5),
        "month": state.get("month", "October"),
        "budget": state.get("budget", "$2000"),
        "num_travelers": state.get("num_travelers", 1),
        "interests": state.get("interests", "general sightseeing"),
        "travel_style": state.get("travel_style", "balanced"),
        "rag_context": state.get("rag_context") or "No knowledge base context available.",
        "flight_data": state.get("flight_data") or "No flight data (no departure city provided).",
        "hotel_data": state.get("hotel_data") or "No hotel data available.",
        "weather_data": state.get("weather_data") or "No weather data available.",
        "restaurant_data": state.get("restaurant_data") or "No restaurant data available.",
        "visa_info": state.get("visa_info") or "No visa info — verify requirements at official embassy website.",
        "currency_info": state.get("currency_info") or "No currency data available.",
        "budget_analysis": state.get("budget_analysis") or "No budget breakdown available.",
        "review_feedback": state.get("review_feedback") or "First draft — no previous feedback.",
    }

    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception(_is_transient_llm_error),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _invoke_writer() -> str:
        return await asyncio.wait_for(
            chain.ainvoke(human_input),
            timeout=settings.agent_timeout_sec,
        )

    try:
        draft = await _invoke_writer()
        revision_count = state.get("revision_count", 0)
        elapsed = time.time() - start_time
        logger.info("WRITER complete in %.2fs | %d chars | revision=%d", elapsed, len(draft), revision_count)

        return {
            "draft_itinerary": draft,
            "revision_count": revision_count,
            "current_node": "writer",
            "node_statuses": {**state.get("node_statuses", {}), "writer": AgentStatusInfo.COMPLETED},
            "node_outputs": {**state.get("node_outputs", {}), "writer": draft[:300]},
            "progress_pct": 70,
            "node_timings": {**state.get("node_timings", {}), "writer": elapsed},
        }

    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("WRITER failed: %s", exc)
        return {
            "errors": list(state.get("errors", [])) + [f"writer: {exc}"],
            "draft_itinerary": f"⚠️ Writer failed: {exc}\n\nPartial data collected:\n\n{state.get('flight_data', '')}\n{state.get('hotel_data', '')}",
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 5: REVIEWER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async def reviewer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Quality gate — reviews the draft itinerary and decides approve/revise.

    TEACHES:
    - LLM-as-judge pattern (using LLM to evaluate LLM output)
    - Structured JSON output parsing
    - Conditional loop control (review_passed → edges decide flow)
    - Self-consistency (separate critic from generator)

    LOOP MECHANISM:
    This node sets review_passed and review_feedback.
    The conditional edge `after_reviewer` reads review_passed:
    - True → proceed to human_approval
    - False AND revision_count < max_revisions → go back to writer
    - False AND revision_count >= max_revisions → proceed anyway

    WHY LIMIT REVISIONS:
    Without max_revisions, you could have infinite writer↔reviewer loops.
    The LLM might keep finding small issues that never satisfy the reviewer.
    2 revisions provides meaningful improvement without runaway costs.
    """
    logger.info("=== REVIEWER NODE START ===")
    start_time = time.time()

    from app.prompts.reviewer_prompt import REVIEWER_SYSTEM_PROMPT, REVIEWER_HUMAN_PROMPT
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    llm = _build_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", REVIEWER_SYSTEM_PROMPT),
        ("human", REVIEWER_HUMAN_PROMPT),
    ])
    chain = prompt | llm | StrOutputParser()

    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", settings.max_revisions)

    human_input = {
        "destination": state["destination"],
        "days": state.get("days", 5),
        "budget": state.get("budget", "$2000"),
        "num_travelers": state.get("num_travelers", 1),
        "month": state.get("month", "October"),
        "interests": state.get("interests", ""),
        "draft_itinerary": state.get("draft_itinerary", ""),
        "revision_count": revision_count,
        "max_revisions": max_revisions,
    }

    try:
        @retry(
            stop=stop_after_attempt(settings.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=4),
            retry=retry_if_exception(_is_transient_llm_error),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        async def _invoke_reviewer() -> str:
            return await asyncio.wait_for(
                chain.ainvoke(human_input),
                timeout=settings.agent_timeout_sec,
            )

        raw_output = await _invoke_reviewer()

        # Parse JSON response from reviewer
        review_passed = False
        feedback = ""
        score = 0

        try:
            # Extract JSON from output (LLM sometimes adds extra text)
            import re
            json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
            if json_match:
                review_data = json.loads(json_match.group())
                review_passed = review_data.get("approved", False)
                score = review_data.get("score", 5)
                feedback = review_data.get("feedback", "")
                issues = review_data.get("issues", [])
                if issues:
                    feedback = f"Issues: {', '.join(issues)}\n{feedback}"
            else:
                # Fallback: if LLM didn't return valid JSON, approve by default
                review_passed = True
                feedback = "Review format issue — auto-approved"

        except json.JSONDecodeError:
            review_passed = True  # Auto-approve on parse error
            feedback = "Review parse error — auto-approved"

        elapsed = time.time() - start_time
        action = "APPROVED" if review_passed else f"NEEDS REVISION (score={score})"
        logger.info("REVIEWER: %s in %.2fs | revision=%d/%d", action, elapsed, revision_count, max_revisions)

        return {
            "review_passed": review_passed,
            "review_feedback": feedback,
            "revision_count": revision_count + (0 if review_passed else 1),
            "current_node": "reviewer",
            "node_statuses": {**state.get("node_statuses", {}), "reviewer": AgentStatusInfo.COMPLETED},
            "node_outputs": {**state.get("node_outputs", {}), "reviewer": f"{action}: {feedback[:150]}"},
            "progress_pct": 75,
            "node_timings": {**state.get("node_timings", {}), "reviewer": elapsed},
        }

    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("REVIEWER failed: %s", exc)
        # On reviewer failure, auto-approve to avoid blocking
        return {
            "review_passed": True,
            "review_feedback": f"Reviewer error (auto-approved): {exc}",
            "errors": list(state.get("errors", [])) + [f"reviewer: {exc}"],
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 6: HUMAN APPROVAL (Human-in-the-Loop)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async def human_approval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Human-in-the-loop checkpoint.

    TEACHES:
    - interrupt() API for pausing graph execution
    - Stateful resumption (graph remembers where it paused)
    - Command API for providing human input to resume graph
    - Why human oversight matters in production AI systems

    HOW IT WORKS:
    1. Graph reaches this node
    2. interrupt() is called — this PAUSES execution and saves state
    3. The SSE stream sends a "human_approval_required" event to UI
    4. UI shows the itinerary for human review
    5. Human clicks Approve/Reject on the UI
    6. POST /graph/approve is called with approved=True/False
    7. Graph resumes from this node with the human's decision

    WHY HUMAN APPROVAL IN PRODUCTION:
    AI-generated travel itineraries may:
    - Recommend outdated information
    - Have budget errors the LLM missed
    - Suggest culturally insensitive activities
    - Have logical inconsistencies (impossible timing)
    A human reviewer catches these before triggering
    real-world actions (email, booking, payment).

    WITHOUT HUMAN APPROVAL (auto_approve=True):
    Graph skips the interrupt and proceeds directly.
    Use for: demo mode, trusted automation pipelines.
    """
    require_approval = state.get("require_human_approval", False)
    already_approved = state.get("human_approved", True)

    if not require_approval or already_approved:
        # Skip human approval (auto-approve mode)
        logger.info("Human approval: SKIPPED (auto-approve)")
        return {
            "human_approved": True,
            "final_itinerary": state.get("draft_itinerary", ""),
            "current_node": "human_approval",
            "node_statuses": {**state.get("node_statuses", {}), "human_approval": AgentStatusInfo.COMPLETED},
            "node_outputs": {**state.get("node_outputs", {}), "human_approval": "Auto-approved"},
            "progress_pct": 85,
        }

    # Human approval required — the LangGraph interrupt mechanism
    try:
        from langgraph.types import interrupt

        # This call PAUSES the graph and saves state to the checkpointer
        # The graph can only resume via graph.invoke(Command(resume={...}), config)
        human_response = interrupt({
            "message": "Please review the itinerary and approve or provide feedback.",
            "draft_itinerary": state.get("draft_itinerary", ""),
            "session_id": state.get("session_id", ""),
        })

        # When resumed, human_response contains the dict from Command(resume={...})
        approved = human_response.get("approved", True)
        feedback = human_response.get("feedback", "")

        logger.info("Human approval: %s | feedback=%s", "APPROVED" if approved else "REJECTED", feedback[:100])

        return {
            "human_approved": approved,
            "human_feedback": feedback,
            "final_itinerary": state.get("draft_itinerary") if approved else None,
            "review_feedback": feedback if not approved else state.get("review_feedback"),
            "current_node": "human_approval",
            "node_statuses": {**state.get("node_statuses", {}), "human_approval": AgentStatusInfo.COMPLETED},
            "node_outputs": {**state.get("node_outputs", {}), "human_approval": f"{'Approved' if approved else 'Rejected'}: {feedback[:100]}"},
            "progress_pct": 85,
        }

    except ImportError:
        # langgraph.types.interrupt not available — auto-approve
        logger.warning("interrupt() not available — auto-approving")
        return {
            "human_approved": True,
            "final_itinerary": state.get("draft_itinerary", ""),
            "current_node": "human_approval",
            "progress_pct": 85,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 7: PDF GENERATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async def pdf_generator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a PDF from the final approved itinerary."""
    logger.info("=== PDF GENERATOR NODE START ===")
    start_time = time.time()

    final_itinerary = state.get("final_itinerary") or state.get("draft_itinerary", "")
    if not final_itinerary:
        return {
            "errors": list(state.get("errors", [])) + ["pdf: no itinerary content"],
            "current_node": "pdf_generator",
            "node_statuses": {**state.get("node_statuses", {}), "pdf_generator": AgentStatusInfo.ERROR},
        }

    try:
        from app.tools.pdf_tool import generate_pdf_itinerary

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: generate_pdf_itinerary.invoke({
                "itinerary_markdown": final_itinerary,
                "destination": state["destination"],
                "session_id": state.get("session_id", "default"),
                "output_dir": settings.pdf_output_dir,
            }),
        )

        # Extract path from result string
        pdf_path = ""
        if "✅ PDF generated:" in result:
            pdf_path = result.split("✅ PDF generated:")[1].split("\n")[0].strip()

        elapsed = time.time() - start_time
        logger.info("PDF GENERATOR complete in %.2fs | path=%s", elapsed, pdf_path)

        return {
            "pdf_path": pdf_path,
            "current_node": "pdf_generator",
            "node_statuses": {**state.get("node_statuses", {}), "pdf_generator": AgentStatusInfo.COMPLETED},
            "node_outputs": {**state.get("node_outputs", {}), "pdf_generator": f"PDF: {pdf_path}"},
            "progress_pct": 92,
            "node_timings": {**state.get("node_timings", {}), "pdf_generator": elapsed},
        }

    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("PDF generation failed: %s", exc)
        return {
            "errors": list(state.get("errors", [])) + [f"pdf: {exc}"],
            "node_statuses": {**state.get("node_statuses", {}), "pdf_generator": AgentStatusInfo.ERROR},
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 8: EMAIL SENDER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async def email_sender_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Send the final itinerary via email if user provided one."""
    logger.info("=== EMAIL SENDER NODE START ===")
    start_time = time.time()

    user_email = state.get("user_email")
    if not user_email or not settings.email_configured:
        reason = "no user email" if not user_email else "email not configured"
        logger.info("Email: SKIPPED (%s)", reason)
        return {
            "email_sent": False,
            "current_node": "email_sender",
            "node_statuses": {**state.get("node_statuses", {}), "email_sender": AgentStatusInfo.COMPLETED},
            "node_outputs": {**state.get("node_outputs", {}), "email_sender": f"Skipped: {reason}"},
            "progress_pct": 98,
        }

    try:
        from app.tools.email_tool import send_itinerary_email

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: send_itinerary_email.invoke({
                "recipient_email": user_email,
                "destination": state["destination"],
                "itinerary_markdown": state.get("final_itinerary", ""),
                "pdf_path": state.get("pdf_path", ""),
                "smtp_host": settings.smtp_host,
                "smtp_port": settings.smtp_port,
                "smtp_user": settings.smtp_user,
                "smtp_password": settings.smtp_password,
                "smtp_from": settings.smtp_from,
            }),
        )

        email_sent = "✅" in result
        elapsed = time.time() - start_time
        logger.info("EMAIL SENDER complete in %.2fs | sent=%s", elapsed, email_sent)

        return {
            "email_sent": email_sent,
            "final_itinerary": state.get("final_itinerary"),
            "current_node": "email_sender",
            "node_statuses": {**state.get("node_statuses", {}), "email_sender": AgentStatusInfo.COMPLETED},
            "node_outputs": {**state.get("node_outputs", {}), "email_sender": result[:100]},
            "progress_pct": 100,
            "node_timings": {**state.get("node_timings", {}), "email_sender": elapsed},
        }

    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("Email send failed: %s", exc)
        return {
            "email_sent": False,
            "errors": list(state.get("errors", [])) + [f"email: {exc}"],
        }
