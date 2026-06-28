"""
app/tools/registry.py
──────────────────────
Centralized tool registry — categorized by agent domain.

ARCHITECTURE DECISION: Categorized Registry
  Phase 2 had a flat list of 2 tools.
  Phase 3 has 10+ tools across 7 agent domains.
  
  WHY CATEGORIZE:
  1. Each specialist agent gets ONLY its relevant tools
     (flight agent shouldn't call email_tool)
  2. Reduces token usage per agent call
     (fewer tool definitions in context = shorter prompts)
  3. Prevents wrong tool calls (LLM can't call tools it doesn't have)
  4. Security: RAG tool shouldn't be accessible to untrusted agents

  PRODUCTION PATTERN: Tool registries often use dependency injection
  (tools with API clients) rather than module-level imports.
  Here we keep it simple — tools initialize lazily on first call.
"""

from __future__ import annotations

from typing import List

from langchain_core.tools import BaseTool

from app.core.logging import get_logger

# ── Import all tools ──────────────────────────────────────────────────────────
from app.tools.wikipedia_tool import get_destination_info
from app.tools.currency_tool import get_exchange_rate
from app.tools.weather_tool import get_weather_forecast
from app.tools.flight_tool import search_flights
from app.tools.hotel_tool import search_hotels
from app.tools.maps_tool import get_location_info
from app.tools.restaurant_tool import search_restaurants
from app.tools.visa_tool import get_visa_requirements
from app.tools.calculator_tool import calculate_trip_budget, convert_currency_amount
from app.tools.pdf_tool import generate_pdf_itinerary
from app.tools.email_tool import send_itinerary_email

logger = get_logger(__name__)


# ── Tool Sets by Agent Domain ─────────────────────────────────────────────────
# WHY: Each agent gets a focused toolset. This mirrors how human specialists
# work — a flight agent doesn't need restaurant recommendations tools.

PLANNER_TOOLS: List[BaseTool] = [
    get_destination_info,      # Wikipedia context for any destination
    get_location_info,         # Coordinates + country info
]

FLIGHT_TOOLS: List[BaseTool] = [
    search_flights,            # Flight search (simulated, Amadeus path)
    get_location_info,         # Airport + city location lookup
    calculate_trip_budget,     # Flight budget calculation
]

HOTEL_TOOLS: List[BaseTool] = [
    search_hotels,             # Hotel search (simulated, Booking.com path)
    get_location_info,         # Neighborhood information
    calculate_trip_budget,     # Accommodation budget calculation
]

WEATHER_TOOLS: List[BaseTool] = [
    get_weather_forecast,      # Open-Meteo (free, no API key)
    get_location_info,         # Geocoding for weather API
]

BUDGET_TOOLS: List[BaseTool] = [
    calculate_trip_budget,     # Full budget breakdown
    get_exchange_rate,         # Frankfurter API (free)
    convert_currency_amount,   # Currency math
]

RESTAURANT_TOOLS: List[BaseTool] = [
    search_restaurants,        # Restaurant recommendations
    get_destination_info,      # Culinary context from Wikipedia
    get_location_info,         # Food district info
]

VISA_TOOLS: List[BaseTool] = [
    get_visa_requirements,     # Visa requirements lookup
    get_destination_info,      # Country context
]

CURRENCY_TOOLS: List[BaseTool] = [
    get_exchange_rate,         # Frankfurter API (free, real-time rates)
    convert_currency_amount,   # Budget conversion math
    calculate_trip_budget,     # Full budget in local currency
]

WRITER_TOOLS: List[BaseTool] = [
    # Writer has NO tools — it synthesizes from provided data
    # WHY: Writer's job is synthesis, not research. Giving it tools
    # could cause it to re-run research instead of using agent outputs.
]

OUTPUT_TOOLS: List[BaseTool] = [
    generate_pdf_itinerary,    # ReportLab PDF generation
    send_itinerary_email,      # SMTP email delivery
]

# ── Master flat list (Phase 2 backward compatibility) ─────────────────────────
_ALL_TOOLS: List[BaseTool] = [
    get_destination_info,
    get_exchange_rate,
    get_weather_forecast,
    search_flights,
    search_hotels,
    get_location_info,
    search_restaurants,
    get_visa_requirements,
    calculate_trip_budget,
    convert_currency_amount,
]


def get_tools() -> List[BaseTool]:
    """Return ALL tools — used for backward-compatible Phase 2 agent."""
    logger.info("Loaded %d tools (all)", len(_ALL_TOOLS))
    return _ALL_TOOLS


def get_tools_for_agent(agent_name: str) -> List[BaseTool]:
    """Return the appropriate tool subset for a named specialist agent.

    This is the preferred method for Phase 3 multi-agent setup.
    Each agent gets only the tools it needs — reducing token usage
    and preventing unintended tool calls.

    Args:
        agent_name: One of: planner, flight, hotel, weather, budget,
                   restaurant, visa, currency, writer, output

    Returns:
        List of tools for the specified agent.
    """
    tool_map = {
        "planner": PLANNER_TOOLS,
        "flight": FLIGHT_TOOLS,
        "hotel": HOTEL_TOOLS,
        "weather": WEATHER_TOOLS,
        "budget": BUDGET_TOOLS,
        "restaurant": RESTAURANT_TOOLS,
        "visa": VISA_TOOLS,
        "currency": CURRENCY_TOOLS,
        "writer": WRITER_TOOLS,
        "output": OUTPUT_TOOLS,
    }
    tools = tool_map.get(agent_name, _ALL_TOOLS)
    logger.debug(
        "Tools for agent=%s: %s",
        agent_name,
        [t.name for t in tools],
    )
    return tools
