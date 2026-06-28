"""
app/tools/flight_tool.py
─────────────────────────
Flight search tool — realistic simulation with production API path.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY SIMULATION (not a real API):
  Real flight APIs (Amadeus, Skyscanner, Kiwi.com) either:
  1. Require paid subscription (Amadeus: $199+/month)
  2. Require business verification (Skyscanner Partner)
  3. Are deprecated/unreliable (many free ones shut down)

  PRODUCTION PATH:
    → Amadeus Flight Offers Search API (free sandbox available at developers.amadeus.com)
    → Kiwi.com Tequila API (free tier: 1000 searches/month)
    → Google Flights via SerpAPI ($50/month)

  The simulation generates REALISTIC price ranges based on:
  - Real route distance categories
  - Real seasonal pricing patterns
  - Real airline tier pricing

ERROR PATTERN:
  Always return a string, never raise — the LLM handles partial failures
  gracefully ("flight data unavailable — recommend checking Google Flights").
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from typing import Optional

from langchain_core.tools import tool

# ── In-memory cache (production: Redis with 24h TTL for flight data) ──────────
_flight_cache: dict = {}
_CACHE_TTL = 86400  # 24 hours — flight prices valid for a day


def _generate_realistic_price(
    origin: str, destination: str, month: str, days: int
) -> dict:
    """
    Generate realistic flight price ranges based on route characteristics.

    WHY: Simulated data must be realistic to be educationally valuable.
    Real Amadeus API would return the same structure.

    Production: Replace this function body with Amadeus API call.
    The function signature stays the same — only the data source changes.
    This is the "strangler fig" pattern for replacing simulated with real.
    """
    # Deterministic "randomness" based on route (same route → same prices)
    seed = hashlib.md5(f"{origin}-{destination}-{month}".encode()).hexdigest()
    rng = random.Random(int(seed[:8], 16))

    # Categorize route distance (simplified)
    domestic_origins = ["mumbai", "delhi", "bangalore", "chennai", "kolkata"]
    is_domestic = (
        any(d in origin.lower() for d in domestic_origins) and
        any(d in destination.lower() for d in domestic_origins)
    )

    # Regional routes (Asia-Asia, Europe-Europe)
    is_regional = any(r in destination.lower() for r in [
        "dubai", "singapore", "bangkok", "kuala lumpur", "hong kong",
        "paris", "london", "rome", "berlin", "amsterdam",
    ])

    if is_domestic:
        base_min, base_max = 50, 200
        duration_hr = rng.uniform(1, 3)
        airlines = ["IndiGo", "Air India", "SpiceJet", "Vistara", "GoAir"]
    elif is_regional:
        base_min, base_max = 200, 600
        duration_hr = rng.uniform(4, 10)
        airlines = ["Emirates", "Qatar Airways", "Singapore Airlines", "Thai Airways", "Air Asia"]
    else:
        # Long-haul international
        base_min, base_max = 500, 1800
        duration_hr = rng.uniform(10, 18)
        airlines = ["Emirates", "Lufthansa", "British Airways", "Air France", "Cathay Pacific", "ANA"]

    # Seasonal adjustments
    peak_months = ["december", "january", "june", "july", "august"]
    if month.lower() in peak_months:
        base_min = int(base_min * 1.4)
        base_max = int(base_max * 1.4)

    price_min = rng.randint(base_min, base_min + (base_max - base_min) // 2)
    price_max = rng.randint(price_min + 50, base_max)

    selected_airlines = rng.sample(airlines, min(3, len(airlines)))

    return {
        "economy_min": price_min,
        "economy_max": price_max,
        "business_min": price_min * 3,
        "business_max": price_max * 4,
        "duration_hours": round(duration_hr, 1),
        "airlines": selected_airlines,
        "stops": 0 if is_domestic else (0 if is_regional and rng.random() > 0.4 else 1),
        "booking_tip": "Book 6-8 weeks in advance for best economy prices",
    }


@tool
def search_flights(
    origin_city: str,
    destination_city: str,
    travel_month: str,
    num_travelers: int = 1,
    trip_days: int = 7,
) -> str:
    """Search for flight options between two cities with price estimates.

    Use this tool when the user has provided a departure city and wants
    international or domestic flights included in their itinerary.

    PRODUCTION NOTE: Currently returns realistic simulated data.
    Replace with Amadeus API (free sandbox at developers.amadeus.com)
    for real prices. The output format remains identical.

    Args:
        origin_city: Departure city (e.g., "Mumbai", "London", "New York")
        destination_city: Destination city (e.g., "Tokyo", "Paris", "Bali")
        travel_month: Month of travel (e.g., "July", "December")
        num_travelers: Number of travelers (affects total cost)
        trip_days: Number of days (for round-trip calculation)

    Returns:
        Formatted flight options with price ranges, airlines, and booking tips.
    """
    if not origin_city or origin_city.strip() == "":
        return (
            "⚠️ No departure city provided. "
            "Assuming nearest major international airport. "
            "Check Google Flights for accurate prices."
        )

    # ── Cache check ───────────────────────────────────────────────────────────
    cache_key = f"flight:{origin_city}:{destination_city}:{travel_month}"
    if cache_key in _flight_cache:
        cached_at, data = _flight_cache[cache_key]
        if time.time() - cached_at < _CACHE_TTL:
            return data

    try:
        prices = _generate_realistic_price(
            origin_city, destination_city, travel_month, trip_days
        )

        total_eco_min = prices["economy_min"] * num_travelers
        total_eco_max = prices["economy_max"] * num_travelers
        total_biz_min = prices["business_min"] * num_travelers
        airlines_str = ", ".join(prices["airlines"])
        stops = "Direct" if prices["stops"] == 0 else f"{prices['stops']} stop"
        duration = prices["duration_hours"]

        result = f"""## ✈️ Flight Options: {origin_city} → {destination_city}
**Month:** {travel_month} | **Travelers:** {num_travelers}

| Class | Per Person | Total ({num_travelers} pax) | Duration | Stops |
|-------|-----------|---------------------------|----------|-------|
| Economy | ${prices['economy_min']}–${prices['economy_max']} | ${total_eco_min}–${total_eco_max} | {duration}h | {stops} |
| Business | ${prices['business_min']}–${prices['business_max']} | ${total_biz_min}+ | {duration}h | {stops} |

**Airlines operating this route:** {airlines_str}

**💡 Booking Tips:**
- {prices['booking_tip']}
- Use Google Flights price alerts for this route
- Check {', '.join(prices['airlines'][:2])} direct for occasional flash sales
- Flexible dates (±3 days) can save 15-30%

> ⚠️ **Simulated data for planning.** Verify current prices on:
> [Google Flights](https://flights.google.com) | [Skyscanner](https://skyscanner.com)

**Production API:** Amadeus Flight Offers Search (free sandbox at developers.amadeus.com)
"""
        _flight_cache[cache_key] = (time.time(), result)
        return result

    except asyncio.CancelledError:
        raise  # Always re-raise — never swallow CancelledError

    except Exception as exc:
        return (
            f"⚠️ Flight search failed: {exc}\n"
            "Check Google Flights for current prices."
        )
