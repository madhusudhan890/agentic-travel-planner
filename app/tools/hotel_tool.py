"""
app/tools/hotel_tool.py
────────────────────────
Hotel search tool — realistic simulation with production API path.

Production alternatives:
  - Booking.com Affiliate API (requires partner application)
  - Hotels.com Rapid API ($10/month via RapidAPI)
  - Amadeus Hotel Search API (free sandbox)
  - TripAdvisor Content API (free tier available)
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import time

from langchain_core.tools import tool

_hotel_cache: dict = {}
_CACHE_TTL = 3600 * 6  # 6 hours

# Hotel data per destination category
_HOTEL_TEMPLATES = {
    "budget": [
        {"name": "{city} Backpackers Hostel", "type": "Hostel", "rating": 4.1},
        {"name": "Budget Inn {city}", "type": "Guesthouse", "rating": 3.8},
        {"name": "{city} Budget Hotel", "type": "Hotel", "rating": 3.9},
    ],
    "midrange": [
        {"name": "Hotel {city} Central", "type": "Boutique Hotel", "rating": 4.3},
        {"name": "{city} City Suites", "type": "Apart-hotel", "rating": 4.2},
        {"name": "The {city} Lodge", "type": "Hotel", "rating": 4.4},
    ],
    "luxury": [
        {"name": "Grand {city} Palace", "type": "5-Star Resort", "rating": 4.8},
        {"name": "The Ritz {city}", "type": "Luxury Hotel", "rating": 4.9},
        {"name": "{city} Marriott", "type": "Business Luxury", "rating": 4.6},
    ],
}

_PRICE_RANGES = {
    "asia": {"budget": (10, 35), "midrange": (40, 120), "luxury": (150, 500)},
    "europe": {"budget": (25, 60), "midrange": (80, 200), "luxury": (250, 800)},
    "americas": {"budget": (20, 55), "midrange": (70, 180), "luxury": (200, 700)},
    "oceania": {"budget": (30, 70), "midrange": (100, 220), "luxury": (280, 900)},
    "africa": {"budget": (15, 45), "midrange": (50, 150), "luxury": (180, 600)},
    "default": {"budget": (20, 50), "midrange": (60, 160), "luxury": (200, 600)},
}

_ASIA_DESTINATIONS = ["tokyo", "kyoto", "bangkok", "bali", "singapore", "mumbai", "delhi", "beijing", "seoul", "vietnam", "cambodia", "thailand", "japan", "india", "china"]
_EUROPE_DESTINATIONS = ["paris", "london", "rome", "barcelona", "amsterdam", "berlin", "prague", "vienna", "lisbon", "greece", "italy", "france", "spain"]


def _get_region(destination: str) -> str:
    dest_lower = destination.lower()
    if any(d in dest_lower for d in _ASIA_DESTINATIONS):
        return "asia"
    if any(d in dest_lower for d in _EUROPE_DESTINATIONS):
        return "europe"
    return "default"


@tool
def search_hotels(
    destination: str,
    check_in_month: str,
    num_nights: int = 5,
    num_travelers: int = 1,
    budget_level: str = "moderate",
) -> str:
    """Search for hotel accommodation options in a travel destination.

    Use this tool for every trip to provide accommodation recommendations.
    Always call before generating the itinerary so real options are included.

    PRODUCTION NOTE: Returns realistic simulated data.
    Replace with Booking.com API or Amadeus Hotel Search for real prices.
    Free sandbox: developers.amadeus.com/self-service/apis-docs/hotel-search

    Args:
        destination: City/region name (e.g., "Kyoto, Japan", "Bali, Indonesia")
        check_in_month: Month of travel for seasonal pricing
        num_nights: Number of nights
        num_travelers: Number of travelers (affects room configuration)
        budget_level: "budget", "moderate", or "luxury"

    Returns:
        Formatted hotel options across budget tiers with prices, amenities, location.
    """
    cache_key = f"hotel:{destination}:{check_in_month}:{budget_level}"
    if cache_key in _hotel_cache:
        cached_at, data = _hotel_cache[cache_key]
        if time.time() - cached_at < _CACHE_TTL:
            return data

    try:
        region = _get_region(destination)
        prices = _PRICE_RANGES.get(region, _PRICE_RANGES["default"])
        seed = hashlib.md5(f"{destination}-{check_in_month}".encode()).hexdigest()
        rng = random.Random(int(seed[:8], 16))

        city_name = destination.split(",")[0].strip()
        peak_months = ["december", "january", "june", "july", "august"]
        peak_mult = 1.3 if check_in_month.lower() in peak_months else 1.0

        lines = [f"## 🏨 Hotel Options: {destination}\n"]
        lines.append(f"**{num_nights} nights | {check_in_month} | {num_travelers} traveler(s)**\n\n")

        for tier in ["budget", "midrange", "luxury"]:
            tier_label = {"budget": "💰 Budget", "midrange": "⭐ Mid-Range", "luxury": "✨ Luxury"}[tier]
            p_min, p_max = prices[tier]
            nightly = rng.randint(p_min, p_max)
            nightly_peak = int(nightly * peak_mult)
            total = nightly_peak * num_nights

            template = rng.choice(_HOTEL_TEMPLATES[tier])
            hotel_name = template["name"].format(city=city_name)
            amenities = {
                "budget": "WiFi, 24h reception, lockers, communal kitchen",
                "midrange": "WiFi, breakfast optional, gym, city view rooms",
                "luxury": "WiFi, breakfast included, pool, spa, concierge, room service",
            }[tier]
            locations = {
                "budget": "5-10 min walk to city center / public transport",
                "midrange": "City center or 2-3 stops on metro",
                "luxury": "Prime central location or premium district",
            }[tier]

            lines.append(f"### {tier_label}: {hotel_name}\n")
            lines.append(f"- **Type:** {template['type']} | **Rating:** {template['rating']}/5.0\n")
            lines.append(f"- **Price:** ${nightly_peak}/night ≈ **${total} total** ({num_nights} nights)\n")
            lines.append(f"- **Location:** {locations}\n")
            lines.append(f"- **Amenities:** {amenities}\n")
            lines.append(f"- **Book at:** Booking.com | Airbnb | Hotels.com\n\n")

        lines.append("> ⚠️ **Simulated data for planning.** Verify on:\n")
        lines.append("> [Booking.com](https://booking.com) | [Agoda](https://agoda.com) | [Airbnb](https://airbnb.com)\n")

        result = "".join(lines)
        _hotel_cache[cache_key] = (time.time(), result)
        return result

    except asyncio.CancelledError:
        raise

    except Exception as exc:
        return f"⚠️ Hotel search failed: {exc}. Check Booking.com for options."
