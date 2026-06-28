"""
app/tools/restaurant_tool.py
─────────────────────────────
Restaurant discovery using Wikipedia + simulation.

Production alternatives:
  - Foursquare Places API (free: 100k calls/day!)
  - Yelp Fusion API (free: 500 calls/day)
  - Google Places API (first $200/month free)

TEACHING POINT — Tool Design:
  Good tools return STRUCTURED data that the LLM can reference specifically.
  Bad tools return vague descriptions the LLM must interpret.
  
  Good: "Sukiyabashi Jiro: 3-star Michelin, ¥35,000/person, requires 2-month advance reservation"
  Bad: "There are many good sushi restaurants in Tokyo"
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import time

from langchain_core.tools import tool

_restaurant_cache: dict = {}
_CACHE_TTL = 3600 * 12  # 12 hours

# Curated restaurant templates by cuisine region
_CUISINE_PROFILES = {
    "japan": {
        "street": ["Tsukemen ramen stalls", "Yakitori alley (Yurakucho)", "Conveyor belt sushi", "Takoyaki street stalls"],
        "casual": ["Ramen-ya (ramen shop)", "Izakaya (Japanese pub)", "Katsu-don restaurant", "Udon noodle house"],
        "upscale": ["Kaiseki (multi-course)", "Omakase sushi counter", "Teppanyaki restaurant", "Traditional ryotei"],
        "budget_range": "$5-15",
        "mid_range": "$20-60",
        "fine_range": "$80-300+",
        "tip": "Tipping is NOT customary in Japan — it can be considered rude",
    },
    "france": {
        "street": ["Boulangerie croissants & baguettes", "Crêpe stands", "Market fresh produce"],
        "casual": ["Bistro du quartier", "Brasserie", "Café lunch (plat du jour)"],
        "upscale": ["Michelin-starred restaurant", "Classic French fine dining", "Wine bar with charcuterie"],
        "budget_range": "€8-20",
        "mid_range": "€25-60",
        "fine_range": "€80-300+",
        "tip": "Service is included (service compris). Additional 5-10% tip is appreciated",
    },
    "india": {
        "street": ["Dhaba (roadside eatery)", "Chaat stalls", "Thali restaurants", "South Indian breakfast spots"],
        "casual": ["Pure vegetarian restaurant", "North Indian curry house", "Biryani specialty"],
        "upscale": ["Rooftop restaurant", "Palace hotel restaurant", "Modern Indian fusion"],
        "budget_range": "₹150-500",
        "mid_range": "₹600-2000",
        "fine_range": "₹2500-8000+",
        "tip": "10-15% tip appreciated at sit-down restaurants",
    },
    "thailand": {
        "street": ["Night market (Chatuchak, etc.)", "Pad Thai stalls", "Som tam (papaya salad) carts", "Mango sticky rice stands"],
        "casual": ["Local shophouse restaurant", "Thai seafood BBQ", "Khao man gai (chicken rice)"],
        "upscale": ["Riverside fine dining", "Rooftop Thai restaurant", "Royal Thai cuisine"],
        "budget_range": "฿50-200 ($1.5-6)",
        "mid_range": "฿300-800 ($9-24)",
        "fine_range": "฿1000-3000+ ($30-90)",
        "tip": "10% tip appreciated at restaurants, not expected at street stalls",
    },
    "default": {
        "street": ["Local food market", "Street food stalls", "Food court"],
        "casual": ["Local restaurant", "Casual dining", "Traditional eatery"],
        "upscale": ["Fine dining", "Hotel restaurant", "Chef's table experience"],
        "budget_range": "$5-20",
        "mid_range": "$25-60",
        "fine_range": "$80-200+",
        "tip": "Check local customs for tipping",
    },
}


def _get_cuisine_profile(destination: str) -> dict:
    dest = destination.lower()
    if any(k in dest for k in ["japan", "tokyo", "kyoto", "osaka"]):
        return _CUISINE_PROFILES["japan"]
    if any(k in dest for k in ["france", "paris", "lyon", "nice"]):
        return _CUISINE_PROFILES["france"]
    if any(k in dest for k in ["india", "mumbai", "delhi", "bangalore", "goa", "rajasthan"]):
        return _CUISINE_PROFILES["india"]
    if any(k in dest for k in ["thailand", "bangkok", "phuket", "chiang mai"]):
        return _CUISINE_PROFILES["thailand"]
    return _CUISINE_PROFILES["default"]


@tool
def search_restaurants(
    destination: str,
    interests: str = "local cuisine",
    budget_per_meal: str = "moderate",
    days: int = 5,
) -> str:
    """Find restaurant recommendations and food experiences for a travel destination.

    Use this to provide dining recommendations that match traveler interests and budget.
    Always call this for trips with food interests or when planning daily meals.

    PRODUCTION PATH: Foursquare Places API (free 100k calls/day)
    Sign up at foursquare.com/developer

    Args:
        destination: City/country name
        interests: Food preferences (e.g., "vegetarian", "street food", "seafood")
        budget_per_meal: "budget", "moderate", or "luxury"
        days: Number of days (determines how many suggestions needed)

    Returns:
        Curated restaurant recommendations with prices, types, and local tips.
    """
    cache_key = f"restaurant:{destination}:{budget_per_meal}"
    if cache_key in _restaurant_cache:
        cached_at, data = _restaurant_cache[cache_key]
        if time.time() - cached_at < _CACHE_TTL:
            return data

    try:
        profile = _get_cuisine_profile(destination)
        seed = hashlib.md5(f"{destination}".encode()).hexdigest()
        rng = random.Random(int(seed[:8], 16))

        lines = [f"## 🍜 Food & Restaurant Guide: {destination}\n\n"]

        # Street food
        lines.append("### 🥢 Street Food & Markets (Must-Try)\n")
        for item in rng.sample(profile["street"], min(3, len(profile["street"]))):
            lines.append(f"- {item}\n")
        lines.append(f"- **Typical cost:** {profile['budget_range']} per meal\n\n")

        # Casual dining
        lines.append("### 🍽️ Casual Restaurants (Daily Dining)\n")
        for item in rng.sample(profile["casual"], min(3, len(profile["casual"]))):
            lines.append(f"- {item}\n")
        lines.append(f"- **Typical cost:** {profile['mid_range']} per person\n\n")

        # Upscale
        lines.append("### ✨ Upscale Dining (Special Occasion)\n")
        for item in rng.sample(profile["upscale"], min(2, len(profile["upscale"]))):
            lines.append(f"- {item}\n")
        lines.append(f"- **Typical cost:** {profile['fine_range']} per person\n")
        lines.append("- **Tip:** Book 2-4 weeks in advance for popular upscale restaurants\n\n")

        # Budget estimate
        lines.append("### 💰 Daily Food Budget Estimate\n")
        lines.append(f"- **Budget traveler:** 3 meals at street food prices = {profile['budget_range']} × 3\n")
        lines.append(f"- **Moderate:** Mix of street food + casual = {profile['mid_range']}/day\n")
        lines.append(f"- **Foodie experience:** 1 upscale + 2 casual = {profile['fine_range']} (incl. 1 special)\n\n")

        # Tipping
        lines.append(f"### 💡 Tipping Culture\n- {profile['tip']}\n\n")

        # Interests-based note
        if "vegetarian" in interests.lower() or "vegan" in interests.lower():
            lines.append("### 🌱 Vegetarian/Vegan Note\n")
            lines.append("- Search HappyCow (happycow.net) for plant-based restaurants in any destination\n")
            lines.append("- Always specify dietary restrictions when ordering\n\n")

        lines.append("> **Production:** Real recommendations via Foursquare API (free 100k calls/day)\n")
        lines.append("> at [foursquare.com/developer](https://foursquare.com/developer)\n")

        result = "".join(lines)
        _restaurant_cache[cache_key] = (time.time(), result)
        return result

    except asyncio.CancelledError:
        raise

    except Exception as exc:
        return f"⚠️ Restaurant search failed: {exc}"
