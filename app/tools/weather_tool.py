"""
app/tools/weather_tool.py
──────────────────────────
Real-time weather tool using Open-Meteo API.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY OPEN-METEO:
  ✓ Completely FREE — no API key needed
  ✓ Professional-grade data (ECMWF model)
  ✓ 7-day forecast + historical data
  ✓ Handles any lat/lon worldwide

  Production alternative: OpenWeatherMap (free tier: 1000 calls/day)
  or Tomorrow.io (free tier: 500 calls/day) for better accuracy.

TOOL CALLING CONCEPT:
  The @tool decorator transforms a Python function into a LangChain tool.
  The LLM sees the function signature + docstring and decides:
    1. When to call it (based on docstring description)
    2. What arguments to pass (based on type hints + docstring)
    3. How to use the result (based on return description)

  This is "function calling" — the LLM outputs JSON arguments,
  the runtime executes the function, result goes back to LLM.

  Production alternative: Tavily, Serper for real-time web search.

CACHING STRATEGY:
  Weather data changes hourly. Cache with TTL=3600s (1 hour).
  Cache key: f"weather:{lat:.2f}:{lon:.2f}:{days}"
  We use a simple in-memory dict here. Production → Redis with TTL.

ERROR HANDLING:
  Network failures should NEVER crash the agent loop.
  Return descriptive error string so the LLM can say
  "weather data unavailable — check weather.com before traveling"
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.request
from typing import Optional

from langchain_core.tools import tool

# ── Simple in-memory cache (TTL-based) ────────────────────────────────────────
# Production: replace with Redis client → cache.set(key, value, ex=3600)
_weather_cache: dict = {}
_CACHE_TTL = 3600  # 1 hour

# ── Geocoding API (Nominatim / OpenStreetMap — free) ──────────────────────────
_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def _get_coordinates(city: str) -> tuple[float, float] | None:
    """Geocode a city name to lat/lon using Nominatim (free OSM service)."""
    cache_key = f"geo:{city.lower()}"
    if cache_key in _weather_cache:
        return _weather_cache[cache_key]

    try:
        url = (
            f"{_NOMINATIM_URL}?q={urllib.parse.quote(city)}"
            f"&format=json&limit=1"
        )
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "AgenticTravelPlanner/3.0 (educational)"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        if data:
            coords = (float(data[0]["lat"]), float(data[0]["lon"]))
            _weather_cache[cache_key] = coords
            return coords
    except Exception:
        pass
    return None


import urllib.parse


@tool
def get_weather_forecast(destination: str, days: int = 7) -> str:
    """Get real weather forecast for a travel destination using Open-Meteo (free, no API key).

    Use this tool ALWAYS when planning any trip — weather affects which
    activities are feasible, what clothing to pack, and when to visit.

    Args:
        destination: City/country name (e.g., "Tokyo, Japan", "Bali, Indonesia")
        days: Number of days to forecast (1-16, default 7)

    Returns:
        Weather summary with temperature, precipitation, and activity recommendations.
    """
    # ── Cache check ───────────────────────────────────────────────────────────
    cache_key = f"weather:{destination}:{days}"
    if cache_key in _weather_cache:
        cached_at, data = _weather_cache[cache_key]
        if time.time() - cached_at < _CACHE_TTL:
            return data  # Cache hit — saves API call + latency

    try:
        # Step 1: Geocode destination to lat/lon
        coords = _get_coordinates(destination)
        if not coords:
            return (
                f"⚠️ Could not geocode '{destination}'. "
                "Weather data unavailable — check weather.com for forecasts."
            )
        lat, lon = coords

        # Step 2: Fetch weather from Open-Meteo
        days_clamped = min(max(days, 1), 16)
        url = (
            f"{_OPEN_METEO_URL}?"
            f"latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
            f"weathercode,windspeed_10m_max,uv_index_max"
            f"&hourly=relativehumidity_2m"
            f"&forecast_days={days_clamped}"
            f"&timezone=auto"
            f"&temperature_unit=celsius"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "AgenticTravelPlanner/3.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            weather = json.loads(resp.read())

        # Step 3: Parse and format
        daily = weather.get("daily", {})
        dates = daily.get("time", [])
        max_temps = daily.get("temperature_2m_max", [])
        min_temps = daily.get("temperature_2m_min", [])
        precip = daily.get("precipitation_sum", [])
        uv = daily.get("uv_index_max", [])
        wind = daily.get("windspeed_10m_max", [])
        wcodes = daily.get("weathercode", [])

        # WMO Weather Code → description mapping
        wmo_desc = {
            0: "☀️ Clear sky", 1: "🌤 Mainly clear", 2: "⛅ Partly cloudy",
            3: "☁️ Overcast", 45: "🌫 Foggy", 51: "🌦 Light drizzle",
            61: "🌧 Light rain", 63: "🌧 Moderate rain", 71: "❄️ Light snow",
            80: "🌦 Rain showers", 95: "⛈ Thunderstorm",
        }

        lines = [f"## 🌤 Weather Forecast: {destination}\n"]
        avg_max = sum(max_temps) / len(max_temps) if max_temps else 0
        avg_min = sum(min_temps) / len(min_temps) if min_temps else 0
        total_rain = sum(precip) if precip else 0

        lines.append(f"**Overview:** Avg {avg_min:.0f}–{avg_max:.0f}°C | Total rain: {total_rain:.0f}mm\n")
        lines.append("| Date | Conditions | Temp (°C) | Rain | Wind | UV |\n")
        lines.append("|------|------------|-----------|------|------|----|\n")

        for i, date in enumerate(dates[:min(days, 7)]):
            wcode = wcodes[i] if i < len(wcodes) else 0
            desc = wmo_desc.get(wcode, "🌥 Cloudy")
            t_max = f"{max_temps[i]:.0f}" if i < len(max_temps) else "N/A"
            t_min = f"{min_temps[i]:.0f}" if i < len(min_temps) else "N/A"
            rain = f"{precip[i]:.1f}mm" if i < len(precip) else "N/A"
            uv_val = f"{uv[i]:.0f}" if i < len(uv) else "N/A"
            wind_val = f"{wind[i]:.0f}km/h" if i < len(wind) else "N/A"
            lines.append(f"| {date} | {desc} | {t_min}–{t_max} | {rain} | {wind_val} | {uv_val} |\n")

        # Activity recommendations based on weather
        lines.append("\n**🎒 Activity Recommendations:**\n")
        if avg_max > 30:
            lines.append("- 🌡️ Hot weather: schedule outdoor activities early morning or late afternoon\n")
            lines.append("- 💧 Stay hydrated, use sunscreen (UV may be high)\n")
        if avg_max < 10:
            lines.append("- 🧥 Cold weather: pack layers, thermal underwear recommended\n")
        if total_rain > 50:
            lines.append("- ☔ Significant rain expected: plan indoor alternatives for each day\n")
            lines.append("- 🌂 Pack a compact umbrella or rain jacket\n")
        if avg_max >= 20 and avg_max <= 28 and total_rain < 30:
            lines.append("- ✅ Excellent weather conditions — ideal for outdoor sightseeing\n")

        result = "".join(lines)
        _weather_cache[cache_key] = (time.time(), result)
        return result

    except asyncio.CancelledError:
        # NEVER catch CancelledError — always re-raise
        # This ensures Ctrl+C properly propagates
        raise

    except Exception as exc:
        return (
            f"⚠️ Weather data unavailable for '{destination}': {exc}\n"
            "Please check weather.com or Google Weather before traveling."
        )
