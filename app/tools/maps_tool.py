"""
app/tools/maps_tool.py
───────────────────────
Maps and location tool using Nominatim (OpenStreetMap — completely free).

WHY NOMINATIM:
  ✓ Free, no API key needed
  ✓ Global coverage
  ✓ Returns structured address data

  Production: Google Maps Platform ($200/month free credit)
  or Mapbox (50,000 requests/month free).

RATE LIMITING: Nominatim requires 1 request/second max.
  Production: implement a token bucket rate limiter.
  Here: simple sleep(0.5) between requests.
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.parse
import urllib.request

from langchain_core.tools import tool

_maps_cache: dict = {}
_CACHE_TTL = 86400  # 24 hours — location data is stable


@tool
def get_location_info(place_name: str) -> str:
    """Get detailed location information for a place using OpenStreetMap (free).

    Use this to get:
    - Exact coordinates (lat/lon) for weather API calls
    - Country and region context
    - Address details for navigation planning

    Args:
        place_name: Place, city, landmark, or address to look up

    Returns:
        Location details including coordinates, country, and address.
    """
    cache_key = f"location:{place_name.lower()}"
    if cache_key in _maps_cache:
        cached_at, data = _maps_cache[cache_key]
        if time.time() - cached_at < _CACHE_TTL:
            return data

    try:
        url = (
            f"https://nominatim.openstreetmap.org/search?"
            f"q={urllib.parse.quote(place_name)}"
            f"&format=json&limit=3&addressdetails=1"
        )
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "AgenticTravelPlanner/3.0 (educational project)"},
        )
        time.sleep(0.5)  # Nominatim rate limit: 1 req/sec
        with urllib.request.urlopen(req, timeout=8) as resp:
            results = json.loads(resp.read())

        if not results:
            return f"⚠️ Location '{place_name}' not found in OpenStreetMap."

        top = results[0]
        addr = top.get("address", {})
        country = addr.get("country", "Unknown")
        state = addr.get("state", addr.get("region", ""))
        city = addr.get("city", addr.get("town", addr.get("village", "")))

        # Collect nearby alternatives
        alternatives = []
        for r in results[1:3]:
            alt_country = r.get("address", {}).get("country", "")
            if alt_country and alt_country != country:
                alternatives.append(f"{r['display_name'][:60]}")

        result = f"""## 📍 Location: {place_name}
- **Country:** {country}
- **State/Region:** {state}
- **City:** {city}
- **Coordinates:** {float(top['lat']):.4f}°N, {float(top['lon']):.4f}°E
- **OSM Display Name:** {top['display_name'][:100]}
"""
        if alternatives:
            result += f"- **Disambiguation:** Also matches: {' | '.join(alternatives)}\n"

        _maps_cache[cache_key] = (time.time(), result)
        return result

    except asyncio.CancelledError:
        raise

    except Exception as exc:
        return f"⚠️ Location lookup failed for '{place_name}': {exc}"
