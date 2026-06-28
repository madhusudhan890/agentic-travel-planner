"""
app/tools/visa_tool.py
───────────────────────
Visa requirements lookup using Wikipedia and curated data.

WHY THIS EXISTS:
  Missing visa information is one of the most expensive travel mistakes.
  An AI planner that doesn't mention visa requirements is dangerous.
  This tool ensures visa info is always surfaced.

  Production: IATA Travel Centre API (industry standard, requires subscription)
  or Sherpa (sherpa.io — free API for visa requirements).
  Sherpa offers a free tier at developers.sherpa.io — excellent for production.
"""

from __future__ import annotations

import asyncio
import time

import wikipedia
from langchain_core.tools import tool

_visa_cache: dict = {}
_CACHE_TTL = 86400 * 7  # 7 days — visa rules change infrequently


# Curated visa-free access data (simplified, for common destinations)
# Production: Replace with Sherpa API or IATA TravelCentre
_COMMON_VISA_INFO = {
    "japan": {
        "visa_free": ["USA", "UK", "EU", "Canada", "Australia", "Singapore", "South Korea"],
        "e_visa": ["India (eVisa available for some)"],
        "visa_required": ["Pakistan", "Afghanistan"],
        "general": "Most Western passport holders get 90-day visa-free entry. Indian passport holders need visa.",
        "cost": "Free for eligible passports | ¥3,000-6,000 for visa required",
        "processing": "1-3 business days for e-visa | 1-2 weeks for sticker visa",
        "official": "https://www.mofa.go.jp/j_info/visit/visa/index.html",
        "requirements": "Valid passport (6+ months validity), return ticket, hotel confirmation",
    },
    "france": {
        "visa_free": ["USA", "UK (90 days)", "Canada", "Australia"],
        "schengen": True,
        "general": "Part of Schengen Zone. 90 days in 180-day period for most Western passports.",
        "cost": "€80 Schengen visa fee (where required)",
        "processing": "2-3 weeks",
        "official": "https://france-visas.gouv.fr",
        "requirements": "Valid EU/Schengen visa for non-exempt nationalities",
        "etias_2025": "ETIAS required from 2025 for visa-exempt travelers ($7 fee)",
    },
    "thailand": {
        "visa_free": ["USA", "UK", "EU", "Canada", "Australia", "Japan", "South Korea", "India (30 days)"],
        "visa_on_arrival": ["India (certain nationalities)"],
        "general": "Very open visa policy. Most nationalities get 30-60 days on arrival.",
        "cost": "Free visa-free entry | THB 2,000 ($55) for visa on arrival",
        "processing": "Instant on arrival",
        "official": "https://www.thaiembassy.com",
        "requirements": "Passport valid 6+ months, return ticket, proof of funds (THB 20,000/person)",
    },
    "india": {
        "visa_free": [],
        "e_visa": ["USA", "UK", "EU", "Canada", "Australia", "Japan", "South Korea", "Singapore", "most nationalities"],
        "general": "eVisa (e-Tourist Visa) available for 160+ countries. Apply online 4 days before travel.",
        "cost": "$25 (30-day), $40 (1-year), $80 (5-year)",
        "processing": "72 hours online (eVisa)",
        "official": "https://indianvisaonline.gov.in",
        "requirements": "Valid passport, passport photo, return ticket, hotel confirmation",
    },
    "usa": {
        "visa_free": ["UK", "EU (ESTA required)", "Canada", "Japan", "Australia", "South Korea", "Singapore"],
        "esta": "ESTA required for Visa Waiver Program countries ($21, valid 2 years)",
        "general": "Visa-free (with ESTA) for 40 Visa Waiver Program countries. Others need B1/B2 visa.",
        "cost": "ESTA: $21 | B1/B2 visa: $185",
        "processing": "ESTA: instant | B1/B2: 2-12 months (varies by embassy)",
        "official": "https://travel.state.gov",
        "requirements": "ESTA approval required before travel for VWP countries",
    },
    "uk": {
        "visa_free": ["EU (but not Schengen)", "USA", "Canada", "Australia", "Japan"],
        "eta_2024": "UK ETA required from 2024 for non-visa nationals (£10)",
        "general": "Not part of Schengen. Separate UK visitor visa may be needed.",
        "cost": "UK ETA: £10 | Standard visitor visa: £115",
        "processing": "ETA: 72 hours | Visa: 3-4 weeks",
        "official": "https://www.gov.uk/visit-uk",
        "requirements": "ETA or visa depending on nationality",
    },
}


def _find_visa_info(destination: str) -> dict | None:
    dest_lower = destination.lower()
    for key, info in _COMMON_VISA_INFO.items():
        if key in dest_lower:
            return info
    return None


@tool
def get_visa_requirements(destination: str, nationality_hint: str = "general") -> str:
    """Get visa requirements and entry conditions for a travel destination.

    CRITICAL: Always call this for international trips. Visa requirements
    can vary by nationality — always verify on the official government website.

    PRODUCTION PATH: Sherpa API (sherpa.io) — free tier available.
    Returns real-time visa requirements by passport nationality.

    Args:
        destination: Country or city to visit (e.g., "Japan", "France", "Thailand")
        nationality_hint: Passport country if known (e.g., "Indian", "American", "British")

    Returns:
        Visa requirements, costs, processing times, and official links.
    """
    cache_key = f"visa:{destination.lower()}:{nationality_hint.lower()}"
    if cache_key in _visa_cache:
        cached_at, data = _visa_cache[cache_key]
        if time.time() - cached_at < _CACHE_TTL:
            return data

    result_parts = [f"## 🛂 Visa Requirements: {destination}\n\n"]

    # Check curated data first
    info = _find_visa_info(destination)
    if info:
        result_parts.append(f"**Overview:** {info['general']}\n\n")
        if info.get("visa_free"):
            result_parts.append(f"**✅ Visa-Free Nationalities:** {', '.join(info['visa_free'][:8])}\n")
        if info.get("e_visa"):
            result_parts.append(f"**🖥️ eVisa Available:** {', '.join(info['e_visa'][:5])}\n")
        if info.get("visa_on_arrival"):
            result_parts.append(f"**🛬 Visa on Arrival:** {', '.join(info['visa_on_arrival'])}\n")
        result_parts.append(f"\n**Cost:** {info['cost']}\n")
        result_parts.append(f"**Processing Time:** {info['processing']}\n")
        result_parts.append(f"**Requirements:** {info['requirements']}\n")
        result_parts.append(f"**Official Website:** {info['official']}\n")

        # Special notes
        if info.get("etias_2025"):
            result_parts.append(f"\n⚠️ **2025 Update:** {info['etias_2025']}\n")
        if info.get("eta_2024"):
            result_parts.append(f"\n⚠️ **2024 Update:** {info['eta_2024']}\n")
        if info.get("schengen"):
            result_parts.append(f"\n📋 **Schengen Zone:** Single visa covers 27 EU countries for up to 90 days\n")

    else:
        # Fallback: Wikipedia lookup for lesser-known destinations
        try:
            wiki_query = f"{destination} visa requirements travel"
            wiki_result = wikipedia.summary(wiki_query, sentences=3)
            result_parts.append(f"**General Info:**\n{wiki_result}\n\n")
        except Exception:
            result_parts.append(f"Specific visa data not in database for {destination}.\n\n")

        result_parts.append("**📋 General Visa Checklist:**\n")
        result_parts.append("- Passport valid for 6+ months beyond travel dates\n")
        result_parts.append("- Sufficient blank passport pages (2+ pages)\n")
        result_parts.append("- Return/onward ticket\n")
        result_parts.append("- Proof of accommodation\n")
        result_parts.append("- Proof of sufficient funds\n")
        result_parts.append("- Travel insurance (required in some countries)\n")

    result_parts.append("\n> ⚠️ **ALWAYS verify** current requirements on the official embassy website\n")
    result_parts.append("> or at [iata.org/timatic](https://www.iata.org/en/services/travel-documents/) before booking.\n")
    result_parts.append("> Requirements change frequently and vary by passport.\n")
    result_parts.append("\n> **Production:** [Sherpa API](https://developers.sherpa.io) provides real-time visa data by passport nationality.\n")

    result = "".join(result_parts)
    _visa_cache[cache_key] = (time.time(), result)
    return result
