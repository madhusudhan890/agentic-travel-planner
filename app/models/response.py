"""
app/models/response.py
───────────────────────
Pydantic response schemas for the travel planner API.
"""

from __future__ import annotations

from pydantic import BaseModel


class TripResponse(BaseModel):
    """Response schema for POST /chat/plan"""

    destination: str
    days: int
    month: str
    itinerary: str
    llm_provider: str

    model_config = {"json_schema_extra": {
        "example": {
            "destination": "Tokyo, Japan",
            "days": 5,
            "month": "October",
            "itinerary": "**Day 1: Arrival & Shinjuku**\n\nMorning: ...",
            "llm_provider": "gemini",
        }
    }}
