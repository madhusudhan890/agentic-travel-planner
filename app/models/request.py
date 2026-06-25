"""
app/models/request.py
──────────────────────
Pydantic request schemas for the travel planner API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TripRequest(BaseModel):
    """Input schema for POST /chat/plan"""

    destination: str = Field(
        ...,
        description="The travel destination (city, country, or region).",
        examples=["Tokyo, Japan", "Goa, India", "Paris, France"],
    )
    days: int = Field(
        default=5,
        ge=1,
        le=30,
        description="Number of days for the trip (1–30).",
    )
    month: str = Field(
        default="October",
        description="Month of travel — affects weather and activity recommendations.",
        examples=["January", "July", "December"],
    )
    budget: str = Field(
        default="$1000",
        description="Total budget for the trip (currency + amount).",
        examples=["$500", "₹50,000", "€1500"],
    )
    interests: str | None = Field(
        default=None,
        description="Traveler's interests (e.g. food, history, adventure, beaches).",
        examples=["food and culture", "adventure sports", "history and museums"],
    )
    travel_style: str | None = Field(
        default=None,
        description="Preferred travel style (e.g. backpacker, luxury, family, solo).",
        examples=["budget backpacker", "luxury", "family with kids", "solo adventurer"],
    )
