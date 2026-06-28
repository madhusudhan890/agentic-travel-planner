"""
app/models/request.py
──────────────────────
Pydantic request schemas for the travel planner API.

Phase 3 additions:
  - session_id: enables LangGraph checkpointing (conversation resumption)
  - user_id: enables long-term memory (user preferences across sessions)
  - user_email: enables PDF email delivery
  - require_human_approval: enables human-in-the-loop workflow
"""

from __future__ import annotations

import uuid
from typing import Optional

from pydantic import BaseModel, Field


class TripRequest(BaseModel):
    """Input schema for POST /graph/plan"""

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
        default="$2000",
        description="Total budget for the trip (currency + amount).",
        examples=["$1000", "₹80,000", "€2000", "£1500"],
    )
    interests: str | None = Field(
        default=None,
        description="Traveler's interests (e.g. food, history, adventure, beaches).",
        examples=["food and culture", "adventure sports", "history and museums"],
    )
    travel_style: str | None = Field(
        default=None,
        description="Preferred travel style.",
        examples=["budget backpacker", "luxury", "family with kids", "solo adventurer"],
    )
    num_travelers: int = Field(
        default=1,
        ge=1,
        le=20,
        description="Number of travelers — affects hotel/transport pricing.",
    )
    departure_city: str = Field(
        default="",
        description="Departure city for flight search.",
        examples=["New York", "London", "Mumbai"],
    )

    # ── Session / User Identity ───────────────────────────────────────────────
    # WHY session_id: LangGraph checkpoints are keyed by thread_id (session_id).
    # This allows the graph to be paused (human approval) and resumed correctly.
    # Without it, every request starts fresh — no memory, no resumption.
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique session ID — auto-generated if not provided. "
                    "Reuse the same session_id to continue a conversation.",
    )
    user_id: str = Field(
        default="anonymous",
        description="User identifier for long-term memory (preferences across sessions).",
    )
    user_email: Optional[str] = Field(
        default=None,
        description="User email — if provided, the final itinerary is emailed as PDF.",
        examples=["traveler@example.com"],
    )

    # ── Workflow Control ──────────────────────────────────────────────────────
    require_human_approval: bool = Field(
        default=False,
        description="If true, the graph pauses before finalizing for human review. "
                    "Resume via POST /graph/approve with session_id.",
    )


class HumanApprovalRequest(BaseModel):
    """Input schema for POST /graph/approve — resume after human-in-the-loop."""
    session_id: str
    approved: bool
    feedback: Optional[str] = Field(
        default=None,
        description="Feedback to send back to the writer if not approved.",
    )
