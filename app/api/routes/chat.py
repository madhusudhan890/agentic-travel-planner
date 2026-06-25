"""
app/api/routes/chat.py  —  /chat
──────────────────────────────────
POST /chat/plan
  • Accept a TripRequest (destination, days, month, budget, interests, style)
  • Invoke the active TravelChain (LangChain LCEL pipeline in Phase 1)
  • Return a TripResponse with the full itinerary

The route has ZERO knowledge of which LLM is used — it only talks to
the TravelChain abstract interface via dependency injection.
In Phase 2: chain internally uses tools. Route doesn't change.
In Phase 3: chain internally uses LangGraph agent. Route doesn't change.
"""

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_travel_chain
from app.chain.base import TravelChain
from app.core.logging import get_logger
from app.models.request import TripRequest
from app.models.response import TripResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    "/plan",
    response_model=TripResponse,
    summary="Generate a detailed travel itinerary",
)
async def plan_trip(
    body: TripRequest,
    chain: TravelChain = Depends(get_travel_chain),
) -> TripResponse:
    """
    **Travel Planning Flow (Phase 1 — LangChain Chain):**

    1. Receives a structured trip request.
    2. Passes it to the active TravelChain (Gemini-powered LCEL pipeline).
    3. Chain builds the prompt → calls Gemini → parses the string output.
    4. Returns the itinerary with metadata.

    In future phases, step 2 will transparently upgrade to tool-calling
    and LangGraph agents — this route will not change.
    """
    try:
        itinerary = await chain.plan_trip(body)
    except RuntimeError as exc:
        # Clean user-facing error raised by the chain's retry logic (high demand / exhausted retries)
        err_msg = str(exc)
        logger.warning("TravelChain service unavailable for destination=%s | %s", body.destination, err_msg)
        raise HTTPException(
            status_code=503,
            detail=err_msg,
        )
    except Exception as exc:

        logger.exception(
            "TravelChain '%s' failed for destination=%s | %s",
            chain.name, body.destination, exc,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate itinerary: {exc}",
        )

    logger.info(
        "Itinerary generated | destination=%s | days=%d | provider=%s",
        body.destination, body.days, chain.name,
    )

    return TripResponse(
        destination=body.destination,
        days=body.days,
        month=body.month,
        itinerary=itinerary,
        llm_provider=chain.name,
    )
