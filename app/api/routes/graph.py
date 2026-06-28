"""
app/api/routes/graph.py
────────────────────────
LangGraph multi-agent endpoints — the Phase 3 API.

ENDPOINTS:
  POST /graph/plan/stream  — SSE streaming execution
  POST /graph/approve      — Resume after human-in-the-loop
  GET  /graph/status/{id}  — Check graph execution status
  GET  /graph/stats        — Knowledge base and vector store stats
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from app.api.dependencies import verify_api_key
from app.core.logging import get_logger
from app.models.request import TripRequest, HumanApprovalRequest
from app.services.graph_service import stream_travel_plan, resume_after_human_approval

logger = get_logger(__name__)
router = APIRouter(prefix="/graph", tags=["LangGraph Multi-Agent"])


@router.post(
    "/plan/stream",
    summary="Stream a multi-agent travel plan via SSE",
    description="""
Execute the full LangGraph travel planning workflow with real-time streaming.

**Events emitted:**
- `start` — Graph execution begins
- `node_start` — A node is starting (planner, agents, writer, etc.)
- `node_end` — A node completed with output
- `agent_data` — Specialist agent research results
- `draft_ready` — Writer produced a draft itinerary
- `text_chunk` — Token-by-token text streaming
- `tool_start/end` — Tool calls in progress
- `human_approval_required` — Graph paused for human review
- `complete` — All done, final itinerary available
- `error` — An error occurred

**Client usage (JavaScript):**
```javascript
const response = await fetch('/graph/plan/stream', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(request),
});
const reader = response.body.getReader();
// Read events as they stream...
```
    """,
)
async def stream_plan(request: Request, body: TripRequest):
    """
    SSE streaming endpoint for the LangGraph workflow.

    WHY StreamingResponse instead of EventSourceResponse:
    EventSourceResponse (sse-starlette) is for GET requests.
    StreamingResponse works for POST requests with streaming bodies.

    CANCELLATION:
    When client disconnects (browser closed, Ctrl+C), FastAPI detects
    request.is_disconnected() and stops the generator. The CancelledError
    propagates through the entire graph execution.
    """
    async def generate():
        try:
            async for event in stream_travel_plan(body):
                # Check if client disconnected
                if await request.is_disconnected():
                    logger.info("Client disconnected, stopping stream")
                    break
                yield event
        except asyncio.CancelledError:
            logger.info("Stream cancelled (Ctrl+C or client disconnect)")
            raise
        except Exception as exc:
            logger.error("Stream error: %s", exc)
            yield f'data: {{"type": "error", "payload": {{"message": "{str(exc)[:200]}"}}}}\n\n'

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering for real-time SSE
        },
    )


@router.post(
    "/approve",
    summary="Resume graph after human review",
    description="""
Resume a paused graph execution after human-in-the-loop review.

Call this after receiving a `human_approval_required` SSE event.
The graph will resume from the checkpoint and either:
- **Approved**: Continue to PDF generation and email
- **Rejected**: Loop back to writer with your feedback
    """,
)
async def approve_plan(body: HumanApprovalRequest):
    """Resume the graph with human's approval decision."""
    try:
        result = await resume_after_human_approval(
            session_id=body.session_id,
            approved=body.approved,
            feedback=body.feedback or "",
        )
        return {
            "status": "resumed",
            "approved": body.approved,
            "session_id": body.session_id,
            "final_itinerary": result.get("final_itinerary", ""),
            "pdf_path": result.get("pdf_path", ""),
            "email_sent": result.get("email_sent", False),
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("Resume failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Resume failed: {exc}")


@router.get(
    "/status/{session_id}",
    summary="Check graph execution status",
)
async def get_graph_status(session_id: str):
    """Check the current state of a graph execution by session ID."""
    try:
        from app.graph.builder import get_travel_graph

        graph = get_travel_graph()
        config = {"configurable": {"thread_id": session_id}}
        state = graph.get_state(config)

        if not state:
            raise HTTPException(status_code=404, detail="Session not found")

        values = state.values
        return {
            "session_id": session_id,
            "current_node": values.get("current_node", "unknown"),
            "progress_pct": values.get("progress_pct", 0),
            "review_passed": values.get("review_passed", False),
            "human_approved": values.get("human_approved", False),
            "has_final_itinerary": bool(values.get("final_itinerary")),
            "pdf_path": values.get("pdf_path", ""),
            "email_sent": values.get("email_sent", False),
            "errors": values.get("errors", []),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/stats", summary="Knowledge base and system stats")
async def get_system_stats():
    """Return stats about the RAG knowledge base and system configuration."""
    try:
        from app.rag.vectorstore import get_collection_stats
        from app.core.config import get_settings

        s = get_settings()
        rag_stats = get_collection_stats()

        return {
            "phase": "3 — LangGraph Multi-Agent",
            "llm_model": s.gemini_model,
            "vector_db": "ChromaDB",
            "rag_documents": rag_stats.get("document_count", 0),
            "rag_collection": s.chroma_collection_name,
            "max_revisions": s.max_revisions,
            "langsmith_enabled": s.langsmith_enabled,
            "email_configured": s.email_configured,
            "agents": [
                "planner", "flight", "hotel", "weather",
                "budget", "restaurant", "visa", "currency",
                "writer", "reviewer",
            ],
        }
    except Exception as exc:
        return {"error": str(exc)}
