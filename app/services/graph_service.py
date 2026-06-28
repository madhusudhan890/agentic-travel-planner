"""
app/services/graph_service.py
──────────────────────────────
Graph execution orchestrator with Server-Sent Events (SSE) streaming.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STREAMING CONCEPT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Without streaming: User waits 20-30 seconds with no feedback.
With SSE streaming: User sees real-time updates as each agent completes.

SSE (Server-Sent Events): HTTP connection stays open, server pushes
text events to the client. Each event: "data: {json}\n\n"

Why SSE over WebSocket?
  SSE: Unidirectional (server → client), HTTP-native, simpler
  WebSocket: Bidirectional, separate protocol, more complex
  For AI output streaming, SSE is the right choice (most AI products use it).

LangGraph + SSE:
  graph.astream_events(input, config) → async generator of events
  We map LangGraph events → our custom SSE event types:
    on_chain_start (node) → node_start
    on_chain_end (node) → node_end
    on_llm_stream → text_chunk
    on_chain_end (final) → complete
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncGenerator, Dict

from app.core.config import get_settings
from app.core.logging import get_logger
from app.graph.builder import get_travel_graph
from app.graph.state import make_initial_state, AgentStatusInfo
from app.models.request import TripRequest

logger = get_logger(__name__)
settings = get_settings()

# Node to progress percentage mapping
NODE_PROGRESS = {
    "planner": 10,
    "retriever": 20,
    "run_agents": 60,
    "writer": 72,
    "reviewer": 78,
    "human_approval": 85,
    "pdf_generator": 93,
    "email_sender": 100,
}

NODE_DISPLAY_NAMES = {
    "planner": "🧠 Trip Planner",
    "retriever": "🔍 Knowledge Retriever",
    "run_agents": "⚡ Specialist Agents",
    "writer": "✍️ Itinerary Writer",
    "reviewer": "👀 Quality Reviewer",
    "human_approval": "👤 Human Approval",
    "pdf_generator": "📄 PDF Generator",
    "email_sender": "📧 Email Sender",
}


def _make_event(event_type: str, data: Dict[str, Any]) -> str:
    """Format a Server-Sent Event string."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"data: {json.dumps({'type': event_type, 'payload': data})}\n\n"


async def stream_travel_plan(request: TripRequest) -> AsyncGenerator[str, None]:
    """
    Execute the LangGraph travel planning workflow and stream SSE events.

    This is the CORE of the Phase 3 system. It:
    1. Initializes graph state from the request
    2. Runs graph.astream_events() to get real-time LangGraph events
    3. Maps LangGraph events → SSE events for the UI
    4. Handles Ctrl+C (CancelledError) properly — never swallowed

    Args:
        request: TripRequest from the API endpoint

    Yields:
        SSE event strings (formatted as "data: {...}\n\n")
    """
    graph = get_travel_graph()
    session_id = request.session_id
    start_time = time.time()

    # Build initial state
    initial_state = make_initial_state(
        destination=request.destination,
        days=request.days,
        month=request.month,
        budget=request.budget,
        interests=request.interests or "general sightseeing",
        travel_style=request.travel_style or "balanced",
        user_id=request.user_id,
        session_id=session_id,
        num_travelers=request.num_travelers,
        departure_city=request.departure_city,
        user_email=request.user_email,
        require_human_approval=request.require_human_approval,
    )

    # LangGraph config — thread_id isolates this user's state
    config = {
        "configurable": {
            "thread_id": session_id,
        },
        "recursion_limit": 20,  # Safety: max graph steps before halting
    }

    logger.info(
        "Graph stream starting | session=%s | destination=%s | agents_to_run=TBD",
        session_id[:8], request.destination,
    )

    # ── Send initial event ────────────────────────────────────────────────────
    yield _make_event("start", {
        "session_id": session_id,
        "destination": request.destination,
        "message": f"Starting AI Travel Planner for {request.destination}...",
    })

    current_node = None
    final_state = None

    try:
        # ── Stream LangGraph events ───────────────────────────────────────────
        # astream_events() is an async generator that yields events as they happen.
        # version="v2" is the newer event format with more detail.
        async for event in graph.astream_events(initial_state, config, version="v2"):
            event_name = event.get("event", "")
            event_tags = event.get("tags", [])
            event_data = event.get("data", {})
            event_metadata = event.get("metadata", {})

            # ── Node START event ──────────────────────────────────────────────
            if event_name == "on_chain_start":
                node_name = event.get("name", "")
                if node_name in NODE_PROGRESS:
                    current_node = node_name
                    display_name = NODE_DISPLAY_NAMES.get(node_name, node_name)
                    progress = NODE_PROGRESS.get(node_name, 0)

                    logger.info("Graph: node_start=%s", node_name)
                    yield _make_event("node_start", {
                        "node": node_name,
                        "display_name": display_name,
                        "progress": progress,
                        "message": f"Starting {display_name}...",
                        "elapsed_sec": round(time.time() - start_time, 1),
                    })

            # ── Node END event ────────────────────────────────────────────────
            elif event_name == "on_chain_end":
                node_name = event.get("name", "")
                if node_name in NODE_PROGRESS:
                    output = event_data.get("output", {})
                    node_output = ""
                    if isinstance(output, dict):
                        node_outputs = output.get("node_outputs", {})
                        node_output = node_outputs.get(node_name, "")

                    display_name = NODE_DISPLAY_NAMES.get(node_name, node_name)
                    progress = NODE_PROGRESS.get(node_name, 0)

                    yield _make_event("node_end", {
                        "node": node_name,
                        "display_name": display_name,
                        "progress": progress,
                        "status": AgentStatusInfo.COMPLETED,
                        "output_preview": (node_output or "")[:200],
                        "elapsed_sec": round(time.time() - start_time, 1),
                    })

                    # After agents complete, send their outputs
                    if node_name == "run_agents" and isinstance(output, dict):
                        agent_data = {
                            "flight": output.get("flight_data", ""),
                            "hotel": output.get("hotel_data", ""),
                            "weather": output.get("weather_data", ""),
                            "budget": output.get("budget_analysis", ""),
                            "restaurant": output.get("restaurant_data", ""),
                            "visa": output.get("visa_info", ""),
                            "currency": output.get("currency_info", ""),
                        }
                        yield _make_event("agent_data", {
                            "agents": {k: v for k, v in agent_data.items() if v},
                        })

                    # After writer, stream the draft
                    if node_name == "writer" and isinstance(output, dict):
                        draft = output.get("draft_itinerary", "")
                        if draft:
                            yield _make_event("draft_ready", {
                                "draft": draft,
                                "revision": output.get("revision_count", 0),
                            })

                    # Save final state
                    final_state = output

            # ── LLM streaming chunks (token-by-token) ─────────────────────────
            elif event_name == "on_chat_model_stream":
                chunk = event_data.get("chunk", {})
                if hasattr(chunk, "content") and chunk.content:
                    yield _make_event("text_chunk", {
                        "text": chunk.content,
                        "node": current_node or "unknown",
                    })

            # ── Tool call events ───────────────────────────────────────────────
            elif event_name == "on_tool_start":
                tool_name = event.get("name", "unknown_tool")
                tool_input = event_data.get("input", {})
                yield _make_event("tool_start", {
                    "tool": tool_name,
                    "input": str(tool_input)[:100],
                })

            elif event_name == "on_tool_end":
                tool_name = event.get("name", "unknown_tool")
                yield _make_event("tool_end", {
                    "tool": tool_name,
                    "status": "completed",
                })

            # ── Human interrupt event ─────────────────────────────────────────
            elif event_name == "on_chain_end" and "interrupt" in str(event_data):
                yield _make_event("human_approval_required", {
                    "session_id": session_id,
                    "message": "Please review and approve the itinerary.",
                    "draft": initial_state.get("draft_itinerary", ""),
                })

    except asyncio.CancelledError:
        # Ctrl+C or connection dropped — clean shutdown
        logger.info("Stream cancelled for session=%s", session_id[:8])
        yield _make_event("cancelled", {"message": "Request cancelled."})
        raise

    except Exception as exc:
        logger.error("Stream error for session=%s: %s", session_id[:8], exc)
        yield _make_event("error", {
            "message": f"Error: {str(exc)[:200]}",
            "session_id": session_id,
        })

    # ── Complete event ────────────────────────────────────────────────────────
    total_elapsed = round(time.time() - start_time, 1)

    # Get final state from checkpoint
    try:
        checkpoint_state = graph.get_state(config)
        final_values = checkpoint_state.values if checkpoint_state else {}
        final_itinerary = final_values.get("final_itinerary") or final_values.get("draft_itinerary", "")
        pdf_path = final_values.get("pdf_path", "")
        email_sent = final_values.get("email_sent", False)
        errors = final_values.get("errors", [])
    except Exception:
        final_itinerary = ""
        pdf_path = ""
        email_sent = False
        errors = []

    yield _make_event("complete", {
        "session_id": session_id,
        "destination": request.destination,
        "final_itinerary": final_itinerary,
        "pdf_path": pdf_path,
        "email_sent": email_sent,
        "total_elapsed_sec": total_elapsed,
        "errors": errors[:5],  # Don't flood with errors
        "message": f"✅ Travel plan complete! Generated in {total_elapsed}s",
    })

    logger.info(
        "Graph stream complete | session=%s | elapsed=%.1fs | errors=%d",
        session_id[:8], total_elapsed, len(errors),
    )


async def resume_after_human_approval(
    session_id: str,
    approved: bool,
    feedback: str = "",
) -> Dict[str, Any]:
    """
    Resume a graph that paused at human_approval checkpoint.

    HUMAN-IN-THE-LOOP FLOW:
    1. stream_travel_plan() pauses when graph.interrupt() is called
    2. User sees the draft itinerary in UI
    3. User clicks Approve or Reject
    4. POST /graph/approve calls this function
    5. graph.invoke(Command(resume={...})) resumes from checkpoint

    Args:
        session_id: The thread_id of the paused graph execution
        approved: Whether the human approved the itinerary
        feedback: Human feedback (used if rejected, sent to writer)

    Returns:
        Final state after graph completes.
    """
    from langgraph.types import Command

    graph = get_travel_graph()
    config = {"configurable": {"thread_id": session_id}}

    # Check that the graph is actually paused at human_approval
    current_state = graph.get_state(config)
    if not current_state:
        raise ValueError(f"No paused graph found for session_id={session_id}")

    logger.info(
        "Resuming graph | session=%s | approved=%s",
        session_id[:8], approved,
    )

    # Resume with human's decision
    # Command(resume=...) passes data to the interrupt() call in human_approval_node
    result = await asyncio.wait_for(
        asyncio.get_event_loop().run_in_executor(
            None,
            lambda: graph.invoke(
                Command(resume={"approved": approved, "feedback": feedback}),
                config,
            ),
        ),
        timeout=settings.agent_timeout_sec * 3,  # Allow extra time for remaining nodes
    )

    return result
