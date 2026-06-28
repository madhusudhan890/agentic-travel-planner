"""
app/graph/state.py
──────────────────
The TravelState TypedDict — the heart of the LangGraph system.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY STATE EXISTS (LangGraph vs AgentExecutor)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AgentExecutor (Phase 2): The agent's "state" is a Python dict passed
through a ReAct loop. It only has input, scratchpad, and output.
You can't pause it, fork it, or inspect intermediate results.

LangGraph StateGraph: Every node reads the full state and returns
a PARTIAL state update (only the fields it changed). LangGraph
MERGES these updates using "reducers":
  - Default reducer: replace the field entirely (last write wins)
  - add_messages reducer: append new messages to the list
  - operator.add reducer: accumulate list values across parallel nodes

This design enables:
  ✓ Human-in-the-loop (pause/resume with full context)
  ✓ Parallel agents (each writes different state fields)
  ✓ Loops (writer → reviewer → writer uses revision_count)
  ✓ Checkpointing (serialize state to DB, resume after crash)
  ✓ Time-travel debugging (replay from any checkpoint)
  ✓ Streaming (UI sees state changes in real-time)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOKEN IMPLICATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The messages field uses add_messages — this ACCUMULATES all messages.
After 10 nodes, if each node adds 500 tokens, you have 5,000 tokens
in the messages list. Every subsequent node gets ALL previous messages
in context — this is how conversation memory works but also how token
costs grow. Production strategy: use ConversationSummaryMemory to
compress old messages and keep context within budget.

Token budget per field (estimate for typical trip):
  - rag_context: ~1,000 tokens (5 chunks × 200 tokens each)
  - flight_data: ~500 tokens
  - hotel_data: ~500 tokens
  - weather_data: ~300 tokens
  - restaurant_data: ~400 tokens
  - visa_info: ~300 tokens
  - draft_itinerary: ~2,000 tokens
  Total state passed to writer: ~5,000 tokens
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# ── Agent Status (for UI real-time visualization) ─────────────────────────────
# WHY: The UI needs to know the exact status of every agent to animate
# the graph visualization. Stored in state so it survives checkpoints.
class AgentStatusInfo:
    """Mutable dataclass tracking per-agent execution status for the UI."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    SKIPPED = "skipped"


# ── ALL GRAPH NODES (for status tracking in state) ───────────────────────────
ALL_NODES = [
    "planner", "router",
    "retriever", "search",
    "flight_agent", "hotel_agent", "weather_agent",
    "budget_agent", "restaurant_agent", "visa_agent", "currency_agent",
    "writer", "reviewer", "human_approval",
    "pdf_generator", "email_sender",
]


class TravelState(dict):
    """
    Production LangGraph state for the Travel Planner multi-agent system.

    WHY TypedDict: Python TypedDicts give you IDE type-checking and
    self-documenting field names without runtime overhead.
    LangGraph reads the type annotations to discover Annotated reducers.

    FIELD CATEGORIES:
      1. Input fields      — set at graph START, never modified
      2. Planner output    — structured trip plan from planner node
      3. Router decisions  — which agents/searches to run
      4. Agent outputs     — one field per specialist agent
      5. RAG/Search        — retrieved context for grounding
      6. Writer/Review     — itinerary drafts and feedback
      7. Human-in-the-loop — approval state
      8. Final output      — PDF path, email status
      9. Observability     — token counts, timing, cost
      10. UI state          — real-time visualization data
    """
    pass


def make_initial_state(
    destination: str,
    days: int,
    month: str,
    budget: str,
    interests: str,
    travel_style: str,
    user_id: str,
    session_id: str,
    num_travelers: int = 1,
    departure_city: str = "",
    user_email: Optional[str] = None,
    require_human_approval: bool = False,
) -> Dict[str, Any]:
    """
    Factory: creates a fully-initialized state dict for a new graph execution.

    WHY a factory function: ensures EVERY field has a defined initial value.
    LangGraph merges partial updates from nodes — if a field is missing
    on first access, you get a KeyError. The factory prevents this.

    PRODUCTION PATTERN: Always use a factory. Never assume state fields exist.
    """
    return {
        # ── 1. Input (immutable after creation) ───────────────────────────────
        "destination": destination,
        "days": days,
        "month": month,
        "budget": budget,
        "interests": interests or "general sightseeing",
        "travel_style": travel_style or "balanced",
        "num_travelers": num_travelers,
        "departure_city": departure_city,
        "user_id": user_id,
        "session_id": session_id,
        "user_email": user_email,
        "require_human_approval": require_human_approval,

        # ── 2. Messages (add_messages reducer: append-only) ────────────────────
        # WHY add_messages: unlike a simple list, add_messages handles
        # message deduplication by ID and supports tool call message merging.
        # The messages list is what the LLM sees as conversation history.
        "messages": [],

        # ── 3. Planner output ──────────────────────────────────────────────────
        # The planner extracts structure from the free-form request
        "trip_plan": None,         # Dict with structured trip details

        # ── 4. Router decisions ────────────────────────────────────────────────
        # WHY router: not every trip needs every agent.
        # A 1-day city trip doesn't need visa/flight research.
        # Router saves tokens + latency by skipping irrelevant agents.
        "agents_to_run": [],       # e.g. ["flight", "hotel", "weather", "restaurant"]
        "search_queries": [],      # Specific search queries for each agent

        # ── 5. RAG + Search context ───────────────────────────────────────────
        "rag_context": None,       # Retrieved from ChromaDB knowledge base
        "web_context": None,       # Retrieved via Wikipedia/web tools

        # ── 6. Specialist agent outputs ────────────────────────────────────────
        # Each agent writes ONLY its own field — they're independent.
        # WHY separate fields: agents run in parallel (asyncio.gather).
        # If they all wrote to the same field, there'd be a write conflict.
        "flight_data": None,
        "hotel_data": None,
        "weather_data": None,
        "budget_analysis": None,
        "restaurant_data": None,
        "visa_info": None,
        "currency_info": None,

        # ── 7. Errors (add reducer: accumulate from all nodes) ─────────────────
        # WHY operator.add: if planner has error AND hotel has error,
        # both errors are preserved. Replace would lose one.
        "errors": [],

        # ── 8. Writer / Reviewer ───────────────────────────────────────────────
        "draft_itinerary": None,
        "review_feedback": None,   # Reviewer's specific feedback for revision
        "review_passed": False,    # True when reviewer approves
        "revision_count": 0,       # Incremented each writer→reviewer loop
        "max_revisions": 2,        # Configurable limit to prevent infinite loops

        # ── 9. Human-in-the-Loop ──────────────────────────────────────────────
        # WHY: In enterprise AI, humans must approve AI-generated content
        # before it triggers real-world actions (email, booking, payment).
        # LangGraph's interrupt() pauses execution here.
        "human_approved": not require_human_approval,  # auto-approve if not required
        "human_feedback": None,

        # ── 10. Final output ───────────────────────────────────────────────────
        "final_itinerary": None,
        "pdf_path": None,
        "email_sent": False,

        # ── 11. Observability ──────────────────────────────────────────────────
        # WHY track tokens: token cost IS the infrastructure cost in AI systems.
        # $0.002/1K input tokens × 10,000 tokens = $0.02 per plan.
        # At 1,000 plans/day = $20/day. At 100K plans/day = $2,000/day.
        # You MUST know your token usage to manage costs.
        "total_tokens_in": 0,
        "total_tokens_out": 0,
        "total_cost_usd": 0.0,
        "node_timings": {},        # {"planner": 1.2, "writer": 3.4} seconds

        # ── 12. UI State (for real-time visualization) ────────────────────────
        # WHY: The SSE stream sends these values to the frontend so it can
        # animate the graph visualization — showing which agent is active.
        "current_node": "start",
        "node_statuses": {node: AgentStatusInfo.PENDING for node in ALL_NODES},
        "node_outputs": {},        # {"planner": "Extracted: 5 days Tokyo...", ...}
        "progress_pct": 0,         # 0–100 for progress bar
    }
