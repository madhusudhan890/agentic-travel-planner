"""
app/graph/builder.py
─────────────────────
StateGraph construction — wires all nodes and edges into the complete graph.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LANGGRAPH STATEGRAPH CONCEPTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

StateGraph is the core primitive. It's a directed graph where:
  - Nodes: Python functions (async or sync)
  - Edges: Connections between nodes (static or conditional)
  - State: Shared dict that flows through all nodes

vs. AgentExecutor (Phase 2):
  AgentExecutor is a LOOP: reason → act → observe → repeat
  StateGraph is a GRAPH: multiple paths, forks, merges, loops

vs. Chains (Phase 1):
  Chains are LINEAR: step1 → step2 → step3
  StateGraph can BRANCH: step1 → {step2a AND step2b} → step3

GRAPH TOPOLOGY (what we're building):
  START
    │
  planner ──────────────────────────────────── Sets agents_to_run
    │
  retriever ────────────────────────────────── Parallel RAG lookup
    │
  run_agents ───────────────────────────────── All specialists parallel
    │
  writer ───────────────────────────────────── Synthesizes everything
    │
  reviewer ─┬── approved ────────────────────── human_approval
            └── needs_revision (count < max) ── writer (LOOP)
                 exceeded max? ───────────────── human_approval
    │
  human_approval ─┬── approved ──────────────── pdf_generator
                  └── rejected ─────────────── writer (LOOP with feedback)
    │
  pdf_generator ──────────────────────────────── Generates PDF
    │
  email_sender ───────────────────────────────── Sends email
    │
  END

GRAPH COMPILATION:
  graph.compile() returns a CompiledGraph (Runnable).
  The compiled graph is cached and reused for all requests.
  Compilation is expensive (~500ms), invocation is fast.
  WHY: compilation validates the graph structure (detects cycles,
  missing nodes, invalid edge targets) before any user request.

INTERRUPT POINTS:
  human_approval_node uses interrupt() which requires:
  1. A checkpointer (to save state before interrupting)
  2. interrupt_before or interrupt_after specification in compile()

INTERVIEW QUESTIONS:
  Q: How does LangGraph differ from LangChain?
  A: LangChain: prompt templates, chains, retrievers — building blocks
     LangGraph: orchestration layer — defines HOW those blocks flow

  Q: When would you use StateGraph over AgentExecutor?
  A: AgentExecutor: single agent, simple loop, fixed set of tools
     StateGraph: multi-agent, complex routing, human-in-the-loop,
     parallel execution, conditional logic, loops
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.graph.checkpointer import get_checkpointer
from app.graph.edges import after_reviewer_edge, after_human_approval_edge
from app.graph.nodes import (
    planner_node,
    retriever_node,
    run_agents_node,
    writer_node,
    reviewer_node,
    human_approval_node,
    pdf_generator_node,
    email_sender_node,
)

logger = get_logger(__name__)

# Module-level compiled graph singleton
_compiled_graph = None


def build_travel_graph():
    """
    Construct and compile the full LangGraph travel planning workflow.

    Returns:
        CompiledGraph — a Runnable that accepts state dicts and
        returns the final state after all nodes have executed.

    COMPILATION STEPS:
    1. Create StateGraph
    2. Add all nodes (register Python functions as graph nodes)
    3. Set entry point (first node after START)
    4. Add static edges (unconditional A → B connections)
    5. Add conditional edges (routing based on state)
    6. Compile with checkpointer (enables human-in-the-loop + persistence)
    """
    from langgraph.graph import StateGraph, START, END

    # StateGraph accepts a state schema.
    # Using dict as schema (flexible, no type enforcement at runtime).
    # Production: use a TypedDict class for IDE support + validation.
    graph = StateGraph(dict)

    # ── Step 1: Add all nodes ─────────────────────────────────────────────────
    # WHY these names: they match node_statuses keys in state.py
    # so the UI can look up status by node name.
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("run_agents", run_agents_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("pdf_generator", pdf_generator_node)
    graph.add_node("email_sender", email_sender_node)

    # ── Step 2: Set entry point ───────────────────────────────────────────────
    # START is a special constant — the graph begins here
    graph.add_edge(START, "planner")

    # ── Step 3: Static edges (unconditional) ──────────────────────────────────
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "run_agents")
    graph.add_edge("run_agents", "writer")
    # NOTE: writer → reviewer is handled, but reviewer can loop back to writer

    # ── Step 4: Writer → Reviewer (static — reviewer always follows writer) ───
    graph.add_edge("writer", "reviewer")

    # ── Step 5: Conditional edges (routing based on state) ────────────────────

    # After reviewer: approve → human_approval, reject → writer (loop)
    graph.add_conditional_edges(
        "reviewer",              # Source node
        after_reviewer_edge,     # Edge function (reads state, returns string)
        {
            "writer": "writer",                    # Retry loop
            "human_approval": "human_approval",    # Proceed
        },
    )

    # After human approval: approved → pdf_generator, rejected → writer (loop)
    graph.add_conditional_edges(
        "human_approval",
        after_human_approval_edge,
        {
            "pdf_generator": "pdf_generator",  # Approved
            "writer": "writer",                # Human rejected — loop with feedback
        },
    )

    # Final static edges: output pipeline
    graph.add_edge("pdf_generator", "email_sender")
    graph.add_edge("email_sender", END)

    # ── Step 6: Compile with checkpointer ─────────────────────────────────────
    # interrupt_before=["human_approval"]: pause graph BEFORE this node executes
    # This lets us send the draft to the UI before asking for human input.
    checkpointer = get_checkpointer()

    compiled = graph.compile(
        checkpointer=checkpointer,
        # Interrupt before human_approval so UI can show the draft
        # before asking the human to approve
        interrupt_before=["human_approval"],
    )

    logger.info(
        "Travel graph compiled | nodes=%d | checkpointer=%s",
        len(graph.nodes),
        type(checkpointer).__name__,
    )
    return compiled


def get_travel_graph():
    """Return the compiled travel planning graph (singleton)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_travel_graph()
    return _compiled_graph
