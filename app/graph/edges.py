"""
app/graph/edges.py
───────────────────
Conditional edge functions for the LangGraph StateGraph.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT EDGES ARE IN LANGGRAPH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

In a StateGraph, edges define the FLOW between nodes.

STATIC EDGE: Always goes from A → B
  graph.add_edge("planner", "router")
  WHY: Every plan always needs routing.

CONDITIONAL EDGE: Examines state, returns which node to go to next
  graph.add_conditional_edges("reviewer", after_reviewer_edge, {...})
  WHY: Reviewer can approve (→ human_approval) or reject (→ writer).

The edge function receives the STATE and returns a STRING (node name).
The string must be a key in the path_map dict passed to add_conditional_edges.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOOPS IN LANGGRAPH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The writer → reviewer → writer loop is implemented via:
  after_reviewer_edge returns "writer" (not END) when needs revision

Without a loop limit (max_revisions), this could run forever.
The edge function checks revision_count and forces END after limit.

INTERVIEW QUESTIONS:
  Q: How do you prevent infinite loops in LangGraph?
  A: Track iteration count in state. Add a condition in the edge function
     that returns a terminal node (END or next step) when count >= limit.
     This is WHY we have revision_count and max_revisions in state.

  Q: How do conditional edges differ from just using if/else in a node?
  A: Edges separate ROUTING logic from PROCESSING logic.
     Nodes should process data. Edges should decide flow.
     This makes each component independently testable.
     You can test "does after_reviewer_edge return 'writer' when score < 7"
     without running any LLM calls.
"""

from __future__ import annotations

from typing import Any, Dict, Literal

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ── Edge: after reviewer — approve or loop back to writer ─────────────────────
def after_reviewer_edge(
    state: Dict[str, Any],
) -> Literal["writer", "human_approval"]:
    """
    Decide whether to loop back to writer or proceed to human approval.

    LOGIC:
      - review_passed = True → proceed to human_approval
      - review_passed = False AND revision_count < max_revisions → writer (loop)
      - review_passed = False AND revision_count >= max_revisions → human_approval
        (force forward — can't loop forever, human can catch issues)

    WHY FORCE FORWARD at max_revisions:
    A reviewer that never approves could block the entire pipeline.
    After N revisions, let the human decide if it's acceptable.
    This is the same logic used in multi-agent debate systems.
    """
    review_passed = state.get("review_passed", False)
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", settings.max_revisions)

    if review_passed:
        logger.info("Edge: reviewer APPROVED → human_approval")
        return "human_approval"

    if revision_count < max_revisions:
        logger.info(
            "Edge: reviewer REJECTED (revision %d/%d) → writer (loop)",
            revision_count, max_revisions,
        )
        return "writer"

    logger.info(
        "Edge: max revisions reached (%d) → human_approval (forced)",
        max_revisions,
    )
    return "human_approval"


# ── Edge: after human approval — proceed or loop back to writer ───────────────
def after_human_approval_edge(
    state: Dict[str, Any],
) -> Literal["pdf_generator", "writer"]:
    """
    After human review, proceed to PDF or loop back to writer with feedback.

    LOGIC:
      - human_approved = True → pdf_generator
      - human_approved = False → writer (with human_feedback)

    WHY HUMAN CAN LOOP BACK:
    Human-in-the-loop is meaningless if "reject" just ends the flow.
    When rejected, the human's feedback is stored in state.human_feedback
    and the writer reads it in the next iteration via review_feedback.
    This creates a true human-AI collaboration loop.
    """
    human_approved = state.get("human_approved", True)
    require_approval = state.get("require_human_approval", False)

    if not require_approval or human_approved:
        logger.info("Edge: human APPROVED (or not required) → pdf_generator")
        return "pdf_generator"

    logger.info("Edge: human REJECTED → writer (with feedback)")
    return "writer"
