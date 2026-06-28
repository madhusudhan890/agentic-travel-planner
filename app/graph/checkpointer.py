"""
app/graph/checkpointer.py
──────────────────────────
LangGraph checkpointer setup — MemorySaver (dev) with Postgres migration path.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY CHECKPOINTERS EXIST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Without a checkpointer:
  - Graph runs from START to END in one shot
  - If it crashes mid-way, all work is lost
  - Human-in-the-loop is impossible (can't pause execution)
  - No conversation history across requests

With a checkpointer (MemorySaver/Postgres):
  - State is saved after EVERY node execution
  - Graph can be paused and resumed (human approval)
  - Multiple threads (users) have isolated state
  - Time-travel debugging: replay from any checkpoint

CHECKPOINT DATA STRUCTURE:
  Each checkpoint contains:
    - thread_id: unique per conversation (session_id)
    - step: which node just completed
    - state: full state dict at that point
    - metadata: timestamp, tags

THREAD ID = SESSION ID:
  LangGraph uses thread_id to isolate concurrent users.
  Two users with different thread_ids get completely separate state.
  Same user using same thread_id gets conversation continuity.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MIGRATION PATH: MemorySaver → PostgreSQL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Development: MemorySaver (in-memory, process-local)
  Pros: Zero setup, zero dependencies
  Cons: Lost on restart, single-process only

Production: PostgresSaver (persistent, multi-process safe)
  Migration steps:
  1. Install: uv add langgraph-checkpoint-postgres psycopg
  2. Create DB: psql -c "CREATE DATABASE travel_planner_checkpoints;"
  3. Set POSTGRES_URL in .env
  4. Update build_checkpointer() below:

  ```python
  from langgraph.checkpoint.postgres import PostgresSaver
  with PostgresSaver.from_conn_string(POSTGRES_URL) as checkpointer:
      checkpointer.setup()  # Creates tables on first run
      return checkpointer
  ```

  The GRAPH CODE DOESN'T CHANGE — only the checkpointer changes.
  This is the power of dependency injection for infrastructure.

ALTERNATIVE: AsyncPostgresSaver (async-native, better performance)
  from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
  Use this for FastAPI (async) applications.
"""

from __future__ import annotations

from app.core.logging import get_logger

logger = get_logger(__name__)


def build_checkpointer():
    """
    Build and return the LangGraph checkpointer.

    Currently: MemorySaver (in-memory, development)
    Production: See module docstring for PostgresSaver migration steps.

    Returns:
        MemorySaver instance for graph checkpointing.
    """
    from langgraph.checkpoint.memory import MemorySaver

    checkpointer = MemorySaver()
    logger.info(
        "Checkpointer: MemorySaver (in-memory, dev mode). "
        "Migrate to PostgresSaver for production persistence."
    )
    return checkpointer


# ── Singleton checkpointer (shared across all graph instances) ─────────────────
# WHY SINGLETON: MemorySaver stores state in its own memory.
# A new instance = empty memory = lost checkpoints = broken human-in-the-loop.
# One shared instance = all sessions, all checkpoints, across all requests.
_checkpointer_instance = None


def get_checkpointer():
    """Return the singleton checkpointer instance."""
    global _checkpointer_instance
    if _checkpointer_instance is None:
        _checkpointer_instance = build_checkpointer()
    return _checkpointer_instance
