"""
main.py
───────
Application entry point.

── How to start the server ───────────────────────────────────────────────────

  DEVELOPMENT (with hot-reload — Ctrl+C works correctly):
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

  PRODUCTION (no reload, stable signal handling):
    python main.py
      — or —
    gunicorn main:app -k uvicorn.workers.UvicornWorker --workers 4

── Why NOT `python main.py --reload` or `RELOAD=true` in .env ───────────────
  When uvicorn.run(reload=True) is called from Python, it spawns a separate
  reloader process that catches SIGINT (Ctrl+C) and interferes with graceful
  shutdown of async tasks (like asyncio.sleep in retry loops).

  The correct solution (per uvicorn docs) is to run uvicorn's OWN CLI directly:
    `uvicorn main:app --reload`
  In this mode, uvicorn manages its own signal handlers correctly and
  Ctrl+C cleanly cancels all in-flight async tasks before shutting down.

  `reload` is a development-only CLI flag — it does NOT belong in .env.
"""

import uvicorn
from app.server import create_app
from app.core.config import get_settings

app = create_app()

if __name__ == "__main__":
    # Direct `python main.py` — production mode, no reload.
    # For dev with reload: use `uvicorn main:app --reload` in terminal instead.
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=False,          # Reload via CLI only: `uvicorn main:app --reload`
        log_level=settings.log_level,
    )
