"""
app/rag/ingestor.py
────────────────────
Document ingestion pipeline — loads and indexes travel knowledge.

This runs ONCE at startup (or manually) to populate ChromaDB.
After ingestion, every query can use semantic search over this knowledge.

WHY A SEPARATE INGESTOR:
  Data pipeline is separated from retrieval pipeline.
  You can re-ingest when documents are updated without changing the app.
  Production: this becomes a scheduled job (Celery task, cron, Airflow DAG).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.chunker import chunk_markdown_by_headers, chunk_plain_text
from app.rag.vectorstore import add_documents, get_collection_stats

logger = get_logger(__name__)


def ingest_knowledge_base(force_reingest: bool = False) -> dict:
    """
    Load travel knowledge documents and index them in ChromaDB.

    Called once at application startup if the vector store is empty.
    Call with force_reingest=True to rebuild the index.

    Args:
        force_reingest: If True, re-index even if documents already exist

    Returns:
        Ingestion stats: {"documents_added": N, "status": "ok"}
    """
    settings = get_settings()

    # Check if already ingested
    stats = get_collection_stats()
    if not force_reingest and stats.get("document_count", 0) > 0:
        logger.info(
            "Knowledge base already indexed (%d chunks). Skipping ingestion.",
            stats["document_count"],
        )
        return {"documents_added": 0, "status": "already_indexed", **stats}

    knowledge_dir = settings.knowledge_dir
    if not knowledge_dir.exists():
        logger.warning("Knowledge dir not found: %s", knowledge_dir)
        return {"documents_added": 0, "status": "no_knowledge_dir"}

    all_chunks: List[Document] = []

    # Load all .md and .txt files from the knowledge directory
    for filepath in sorted(knowledge_dir.glob("**/*.md")):
        try:
            text = filepath.read_text(encoding="utf-8")
            doc_name = filepath.stem
            doc_type = _infer_document_type(filepath.name)

            chunks = chunk_markdown_by_headers(
                markdown_text=text,
                source=filepath.name,
                metadata={
                    "source_type": doc_type,
                    "doc_name": doc_name,
                    "file_path": str(filepath),
                },
            )
            all_chunks.extend(chunks)
            logger.info("Loaded %s: %d chunks (type=%s)", filepath.name, len(chunks), doc_type)

        except Exception as e:
            logger.error("Failed to load %s: %s", filepath, e)

    for filepath in sorted(knowledge_dir.glob("**/*.txt")):
        try:
            text = filepath.read_text(encoding="utf-8")
            chunks = chunk_plain_text(
                text=text,
                source=filepath.name,
                metadata={"source_type": "general", "doc_name": filepath.stem},
            )
            all_chunks.extend(chunks)

        except Exception as e:
            logger.error("Failed to load %s: %s", filepath, e)

    if not all_chunks:
        logger.warning("No knowledge documents found in %s", knowledge_dir)
        return {"documents_added": 0, "status": "no_documents"}

    # Index in ChromaDB
    added = add_documents(all_chunks)
    logger.info(
        "Knowledge base ingestion complete: %d chunks indexed from %d files",
        added,
        len(list(knowledge_dir.glob("**/*.md"))) + len(list(knowledge_dir.glob("**/*.txt"))),
    )
    return {"documents_added": added, "status": "ok"}


def _infer_document_type(filename: str) -> str:
    """Infer document type from filename for metadata filtering."""
    name_lower = filename.lower()
    if "visa" in name_lower:
        return "visa"
    if "budget" in name_lower:
        return "budget"
    if "tip" in name_lower or "guide" in name_lower:
        return "tips"
    if "destination" in name_lower:
        return "destination"
    if "weather" in name_lower or "climate" in name_lower:
        return "weather"
    return "general"
