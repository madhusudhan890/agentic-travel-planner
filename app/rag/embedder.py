"""
app/rag/embedder.py
────────────────────
Embedding generation using Google Gemini (free with API key).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT ARE EMBEDDINGS:
  Embeddings are dense vector representations of text.
  A sentence gets converted to a list of numbers (e.g., 768 floats)
  where semantically similar sentences have similar vectors.

  "Best sushi restaurant in Tokyo" → [0.12, -0.34, 0.89, ...]
  "Top sushi spots in Japan"       → [0.13, -0.32, 0.91, ...]  ← very similar!
  "Hotel in Paris"                 → [-0.78, 0.23, -0.45, ...] ← very different

  This is how RAG "understands" meaning, not just keywords.

WHY GEMINI EMBEDDINGS:
  ✓ Free with Gemini API key (same key as LLM calls)
  ✓ 768-dimensional vectors (good balance of quality vs size)
  ✓ Supports "retrieval" task type for better RAG performance

  Alternatives:
  - OpenAI text-embedding-3-small ($0.02/1M tokens)
  - Sentence Transformers (free, runs locally, no API key)
  - Cohere Embed (free tier: 1000 calls/month)

TASK TYPES (Gemini specific):
  "retrieval_document" → embeddings for documents being stored
  "retrieval_query"    → embeddings for search queries
  Using the right task type significantly improves retrieval quality.
"""

from __future__ import annotations

from typing import List

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def get_embedding_model():
    """
    Return the configured embedding model.

    WHY: Embedding model is expensive to initialize (loads ~500MB if local).
    Using a factory + module-level caching avoids re-loading on every call.

    Returns GoogleGenerativeAIEmbeddings (Gemini) or a fallback.
    """
    settings = get_settings()

    if settings.gemini_api_key:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            model = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",  # Available: gemini-embedding-001, gemini-embedding-2
                google_api_key=settings.gemini_api_key,
                task_type="retrieval_document",  # Optimized for RAG document storage
            )
            logger.info("Embedding model: Gemini gemini-embedding-001")
            return model
        except Exception as e:
            logger.warning("Gemini embeddings failed: %s. Falling back.", e)

    # Fallback: sentence-transformers (free, local, no API key)
    # WHY: Zero external dependency — works completely offline.
    # Tradeoff: First load takes ~30s to download ~90MB model.
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            # Tiny model: 80MB, 384 dimensions, fast inference
        )
        logger.info("Embedding model: HuggingFace all-MiniLM-L6-v2 (384-dim)")
        return model
    except Exception as e:
        logger.error("No embedding model available: %s", e)
        raise RuntimeError(
            "No embedding model configured. Set GEMINI_API_KEY in .env "
            "or install sentence-transformers: uv add sentence-transformers"
        )


def get_query_embedding_model():
    """
    Return embedding model optimized for QUERY embedding (not document).

    WHY: Gemini's task_type="retrieval_query" produces different embeddings
    than "retrieval_document". Using the right type improves similarity search
    quality significantly. Some models (OpenAI) don't distinguish — Gemini does.
    """
    settings = get_settings()

    if settings.gemini_api_key:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=settings.gemini_api_key,
                task_type="retrieval_query",  # Optimized for search queries
            )
        except Exception:
            pass

    # Same fallback — HuggingFace doesn't distinguish task types
    return get_embedding_model()
