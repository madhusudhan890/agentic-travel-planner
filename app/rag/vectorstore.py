"""
app/rag/vectorstore.py
───────────────────────
ChromaDB vector store setup and operations.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY CHROMADB:
  ✓ Embedded in Python process — no server to run
  ✓ Persistent to disk automatically
  ✓ Free, open source
  ✓ Supports metadata filtering
  ✓ Supports cosine, euclidean, and dot product similarity

PRODUCTION MIGRATION PATH:
  ┌──────────────────────────────────────────────────────┐
  │ Development: ChromaDB (embedded)                     │
  │   chroma_persist_dir = "./data/chroma_db"            │
  │                                                      │
  │ Staging: ChromaDB HTTP server                        │
  │   docker run -p 8001:8000 chromadb/chroma            │
  │   Set CHROMA_HOST=localhost, CHROMA_PORT=8001        │
  │                                                      │
  │ Production Option A: Qdrant                          │
  │   docker run -p 6333:6333 qdrant/qdrant              │
  │   pip install qdrant-client langchain-qdrant          │
  │   Qdrant cloud: free 1GB tier at cloud.qdrant.io     │
  │                                                      │
  │ Production Option B: pgvector (if using Postgres)    │
  │   pip install pgvector langchain-postgres             │
  │   No new infra — extends existing Postgres           │
  │                                                      │
  │ Production Option C: Pinecone                        │
  │   Managed, serverless, free tier: 2GB storage        │
  │   Best for: teams that don't want to manage infra    │
  └──────────────────────────────────────────────────────┘

VECTOR SIMILARITY:
  Cosine similarity: measures angle between vectors (direction, not magnitude)
  ✓ Best for text: word count doesn't matter, meaning does
  ✗ Doesn't distinguish between long and short texts well

  Dot product: magnitude × cosine (length matters)
  ✓ Good when document length should affect ranking

  We use cosine (default in ChromaDB) — standard for text RAG.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Module-level singleton — vector store is expensive to initialize
_vectorstore = None


def get_vectorstore():
    """
    Return the ChromaDB vector store singleton.

    WHY SINGLETON: ChromaDB opens a file lock on persist_dir.
    Multiple instances = file lock errors. One instance shared across all requests.

    THREAD SAFETY: ChromaDB is thread-safe for reads. Writes should be
    serialized (use asyncio.Lock for concurrent write protection in production).
    """
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = _initialize_vectorstore()
    return _vectorstore


def _initialize_vectorstore():
    """Initialize ChromaDB with the configured embedding model."""
    from langchain_chroma import Chroma
    from app.rag.embedder import get_embedding_model

    settings = get_settings()
    persist_dir = settings.chroma_persist_dir
    collection_name = settings.chroma_collection_name

    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    embedding_model = get_embedding_model()

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_dir,
        # WHY cosine distance: semantic similarity for text
        collection_metadata={"hnsw:space": "cosine"},
    )

    # Check if already populated
    try:
        count = vectorstore._collection.count()
        logger.info(
            "ChromaDB initialized | collection=%s | documents=%d | persist_dir=%s",
            collection_name, count, persist_dir,
        )
    except Exception:
        logger.info("ChromaDB initialized (empty) | collection=%s", collection_name)

    return vectorstore


def add_documents(documents: List[Document], batch_size: int = 50) -> int:
    """
    Add documents to the vector store in batches.

    WHY BATCHING:
    1. Embedding API has request size limits
    2. Network failures are easier to retry for small batches
    3. Memory: embedding 1000 docs at once = large memory spike

    Args:
        documents: Pre-chunked documents to embed and store
        batch_size: Documents per embedding API call (default 50)

    Returns:
        Number of documents added.
    """
    vectorstore = get_vectorstore()
    total_added = 0

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        try:
            vectorstore.add_documents(batch)
            total_added += len(batch)
            logger.info(
                "Embedded batch %d/%d (%d docs)",
                i // batch_size + 1,
                (len(documents) - 1) // batch_size + 1,
                len(batch),
            )
        except Exception as e:
            logger.error("Failed to embed batch %d: %s", i // batch_size, e)
            # Continue with next batch — partial failures are acceptable

    return total_added


def similarity_search(
    query: str,
    k: int | None = None,
    filter_metadata: Optional[dict] = None,
) -> List[Document]:
    """
    Search for the most semantically similar documents.

    METADATA FILTERING:
    WHY: Without filtering, a query about "Japan visa" might return
    chunks about "France budget" (both scored as "travel content").
    Filtering by source_type="visa_guide" gives precise results.

    HYBRID SEARCH (not implemented here):
    Combines semantic search with BM25 keyword search.
    Production: Use Qdrant or Elasticsearch for hybrid search.
    Keyword search catches exact matches that semantic search misses.

    Args:
        query: Natural language search query
        k: Number of results to return
        filter_metadata: ChromaDB metadata filter (e.g., {"source_type": "visa"})

    Returns:
        List of matching document chunks, ordered by similarity.
    """
    settings = get_settings()
    k = k or settings.rag_top_k
    vectorstore = get_vectorstore()

    try:
        if filter_metadata:
            results = vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter_metadata,
            )
        else:
            results = vectorstore.similarity_search(query=query, k=k)

        logger.debug(
            "Vector search: query=%s | k=%d | results=%d",
            query[:50], k, len(results),
        )
        return results

    except Exception as e:
        logger.error("Vector search failed: %s", e)
        return []


def similarity_search_with_score(
    query: str,
    k: int | None = None,
    score_threshold: float = 0.3,
) -> List[tuple[Document, float]]:
    """
    Search with relevance scores for reranking and threshold filtering.

    WHY SCORES:
    Without scores, you always return exactly k results even if they're
    irrelevant. With scores, you can filter out low-quality matches
    (score < threshold) to prevent hallucination from bad context.

    Score interpretation (cosine similarity):
      1.0 = identical
      0.8+ = very relevant
      0.5-0.8 = somewhat relevant
      <0.5 = probably not relevant

    Returns:
        List of (document, similarity_score) tuples, filtered by threshold.
    """
    settings = get_settings()
    k = k or settings.rag_top_k
    vectorstore = get_vectorstore()

    try:
        results = vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=k,
        )
        # Filter by threshold
        filtered = [(doc, score) for doc, score in results if score >= score_threshold]
        logger.debug(
            "Scored search: %d results before filter, %d after (threshold=%.2f)",
            len(results), len(filtered), score_threshold,
        )
        return filtered

    except Exception as e:
        logger.error("Scored search failed: %s", e)
        return []


def get_collection_stats() -> dict:
    """Return stats about the vector store collection."""
    try:
        vectorstore = get_vectorstore()
        count = vectorstore._collection.count()
        return {"document_count": count, "status": "ok"}
    except Exception as e:
        return {"document_count": 0, "status": f"error: {e}"}
