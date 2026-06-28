"""app/rag/__init__.py"""
from app.rag.embedder import get_embedding_model, get_query_embedding_model
from app.rag.vectorstore import get_vectorstore, similarity_search, similarity_search_with_score
from app.rag.ingestor import ingest_knowledge_base

__all__ = [
    "get_embedding_model",
    "get_query_embedding_model",
    "get_vectorstore",
    "similarity_search",
    "similarity_search_with_score",
    "ingest_knowledge_base",
]
