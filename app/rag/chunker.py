"""
app/rag/chunker.py
───────────────────
Text chunking strategies for RAG document preparation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY CHUNKING EXISTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LLMs have a context window limit (e.g., 128K tokens for Gemini 1.5 Pro,
1M tokens for Gemini Ultra). Embedding models have even smaller limits
(usually 512-8192 tokens).

You CANNOT embed an entire Wikipedia article as one vector — the
embedding model truncates anything beyond its limit, losing information.
You CANNOT put entire books in the LLM context — it's too expensive.

SOLUTION: Split documents into smaller "chunks" that:
1. Fit within embedding model limits (384-1024 tokens per chunk)
2. Are semantically coherent (a chunk should make sense on its own)
3. Have overlap to prevent losing context at chunk boundaries

CHUNKING STRATEGIES:
━━━━━━━━━━━━━━━━━━━

1. FIXED SIZE: Split every N characters regardless of content
   ✓ Simple, predictable
   ✗ Can split mid-sentence, losing semantic coherence
   USE WHEN: Uniform documents (logs, tables)

2. RECURSIVE CHARACTER: Split at paragraph → sentence → word → char
   ✓ Tries to preserve semantic boundaries
   ✓ Good default for general text
   USE WHEN: General text documents (Wikipedia, guides)

3. SEMANTIC: Embed sentences, group similar ones into chunks
   ✓ Highest quality, semantically coherent chunks
   ✗ Expensive (requires embedding every sentence first)
   USE WHEN: High-stakes RAG where quality > speed

4. MARKDOWN/HEADER-BASED: Split at markdown headers
   ✓ Perfect for structured documents
   ✓ Each chunk = one logical section
   USE WHEN: Markdown documentation, structured guides

We implement strategies 1, 2, and 4. Strategy 3 is documented but not
implemented (too slow for this use case, tokens cost money).

CHUNK OVERLAP:
  WHY: If answer spans two chunks, overlap ensures neither chunk
  misses the relevant context. Like overlapping puzzle pieces.
  
  TRADE-OFF: More overlap = more redundant data stored = larger index.
  Typical: 10-20% of chunk_size (800 chunk → 80-160 overlap)

INTERVIEW QUESTIONS:
  Q: What chunk size should you use?
  A: Depends on: embedding model max (respect it), query type (single fact
     → small chunks, complex questions → large chunks), document structure.
     Start with 500-1000 chars, evaluate retrieval quality, iterate.

  Q: How does chunk overlap prevent context loss?
  A: If the answer starts at char 790 in a 800-char chunk, without overlap
     the next chunk starts at 801. With 100-char overlap, the next chunk
     starts at 700, capturing the full answer.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
try:
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter,
        MarkdownHeaderTextSplitter,
        CharacterTextSplitter,
    )
except ImportError:
    from langchain.text_splitter import (  # type: ignore[no-redef]
        RecursiveCharacterTextSplitter,
        MarkdownHeaderTextSplitter,
        CharacterTextSplitter,
    )
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def chunk_documents_recursive(
    documents: List[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Document]:
    """
    Chunk documents using RecursiveCharacterTextSplitter.

    This is the DEFAULT strategy for most RAG use cases.
    It tries to split on: paragraph → sentence → word → character
    preserving semantic boundaries where possible.

    Args:
        documents: Source documents to chunk
        chunk_size: Max chars per chunk (default from config)
        chunk_overlap: Overlap chars between chunks (default from config)

    Returns:
        List of smaller Document chunks with inherited metadata.
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.rag_chunk_size
    chunk_overlap = chunk_overlap or settings.rag_chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # WHY these separators:
        # 1. Double newline = paragraph break (most semantic)
        # 2. Single newline = line break
        # 3. Period/space = sentence end
        # 4. Space = word boundary
        # 5. Empty = character (last resort)
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(documents)

    logger.info(
        "Chunked %d docs → %d chunks | size=%d overlap=%d",
        len(documents), len(chunks), chunk_size, chunk_overlap,
    )
    return chunks


def chunk_markdown_by_headers(
    markdown_text: str,
    source: str = "unknown",
    metadata: dict | None = None,
) -> List[Document]:
    """
    Chunk markdown documents by header structure.

    BEST STRATEGY for structured travel guides because:
    - Each section (## Day 1, ## Visa Info) is a logical unit
    - Queries naturally align with section boundaries
    - Preserves the hierarchical context

    Args:
        markdown_text: Raw markdown content
        source: Source document name (for metadata)
        metadata: Additional metadata to attach to chunks

    Returns:
        Document chunks, one per markdown section.
    """
    # Split by headers (## Heading, ### Subheading)
    # WHY level 1+2: H1 is usually document title (too broad),
    # H2 is the right granularity for travel guide sections.
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,  # Keep headers in chunk content (adds context)
    )

    docs = splitter.split_text(markdown_text)

    # Add source metadata to each chunk
    base_meta = {"source": source, "chunking_strategy": "markdown_header"}
    if metadata:
        base_meta.update(metadata)

    for doc in docs:
        doc.metadata.update(base_meta)

    # SECONDARY SPLIT: If any section is too long, recursively split it
    settings = get_settings()
    secondary = RecursiveCharacterTextSplitter(
        chunk_size=settings.rag_chunk_size * 2,  # Allow longer sections from structured docs
        chunk_overlap=settings.rag_chunk_overlap,
    )
    final_chunks = secondary.split_documents(docs)

    logger.info(
        "Markdown chunking: %d sections → %d final chunks | source=%s",
        len(docs), len(final_chunks), source,
    )
    return final_chunks


def chunk_plain_text(
    text: str,
    source: str,
    metadata: dict | None = None,
) -> List[Document]:
    """
    Chunk plain text with fixed-size strategy.
    Simple, predictable — good for FAQ and Q&A documents.
    """
    settings = get_settings()
    splitter = CharacterTextSplitter(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        separator="\n\n",
    )
    base_meta = {"source": source, "chunking_strategy": "fixed_size"}
    if metadata:
        base_meta.update(metadata)

    doc = Document(page_content=text, metadata=base_meta)
    return splitter.split_documents([doc])
