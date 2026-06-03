"""Token-aware chunking for the Medium RAG pipeline.

`chunk_text` splits an article body into overlapping pieces sized in real
cl100k_base tokens (not characters), so `chunk_size` parameters from the
config map 1:1 to embedder tokens. Pure CPU, no I/O, no config dependency.
"""

from __future__ import annotations

from functools import lru_cache

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter


@lru_cache(maxsize=1)
def _encoder() -> tiktoken.Encoding:
    """Cached cl100k_base encoder. text-embedding-3-small and gpt-5-mini both use
    cl100k-compatible tokenization, so one encoder serves length-counting for
    both. The vocab load is ~20-50 ms, so we pay it once per process."""
    return tiktoken.get_encoding("cl100k_base")


def token_length(text: str) -> int:
    """Return the number of cl100k_base tokens in `text`."""
    if not text:
        return 0
    return len(_encoder().encode(text))


def chunk_text(text: str, chunk_size: int, overlap_ratio: float) -> list[str]:
    """Split `text` into overlapping chunks sized in real tokens.

    Args:
        text: the article body. Empty or whitespace-only input returns [].
        chunk_size: target max tokens per chunk. Must be >= 1.
        overlap_ratio: fraction of chunk_size to overlap between consecutive
            chunks. Must be in [0.0, 1.0). The actual overlap (in tokens) is
            int(chunk_size * overlap_ratio).

    Returns:
        Chunks in document order. Each chunk's token_length is <= chunk_size.

    Raises:
        ValueError: if chunk_size <= 0 or overlap_ratio is not in [0, 1).
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if overlap_ratio < 0.0:
        raise ValueError(f"overlap_ratio must be >= 0.0, got {overlap_ratio}")
    if overlap_ratio >= 1.0:
        raise ValueError(
            f"overlap_ratio must be < 1.0, got {overlap_ratio} "
            f"(>=1.0 means every chunk is fully overlapped)"
        )

    if text is None or not text.strip():
        return []

    chunk_overlap = int(chunk_size * overlap_ratio)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=token_length,
    )
    return splitter.split_text(text)
