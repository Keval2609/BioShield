"""Text cleaning and chunking utilities."""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    """Represents a text chunk with associated metadata."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def clean_text(text: str) -> str:
    """Normalize whitespace and strip extraneous leading/trailing spaces."""
    if not text:
        return ""
    # Collapse any sequence of whitespace characters (spaces, tabs, newlines) into a single space
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Chunk]:
    """Split clean text into overlapping word-based chunks.
    
    Raises:
        ValueError: If overlap is negative or greater than/equal to chunk_size.
    """
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            f"Invalid overlap ({overlap}): overlap must be >= 0 and strictly less than chunk_size ({chunk_size})"
        )

    cleaned = clean_text(text)
    if not cleaned:
        return []

    words = cleaned.split()
    if not words:
        return []

    meta = dict(metadata or {})
    chunks: List[Chunk] = []
    step = chunk_size - overlap

    for start_idx in range(0, len(words), step):
        chunk_words = words[start_idx : start_idx + chunk_size]
        chunk_str = " ".join(chunk_words)
        chunks.append(Chunk(text=chunk_str, metadata=meta.copy()))
        if start_idx + chunk_size >= len(words):
            break

    return chunks
