"""Embedding utilities using sentence-transformers."""

from functools import lru_cache
from typing import List, Optional
from src import config

@lru_cache(maxsize=1)
def _get_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

def embed_texts(texts: List[str], model_name: Optional[str] = None) -> List[List[float]]:
    """Compute embeddings for a list of strings."""
    if not texts:
        return []
    name = model_name or config.EMBEDDING_MODEL
    model = _get_model(name)
    embeddings = model.encode(texts, convert_to_numpy=True)
    return [emb.tolist() for emb in embeddings]

def embed_query(text: str, model_name: Optional[str] = None) -> List[float]:
    """Compute embedding for a single query string."""
    return embed_texts([text], model_name)[0]
