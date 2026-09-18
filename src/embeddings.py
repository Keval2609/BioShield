"""Embedding utilities using sentence-transformers."""

from typing import List, Optional
from src import config


class EmbeddingModel:
    """Wrapper around SentenceTransformer with lazy model loading."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a list of strings."""
        if not texts:
            return []
        model = self._load()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Compute embedding for a single query string."""
        return self.embed_texts([text])[0]
