"""Retriever module for semantic search in ChromaDB."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src import config
from src.embeddings import EmbeddingModel


@dataclass
class Evidence:
    """Represents a single retrieved evidence passage with source metadata and distance."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    distance: float = 0.0


@dataclass
class RetrievalResult:
    """Holds a collection of retrieved evidence items."""
    evidence: List[Evidence] = field(default_factory=list)

    def filtered(self, threshold: float) -> List[Evidence]:
        """Filter evidence items where distance is less than or equal to threshold."""
        return [item for item in self.evidence if item.distance <= threshold]


def has_sufficient_evidence(result: RetrievalResult, threshold: float) -> bool:
    """Determine whether the retrieval result contains at least one evidence item meeting the threshold."""
    if not result or not result.evidence:
        return False
    return len(result.filtered(threshold)) > 0


class ChromaRetriever:
    """Retrieves relevant agricultural documents from local ChromaDB."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[EmbeddingModel] = None,
    ):
        self.persist_dir = persist_directory or str(config.CHROMA_PERSIST_DIR)
        self.collection_name = collection_name or config.CHROMA_COLLECTION_NAME
        self.embedding_model = embedding_model or EmbeddingModel()
        self._client = None
        self._collection = None

    def _get_client(self):
        if self._client is None:
            import chromadb
            from chromadb.config import Settings
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    def _get_collection(self):
        if self._collection is None:
            try:
                client = self._get_client()
                self._collection = client.get_collection(name=self.collection_name)
            except Exception:
                # Collection may not be initialized yet if ingest has not run
                return None
        return self._collection

    def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        """Embed the query and retrieve top_k evidence items from ChromaDB."""
        k = top_k if top_k is not None else config.TOP_K
        collection = self._get_collection()

        if collection is None or collection.count() == 0:
            return RetrievalResult(evidence=[])

        query_embedding = self.embedding_model.embed_query(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        evidence_items: List[Evidence] = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
            dists = results.get("distances", [[]])[0] if results.get("distances") else []

            for i, doc in enumerate(docs):
                meta = metas[i] if i < len(metas) and metas[i] is not None else {}
                dist = dists[i] if i < len(dists) and dists[i] is not None else 0.0
                evidence_items.append(Evidence(text=doc, metadata=meta, distance=dist))

        return RetrievalResult(evidence=evidence_items)
