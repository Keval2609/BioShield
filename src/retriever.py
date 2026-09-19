"""Retriever module for semantic search in ChromaDB with natural farming prioritization."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings

from src import config
from src.embeddings import embed_query


@dataclass
class Evidence:
    """Represents a single retrieved evidence passage with source metadata and distance."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    distance: float = 0.0

    @property
    def document_id(self) -> str:
        return str(self.metadata.get("document_id", ""))

    @property
    def document_title(self) -> str:
        return str(self.metadata.get("document_title", self.metadata.get("title", "Unknown Source")))

    @property
    def source_file(self) -> str:
        return str(self.metadata.get("source_file", self.metadata.get("filename", "")))

    @property
    def page(self) -> int:
        try:
            return int(self.metadata.get("page", 1))
        except (ValueError, TypeError):
            return 1

    @property
    def section(self) -> str:
        return str(self.metadata.get("section", "General Advisory"))

    @property
    def source_type(self) -> str:
        return str(self.metadata.get("source_type", ""))

    @property
    def farming_approach(self) -> str:
        return str(self.metadata.get("farming_approach", "natural_farming"))

    @property
    def region(self) -> str:
        return str(self.metadata.get("region", "India"))

    @property
    def topic(self) -> str:
        return str(self.metadata.get("topic", ""))

    @property
    def similarity_score(self) -> float:
        """Normalized similarity score derived from cosine distance (1.0 - distance)."""
        return round(max(0.0, 1.0 - self.distance), 4)


@dataclass
class RetrievalResult:
    """Holds a collection of retrieved evidence items with filtering and priority ranking."""
    evidence: List[Evidence] = field(default_factory=list)

    def filtered(self, threshold: float) -> List[Evidence]:
        """Filter evidence items where distance is less than or equal to threshold."""
        return [item for item in self.evidence if item.distance <= threshold]

    def has_sufficient_evidence(self, threshold: float) -> bool:
        """Check if at least one evidence item satisfies the similarity threshold."""
        return len(self.filtered(threshold)) > 0

    def get_status(self, threshold: float) -> str:
        """Return status string: 'SUFFICIENT' or 'INSUFFICIENT_EVIDENCE'."""
        return "SUFFICIENT" if self.has_sufficient_evidence(threshold) else "INSUFFICIENT_EVIDENCE"

    def prioritized(self) -> List[Evidence]:
        """Sort evidence items giving priority to natural farming practices over conventional/IPM."""
        return sorted(
            self.evidence,
            key=lambda e: (
                0 if e.farming_approach == "natural_farming" else 1,
                e.distance,
            ),
        )

    def get_sources(self, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Return unique source citations for the evidence."""
        items = self.filtered(threshold) if threshold is not None else self.evidence
        seen = set()
        sources = []
        for item in items:
            key = (item.document_title, item.page)
            if key not in seen:
                seen.add(key)
                sources.append({
                    "document_title": item.document_title,
                    "source_file": item.source_file,
                    "page": item.page,
                    "section": item.section,
                    "farming_approach": item.farming_approach,
                    "similarity_score": item.similarity_score,
                })
        return sources


def has_sufficient_evidence(result: RetrievalResult, threshold: float) -> bool:
    """Determine whether the retrieval result contains at least one evidence item meeting the threshold."""
    if not result or not result.evidence:
        return False
    return result.has_sufficient_evidence(threshold)


class ChromaRetriever:
    """Retrieves relevant agricultural documents from local ChromaDB."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
    ):
        self.persist_dir = persist_directory or str(config.CHROMA_PERSIST_DIR)
        self.collection_name = collection_name or config.CHROMA_COLLECTION_NAME
        self.embedding_model_name = embedding_model_name
        self._client = None
        self._collection = None

    def _get_client(self):
        if self._client is None:
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

        query_embedding = embed_query(query, self.embedding_model_name)
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

        # Prioritize natural farming evidence
        result = RetrievalResult(evidence=evidence_items)
        result.evidence = result.prioritized()
        return result
