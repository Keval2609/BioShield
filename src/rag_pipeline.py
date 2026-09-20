"""RAG pipeline connecting retrieval to IBM Granite grounded generation."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src import config
from src.advisory_schema import (
    PROTOTYPE_LIMITATION,
    AdvisoryOutput,
    InsufficientEvidenceOutput,
    parse_advisory_response,
)
from src.llm import OllamaProvider, GroqProvider
from src.prompts import SYSTEM_PROMPT, build_user_prompt, format_evidence_context
from src.retriever import ChromaRetriever, Evidence, has_sufficient_evidence

logger = logging.getLogger("bioshield.rag_pipeline")

def get_llm_provider() -> Any:
    """Factory to instantiate the configured LLM provider."""
    if config.LLM_PROVIDER == "groq":
        return GroqProvider()
    return OllamaProvider()

SAFE_FALLBACK = (
    "I could not find sufficient verified information in the current knowledge base to answer this safely. "
    "Please consult a qualified agricultural expert or local extension service."
)


@dataclass
class StructuredAdvisoryResult:
    """Advisory result containing response data and retrieved evidence."""
    data: Dict[str, Any]
    retrieved_evidence: List[Evidence] = field(default_factory=list)

    @property
    def evidence_status(self) -> str:
        if self.data.get("status") in ("INSUFFICIENT_EVIDENCE", "LLM_ERROR", "INVALID_MODEL_OUTPUT"):
            return "insufficient"
        return "sufficient"

    @property
    def status(self) -> str:
        return str(self.data.get("status", "SUCCESS"))

    @property
    def answer(self) -> str:
        """Render human-friendly formatted advisory text for UI display."""
        if self.data.get("status") == "LLM_ERROR":
            return str(self.data.get("message", "The retrieved evidence was available, but the language model could not generate the advisory."))
        if self.evidence_status == "insufficient":
            return str(self.data.get("message", SAFE_FALLBACK))

        lines = []
        if self.data.get("possible_issue"):
            lines.append(f"**Possible Issue:** {self.data['possible_issue']}\n")

        practices = self.data.get("evidence_based_practices", [])
        if practices:
            lines.append("**Evidence-Based Practices:**")
            for p in practices:
                lines.append(f"- {p}")
            lines.append("")

        if self.data.get("why_relevant"):
            lines.append(f"**Why Relevant:** {self.data['why_relevant']}\n")

        precautions = self.data.get("precautions", [])
        if precautions:
            lines.append("**Precautions:**")
            for pr in precautions:
                lines.append(f"- {pr}")
            lines.append("")

        limitations = self.data.get("limitations", [])
        if limitations:
            lines.append(f"**Notice:** {limitations[0]}\n")

        return "\n".join(lines).strip()

    @property
    def sources(self) -> List[Dict[str, Any]]:
        return self.data.get("sources", [])

    @property
    def confidence(self) -> str:
        return str(self.data.get("confidence", "Medium"))

    def get(self, key: str, default: Any = None) -> Any:
        """Provide dict-like get method for backwards compatibility."""
        return self.data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return self.data.copy()

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        return key in self.data


def determine_confidence(filtered_evidence: List[Evidence]) -> str:
    """Categorically evaluate confidence based on retrieved evidence depth and distances.
    
    High: Multiple relevant chunks (>= 2) with strong similarity (distance <= 0.28).
    Medium: At least one relevant chunk or moderate similarity.
    Low: Borderline or sparse evidence.
    """
    if not filtered_evidence:
        return "Low"

    close_chunks = [e for e in filtered_evidence if e.distance <= 0.28]
    if len(close_chunks) >= 2:
        return "High"
    if len(filtered_evidence) >= 1:
        return "Medium"
    return "Low"


def extract_verified_sources(filtered_evidence: List[Evidence]) -> List[Dict[str, Any]]:
    """Extract authoritative citation metadata directly from verified retrieval evidence."""
    seen = set()
    sources: List[Dict[str, Any]] = []

    for ev in filtered_evidence:
        title = ev.document_title
        page = ev.page
        section = ev.section
        key = (title, page, section)
        if key not in seen:
            seen.add(key)
            sources.append({
                "title": title,
                "page": page,
                "section": section,
            })

    return sources


def answer_query(
    query: str,
    retriever: Optional[Any] = None,
    llm: Optional[Any] = None,
    threshold: Optional[float] = None,
    crop: str = "",
    problem: str = "",
    preference: str = "",
) -> StructuredAdvisoryResult:
    """Execute the grounded agricultural advisory RAG pipeline.
    
    1. Embeds and retrieves evidence from ChromaDB.
    2. Evaluates evidence against the similarity threshold.
    3. If below threshold: returns INSUFFICIENT_EVIDENCE without invoking the LLM.
    4. If sufficient: constructs prompt with XML delimiters and invokes IBM Granite.
    5. Parses structured JSON response.
    6. Overrides/attaches verified source metadata from retrieval.
    7. Returns StructuredAdvisoryResult.
    """
    eff_threshold = threshold if threshold is not None else config.SIMILARITY_THRESHOLD
    eff_retriever = retriever if retriever is not None else ChromaRetriever()
    eff_llm = llm if llm is not None else get_llm_provider()

    # Build contextual query if crop/problem are provided
    search_terms = []
    if crop:
        search_terms.append(crop)
    if problem:
        search_terms.append(problem)
    search_query = f"{' '.join(search_terms)}: {query}" if search_terms else query

    retrieval_result = eff_retriever.retrieve(search_query)

    # Threshold Check: if retrieval is insufficient, do not call Granite
    if not has_sufficient_evidence(retrieval_result, eff_threshold):
        insufficient_data = InsufficientEvidenceOutput().to_dict()
        return StructuredAdvisoryResult(insufficient_data, retrieved_evidence=[])

    filtered_evidence = retrieval_result.filtered(eff_threshold)
    verified_sources = extract_verified_sources(filtered_evidence)
    calibrated_confidence = determine_confidence(filtered_evidence)

    # Format context and prompt
    context_str = format_evidence_context(filtered_evidence)
    user_prompt = build_user_prompt(
        query=query,
        context=context_str,
        crop=crop,
        problem=problem,
        preference=preference,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Generate response via LLM with error resilience
    try:
        raw_answer = eff_llm.generate(messages, temperature=config.LLM_TEMPERATURE)
        advisory_obj = parse_advisory_response(raw_answer, fallback_sources=verified_sources)
        advisory_data = advisory_obj.to_dict()
        # Always enforce verified retrieval metadata for citations
        advisory_data["sources"] = verified_sources
        advisory_data["confidence"] = calibrated_confidence
        return StructuredAdvisoryResult(advisory_data, retrieved_evidence=filtered_evidence)

    except Exception as exc:
        logger.error(f"Inference error during IBM Granite advisory generation: {exc}")
        # Safe fallback advisory on LLM failure, preserving verified sources
        fallback_data = {
            "status": "LLM_ERROR",
            "message": "The retrieved evidence was available, but the language model could not generate the advisory. Please retry or consult the source documents directly.",
            "possible_issue": "",
            "evidence_based_practices": [],
            "why_relevant": "",
            "precautions": [],
            "sources": verified_sources,
            "confidence": calibrated_confidence,
            "limitations": [PROTOTYPE_LIMITATION],
        }
        return StructuredAdvisoryResult(fallback_data, retrieved_evidence=filtered_evidence)
