"""RAG pipeline connecting retrieval to IBM Granite grounded generation."""

import logging
from typing import Any, Dict, List, Optional

from src import config
from src.advisory_schema import (
    PROTOTYPE_LIMITATION,
    AdvisoryOutput,
    InsufficientEvidenceOutput,
    parse_advisory_response,
)
from src.llm import BaseLLM, GraniteLLM
from src.prompts import SYSTEM_PROMPT, build_user_prompt, format_evidence_context
from src.retriever import ChromaRetriever, Evidence, has_sufficient_evidence

logger = logging.getLogger("bioshield.rag_pipeline")

SAFE_FALLBACK = (
    "I could not find sufficient verified information in the current knowledge base to answer this safely. "
    "Please consult a qualified agricultural expert or local extension service."
)


class StructuredAdvisoryResult(dict):
    """Dual-access advisory result supporting dictionary indexing and object attributes."""

    def __init__(self, data: Dict[str, Any], retrieved_evidence: Optional[List[Evidence]] = None):
        super().__init__(data)
        self._retrieved_evidence = retrieved_evidence or []

    @property
    def evidence_status(self) -> str:
        if self.get("status") == "INSUFFICIENT_EVIDENCE":
            return "insufficient"
        return "sufficient"

    @property
    def status(self) -> str:
        return str(self.get("status", "SUCCESS"))

    @property
    def answer(self) -> str:
        """Render human-friendly formatted advisory text for UI display."""
        if self.evidence_status == "insufficient":
            return str(self.get("message", SAFE_FALLBACK))

        lines = []
        if self.get("possible_issue"):
            lines.append(f"**Possible Issue:** {self['possible_issue']}\n")

        practices = self.get("evidence_based_practices", [])
        if practices:
            lines.append("**Evidence-Based Practices:**")
            for p in practices:
                lines.append(f"- {p}")
            lines.append("")

        if self.get("why_relevant"):
            lines.append(f"**Why Relevant:** {self['why_relevant']}\n")

        precautions = self.get("precautions", [])
        if precautions:
            lines.append("**Precautions:**")
            for pr in precautions:
                lines.append(f"- {pr}")
            lines.append("")

        limitations = self.get("limitations", [])
        if limitations:
            lines.append(f"**Notice:** {limitations[0]}\n")

        return "\n".join(lines).strip()

    @property
    def sources(self) -> List[Dict[str, Any]]:
        return self.get("sources", [])

    @property
    def confidence(self) -> str:
        return str(self.get("confidence", "Medium"))

    @property
    def retrieved_evidence(self) -> List[Evidence]:
        return self._retrieved_evidence

    def to_dict(self) -> Dict[str, Any]:
        return dict(self)


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
    eff_llm = llm if llm is not None else GraniteLLM()

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

    # Generate response via IBM Granite with error resilience
    try:
        raw_answer = eff_llm.generate(messages, temperature=0.0)
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
            "possible_issue": "Retrieved agricultural guidance located, but automated synthesis encountered an error.",
            "evidence_based_practices": [
                f"Refer directly to {s['title']} (Page {s['page']}, Section: {s['section']})"
                for s in verified_sources[:2]
            ],
            "why_relevant": "The verified documents in the knowledge base contain relevant natural farming guidance.",
            "precautions": ["Verify recommendations with local agricultural extension officers."],
            "sources": verified_sources,
            "confidence": calibrated_confidence,
            "limitations": [PROTOTYPE_LIMITATION],
        }
        return StructuredAdvisoryResult(fallback_data, retrieved_evidence=filtered_evidence)
