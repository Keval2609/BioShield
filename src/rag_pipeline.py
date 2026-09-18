"""RAG pipeline connecting retrieval to IBM Granite generation."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src import config
from src.llm import GraniteLLM
from src.prompts import SYSTEM_PROMPT, build_user_prompt, format_evidence_context
from src.retriever import ChromaRetriever, Evidence, has_sufficient_evidence

SAFE_FALLBACK = (
    "I could not find sufficient verified information in the current knowledge base to answer this safely. "
    "Please consult a qualified agricultural expert or local extension service."
)


@dataclass
class PipelineResult:
    """Encapsulates the response from the BioShield RAG pipeline."""
    answer: str
    evidence_status: str  # "sufficient" or "insufficient"
    sources: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_evidence: List[Evidence] = field(default_factory=list)


def answer_query(
    query: str,
    retriever: Optional[Any] = None,
    llm: Optional[Any] = None,
    threshold: Optional[float] = None,
    crop: str = "",
    problem: str = "",
    preference: str = "",
) -> PipelineResult:
    """Execute the grounded agricultural advisory pipeline.
    
    1. Retrieve relevant agricultural documents from vector store.
    2. Evaluate evidence sufficiency against distance threshold.
    3. If insufficient: return SAFE_FALLBACK without calling the LLM.
    4. If sufficient: build prompt from verified context and invoke IBM Granite.
    """
    eff_threshold = threshold if threshold is not None else config.SIMILARITY_THRESHOLD
    eff_retriever = retriever if retriever is not None else ChromaRetriever()
    eff_llm = llm if llm is not None else GraniteLLM()

    # Formulate query for retriever combining crop and problem context if provided
    search_query = query
    context_prefix = []
    if crop:
        context_prefix.append(crop)
    if problem:
        context_prefix.append(problem)
    if context_prefix:
        search_query = f"{' '.join(context_prefix)}: {query}"

    retrieval_result = eff_retriever.retrieve(search_query)

    if not has_sufficient_evidence(retrieval_result, eff_threshold):
        return PipelineResult(
            answer=SAFE_FALLBACK,
            evidence_status="insufficient",
            sources=[],
            retrieved_evidence=[],
        )

    filtered_evidence = retrieval_result.filtered(eff_threshold)
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

    answer = eff_llm.generate(messages, temperature=0.0)
    sources = [item.metadata for item in filtered_evidence]

    return PipelineResult(
        answer=answer,
        evidence_status="sufficient",
        sources=sources,
        retrieved_evidence=filtered_evidence,
    )
