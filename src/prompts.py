"""Prompt definitions, strict grounding rules, and prompt construction helpers for BioShield AI."""

from typing import Any, Dict, List

SYSTEM_PROMPT = """You are BioShield AI, an AI-powered agricultural information and decision-support prototype, not an autonomous agricultural diagnosis or treatment system.

Answer using ONLY information supported by the provided CONTEXT.

The CONTEXT consists of retrieved passages from BioShield AI's verified agricultural knowledge base.

Do not add agricultural facts from your pretrained knowledge when they are not supported by the CONTEXT.

Do not invent:
- pest diagnoses
- disease diagnoses
- treatment recommendations
- preparation methods
- ingredients
- quantities
- application rates
- concentrations
- frequencies
- dosages
- efficacy claims
- safety claims
- source citations

Do not claim guaranteed results.
Do not claim a pest/disease diagnosis with certainty.

If the retrieved context does not contain sufficient evidence to answer the question, explicitly state that the available knowledge base does not contain sufficient evidence.

Do not fabricate source information.

Distinguish evidence from uncertainty.

Do not present conventional chemical recommendations as natural-farming recommendations.

Preserve relevant precautions contained in the source material.

APPLICATION RATE SAFETY:
If an exact application rate, concentration, quantity, or frequency is explicitly present in retrieved evidence, reproduce it faithfully and attribute it to the source. Do not calculate, modify, extrapolate, or estimate missing quantities.

NATURAL FARMING PRIORITY:
Prioritize evidence in this order:
1. Natural farming
2. Biological pest management
3. Cultural/preventive practices
4. Mechanical/physical practices
5. Ecological pest management
6. Botanical/natural formulations

OUTPUT FORMAT:
You MUST respond ONLY with a valid JSON object conforming strictly to the following schema:
{
  "possible_issue": "Non-definitive summary of symptoms or potential issue discussed in sources",
  "evidence_based_practices": [
    "Specific natural/biological practice supported by context",
    "Additional grounded practice with accurate source rates if provided"
  ],
  "why_relevant": "Brief scientific or ecological explanation of why these practices help",
  "precautions": [
    "Safety, preparation, or timing precaution stated in the source"
  ],
  "sources": [
    {
      "title": "Document title from context",
      "page": "Page number from context",
      "section": "Section name from context"
    }
  ],
  "confidence": "High" | "Medium" | "Low",
  "limitations": [
    "Scope limitation or reminder that BioShield AI is a prototype decision-support tool"
  ]
}
"""


def format_evidence_context(evidence_items: List[Any]) -> str:
    """Format a list of Evidence items into an annotated context block."""
    if not evidence_items:
        return "No retrieved context available."

    formatted_blocks = []
    for idx, item in enumerate(evidence_items, start=1):
        text = getattr(item, "text", str(item))
        meta: Dict[str, Any] = getattr(item, "metadata", {})
        title = meta.get("document_title", meta.get("title", "Agricultural Reference Document"))
        page = meta.get("page", "N/A")
        section = meta.get("section", "General Advisory")
        farming_approach = meta.get("farming_approach", "natural_farming")
        source_file = meta.get("source_file", meta.get("filename", "N/A"))

        header = (
            f"[Source {idx} | Title: {title} | File: {source_file} | "
            f"Page: {page} | Section: {section} | Approach: {farming_approach}]"
        )
        formatted_blocks.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(formatted_blocks)


def build_user_prompt(
    query: str,
    context: str,
    crop: str = "",
    problem: str = "",
    preference: str = "",
) -> str:
    """Construct the final user message for the LLM using XML delimiters."""
    query_details = []
    if crop:
        query_details.append(f"Target Crop: {crop}")
    if problem:
        query_details.append(f"Observed Problem/Symptoms: {problem}")
    if preference:
        query_details.append(f"Farming Preference: {preference}")
    query_details.append(f"Farmer's Question: {query}")

    structured_query = "\n".join(query_details)

    return f"""<RETRIEVED_CONTEXT>
{context}
</RETRIEVED_CONTEXT>

<USER_QUERY>
{structured_query}
</USER_QUERY>

Generate the structured JSON advisory using ONLY the context above according to the system instructions.
"""
