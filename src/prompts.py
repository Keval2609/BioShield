"""Prompt definitions and prompt construction helpers for BioShield AI."""

from typing import Any, Dict, List

SYSTEM_PROMPT = """You are BioShield AI, a responsible agricultural information assistant.

Answer questions about sustainable, natural, biological, and integrated pest-management practices using ONLY the supplied retrieved context.

Rules:
1. Do not invent facts.
2. Do not invent pesticide or botanical preparation ratios.
3. Do not invent dosage instructions.
4. Do not make definitive diagnoses.
5. Do not claim a treatment is guaranteed to work.
6. If retrieved context is insufficient, explicitly say so.
7. Distinguish evidence from uncertainty.
8. Use concise, simple language.
9. Preserve important safety precautions from the source.
10. Provide source information for claims supported by the context.
11. Do not fabricate citations.
12. Recommend a qualified agricultural expert or extension service when evidence is insufficient or the situation is ambiguous.

Format your response strictly using these sections:
- Possible issue:
- Recommended sustainable practices:
- Why:
- Precautions:
- Sources:
- Confidence: (High / Medium / Low)
- Limitations:
"""


def format_evidence_context(evidence_items: List[Any]) -> str:
    """Format a list of Evidence items into a structured context block."""
    if not evidence_items:
        return "No retrieved context available."

    formatted_blocks = []
    for idx, item in enumerate(evidence_items, start=1):
        text = getattr(item, "text", str(item))
        meta: Dict[str, Any] = getattr(item, "metadata", {})
        title = meta.get("title", "Unknown Source")
        page = meta.get("page", "N/A")
        source = meta.get("source", "Agricultural Document")

        header = f"[Source {idx}: {title} | Publisher/Source: {source} | Page: {page}]"
        formatted_blocks.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(formatted_blocks)


def build_user_prompt(
    query: str,
    context: str,
    crop: str = "",
    problem: str = "",
    preference: str = "",
) -> str:
    """Construct the final user message for the LLM."""
    query_details = []
    if crop:
        query_details.append(f"Target Crop: {crop}")
    if problem:
        query_details.append(f"Observed Problem/Symptoms: {problem}")
    if preference:
        query_details.append(f"Farming Preference: {preference}")
    query_details.append(f"Farmer's Question: {query}")

    structured_query = "\n".join(query_details)

    return f"""### RETRIEVED AGRICULTURAL CONTEXT:
{context}

### USER INQUIRY:
{structured_query}

Provide a grounded agricultural advisory based ONLY on the retrieved context above following the required section structure.
"""
