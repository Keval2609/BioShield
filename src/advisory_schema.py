"""Structured data schemas and response parsing for BioShield AI advisories."""

from dataclasses import asdict, dataclass, field
import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("bioshield.advisory_schema")

VALID_CONFIDENCE_LEVELS = {"High", "Medium", "Low"}

PROTOTYPE_LIMITATION = (
    "BioShield AI is an AI-powered agricultural information and decision-support prototype, "
    "not an autonomous agricultural diagnosis or treatment system. Consult a local Krishi Vigyan Kendra (KVK) "
    "or extension expert for field validation."
)


@dataclass
class AdvisoryOutput:
    """Structured response container for grounded natural farming advisory."""
    possible_issue: str = ""
    evidence_based_practices: List[str] = field(default_factory=list)
    why_relevant: str = ""
    precautions: List[str] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    confidence: str = "Medium"
    limitations: List[str] = field(default_factory=lambda: [PROTOTYPE_LIMITATION])

    def to_dict(self) -> Dict[str, Any]:
        """Convert advisory to JSON-serializable dictionary."""
        data = asdict(self)
        if not data["limitations"]:
            data["limitations"] = [PROTOTYPE_LIMITATION]
        return data


@dataclass
class InsufficientEvidenceOutput:
    """Safe fallback response when retrieved context falls below similarity threshold."""
    status: str = "INSUFFICIENT_EVIDENCE"
    message: str = (
        "I could not find sufficient verified information in the current knowledge base to answer this safely. "
        "Please consult a qualified agricultural expert or local extension service."
    )
    sources: List[Dict[str, Any]] = field(default_factory=list)
    confidence: str = "Low"

    def to_dict(self) -> Dict[str, Any]:
        """Convert insufficient evidence response to dictionary."""
        return asdict(self)


def strip_markdown_fences(text: str) -> str:
    """Remove markdown json/code fences if present in text."""
    clean = text.strip()
    # Match ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return clean


def parse_advisory_response(
    raw_text: str,
    fallback_sources: Optional[List[Dict[str, Any]]] = None,
) -> AdvisoryOutput:
    """Parse raw LLM output into validated AdvisoryOutput object with error resilience.
    
    Attempts JSON parsing first, falling back to heuristic section extraction if needed.
    """
    cleaned = strip_markdown_fences(raw_text)
    sources = fallback_sources or []

    # Attempt to extract JSON from the text
    json_candidate = None
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_candidate = cleaned[first_brace : last_brace + 1]

    if json_candidate:
        try:
            data = json.loads(json_candidate)
            if isinstance(data, dict):
                practices = data.get("evidence_based_practices") or data.get("recommended_sustainable_practices") or []
                if isinstance(practices, str):
                    practices = [practices]

                precautions = data.get("precautions") or []
                if isinstance(precautions, str):
                    precautions = [precautions]

                limitations = data.get("limitations") or [PROTOTYPE_LIMITATION]
                if isinstance(limitations, str):
                    limitations = [limitations]

                conf = str(data.get("confidence", "Medium")).strip().capitalize()
                if conf not in VALID_CONFIDENCE_LEVELS:
                    conf = "Medium"

                parsed_sources = data.get("sources")
                if not parsed_sources or not isinstance(parsed_sources, list):
                    parsed_sources = sources

                return AdvisoryOutput(
                    possible_issue=str(data.get("possible_issue", "")).strip(),
                    evidence_based_practices=[str(p).strip() for p in practices if str(p).strip()],
                    why_relevant=str(data.get("why_relevant", data.get("why", ""))).strip(),
                    precautions=[str(pr).strip() for pr in precautions if str(pr).strip()],
                    sources=parsed_sources,
                    confidence=conf,
                    limitations=[str(lim).strip() for lim in limitations if str(lim).strip()],
                )
        except Exception as err:
            logger.warning(f"JSON parsing failed on candidate text: {err}")

    # Fallback heuristic parsing for plain-text or malformed responses
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    possible_issue = lines[0] if lines else "Guidance based on natural farming context."
    why_relevant = " ".join(lines[1:]) if len(lines) > 1 else cleaned

    return AdvisoryOutput(
        possible_issue=possible_issue,
        evidence_based_practices=[cleaned] if cleaned else [],
        why_relevant=why_relevant,
        precautions=["Always observe crop response and verify with local agricultural extension."],
        sources=sources,
        confidence="Medium",
        limitations=[PROTOTYPE_LIMITATION],
    )
