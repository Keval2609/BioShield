"""Tests for prompt engineering, XML delimiters, and grounding rules."""

from src.prompts import SYSTEM_PROMPT, build_user_prompt, format_evidence_context
from src.retriever import Evidence


def test_system_prompt_enforces_project_positioning_and_grounding():
    assert "AI-powered agricultural information and decision-support prototype" in SYSTEM_PROMPT
    assert "not an autonomous agricultural diagnosis or treatment system" in SYSTEM_PROMPT
    assert "Do not claim a pest/disease diagnosis with certainty" in SYSTEM_PROMPT
    assert "application rates" in SYSTEM_PROMPT
    assert "Natural farming" in SYSTEM_PROMPT


def test_user_prompt_includes_xml_delimiters():
    evidence = [
        Evidence(
            text="Neem seed kernel extract (NSKE 5%) controls bollworms.",
            metadata={
                "document_title": "Field Guide",
                "page": 45,
                "section": "Botanical Preparations",
                "farming_approach": "natural_farming",
            },
            distance=0.18,
        )
    ]
    context = format_evidence_context(evidence)
    prompt = build_user_prompt(
        query="How to manage bollworms?",
        context=context,
        crop="Cotton",
        problem="Bollworm holes in bolls",
        preference="Natural farming",
    )
    assert "<RETRIEVED_CONTEXT>" in prompt
    assert "</RETRIEVED_CONTEXT>" in prompt
    assert "<USER_QUERY>" in prompt
    assert "</USER_QUERY>" in prompt
    assert "Target Crop: Cotton" in prompt
    assert "NSKE 5%" in prompt
