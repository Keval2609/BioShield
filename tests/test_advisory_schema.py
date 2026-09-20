"""Tests for structured advisory output schema and response parser."""

from src.advisory_schema import (
    AdvisoryOutput,
    InsufficientEvidenceOutput,
    parse_advisory_response,
)


def test_parse_valid_json_response():
    raw_json = """
    {
        "possible_issue": "The symptoms may be consistent with gram pod borer.",
        "evidence_based_practices": [
            "Install pheromone traps at 10-12 traps per hectare",
            "Spray 5% Neem Seed Kernel Extract (NSKE)"
        ],
        "why_relevant": "Traps monitor moth populations while NSKE repels oviposition.",
        "precautions": ["Spray NSKE during evening hours."],
        "sources": [{"title": "Field Guide", "page": "45", "section": "Botanical"}],
        "confidence": "High",
        "limitations": ["Guidance is preventive and based on natural farming protocols."]
    }
    """
    advisory = parse_advisory_response(raw_json)
    assert advisory.confidence == "High"
    assert len(advisory.evidence_based_practices) == 2
    assert "gram pod borer" in advisory.possible_issue
    assert advisory.sources[0]["title"] == "Field Guide"
    assert advisory.to_dict()["confidence"] == "High"


def test_parse_markdown_fenced_json_response():
    raw_markdown = """```json
    {
        "possible_issue": "Symptoms may indicate aphid infestation.",
        "evidence_based_practices": ["Spray Dashaparni ark"],
        "why_relevant": "Contains botanical repellents.",
        "precautions": ["Use fresh preparation."],
        "sources": [],
        "confidence": "Medium",
        "limitations": ["Decision support only."]
    }
    ```"""
    advisory = parse_advisory_response(raw_markdown)
    assert advisory.possible_issue == "Symptoms may indicate aphid infestation."
    assert advisory.confidence == "Medium"
    assert advisory.evidence_based_practices == ["Spray Dashaparni ark"]


def test_parse_malformed_text_response_provides_safe_fallback():
    raw_text = "Some unformatted text from model describing neem spray without valid JSON."
    fallback_sources = [{"title": "Field Guide", "page": 10, "section": "Neem"}]
    advisory = parse_advisory_response(raw_text, fallback_sources=fallback_sources)
    assert advisory.status == "INVALID_MODEL_OUTPUT"
    assert advisory.confidence == "Low"
    assert advisory.sources == fallback_sources


def test_insufficient_evidence_output_structure():
    insuf = InsufficientEvidenceOutput()
    d = insuf.to_dict()
    assert d["status"] == "INSUFFICIENT_EVIDENCE"
    assert "sufficient verified information" in d["message"]
    assert d["confidence"] == "Low"
    assert d["sources"] == []
