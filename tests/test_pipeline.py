"""Tests for grounded RAG generation pipeline using IBM Granite."""

from unittest.mock import patch
import pytest
from src.rag_pipeline import SAFE_FALLBACK, answer_query
from src.retriever import Evidence, RetrievalResult


class FakeRetriever:
    def __init__(self, result):
        self.result = result

    def retrieve(self, query, top_k=None):
        return self.result


class FailingLLM:
    def __init__(self):
        self.called = False

    def generate(self, messages, temperature=0):
        self.called = True
        raise AssertionError("LLM must not be called without sufficient evidence")


class FakeGraniteLLM:
    def __init__(self, response_text):
        self.response_text = response_text
        self.called = False

    def generate(self, messages, temperature=0):
        self.called = True
        return self.response_text


def test_pipeline_returns_insufficient_evidence_when_below_threshold():
    llm = FailingLLM()
    result = answer_query(
        "alien Martian crop invasion",
        retriever=FakeRetriever(RetrievalResult(evidence=[])),
        llm=llm,
        threshold=0.35,
    )

    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["confidence"] == "Low"
    assert result["sources"] == []
    assert "sufficient verified information" in result["message"]
    assert result.evidence_status == "insufficient"
    assert not llm.called


def test_pipeline_generates_grounded_structured_advisory():
    evidence = Evidence(
        text="Spray 5% Neem Seed Kernel Extract (NSKE) during evening hours to repel pod borers.",
        metadata={
            "document_title": "Field Guide for Natural Farming",
            "source_file": "Field_Guide_for_Natural_Farming.pdf",
            "page": 45,
            "section": "Botanical Preparations",
            "farming_approach": "natural_farming",
        },
        distance=0.18,
    )
    llm_json = """
    {
        "possible_issue": "The symptoms may be consistent with pod borer damage.",
        "evidence_based_practices": [
            "Foliar spray of 5% Neem Seed Kernel Extract (NSKE)"
        ],
        "why_relevant": "Azadirachtin in neem acts as an oviposition deterrent and antifeedant.",
        "precautions": [
            "Apply during evening hours to prevent photodegradation."
        ],
        "sources": [
            {
                "title": "Fabricated Title",
                "page": "999",
                "section": "Invented"
            }
        ],
        "confidence": "High",
        "limitations": [
            "BioShield AI is a prototype decision-support tool."
        ]
    }
    """
    llm = FakeGraniteLLM(llm_json)
    result = answer_query(
        query="How to manage pod borer?",
        retriever=FakeRetriever(RetrievalResult(evidence=[evidence])),
        llm=llm,
        threshold=0.35,
    )

    assert llm.called
    assert result["possible_issue"] == "The symptoms may be consistent with pod borer damage."
    assert "5% Neem Seed Kernel Extract (NSKE)" in result["evidence_based_practices"][0]
    assert result["confidence"] in ["High", "Medium"]
    assert len(result["precautions"]) >= 1

    # Verified sources from retrieval metadata must be used
    sources = result["sources"]
    assert len(sources) >= 1
    assert sources[0]["title"] == "Field Guide for Natural Farming"
    assert sources[0]["page"] == 45
    assert sources[0]["section"] == "Botanical Preparations"

    # Backward compatibility checks
    assert result.evidence_status == "sufficient"
    assert len(result.answer) > 0


def test_pipeline_handles_llm_runtime_error_gracefully():
    evidence = Evidence(
        text="Intercropping pigeonpea with sorghum reduces pod borer incidence.",
        metadata={
            "document_title": "Field Guide",
            "source_file": "Field_Guide.pdf",
            "page": 22,
            "section": "Cultural Practices",
        },
        distance=0.20,
    )

    class CrashingLLM:
        def generate(self, messages, temperature=0):
            raise RuntimeError("IBM Granite connection timed out after 45s.")

    result = answer_query(
        query="Tell me about sorghum intercropping",
        retriever=FakeRetriever(RetrievalResult(evidence=[evidence])),
        llm=CrashingLLM(),
        threshold=0.35,
    )

    # Must not crash, should return structured safe fallback
    assert "possible_issue" in result or "message" in result
    assert result["confidence"] in ["Medium", "Low"]
    assert len(result["sources"]) >= 1
