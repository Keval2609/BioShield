"""End-to-end integration tests for the Grounded RAG generation pipeline."""

import json
import pytest
from typing import Dict, List
from src.rag_pipeline import answer_query

class FakeLLM:
    """Mock LLM implementation for tests."""
    def __init__(self, response: str = "{}"):
        self.response = response
        self.last_messages: List[Dict[str, str]] = []

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        self.last_messages = messages
        return self.response
from src.retriever import ChromaRetriever

MOCK_LLM_ADVISORY = json.dumps({
    "possible_issue": "The symptoms may be consistent with insect pest pressure on cotton crops.",
    "evidence_based_practices": [
        "Foliar application of Neem Seed Kernel Extract (NSKE 5%)",
        "Erection of bird perches at 10-15 per acre to encourage insectivorous birds"
    ],
    "why_relevant": "Neem preparations act as antifeedants and deterrents without harming beneficial parasitoids.",
    "precautions": [
        "Prepare NSKE freshly and spray during early morning or late evening."
    ],
    "sources": [
        {"title": "Fabricated Title", "page": "0", "section": "Fake"}
    ],
    "confidence": "High",
    "limitations": [
        "BioShield AI is an AI-powered agricultural information and decision-support prototype."
    ]
})


def test_live_chromadb_with_mock_llm_pipeline():
    retriever = ChromaRetriever()
    coll = retriever._get_collection()
    if coll is None or coll.count() == 0:
        pytest.skip("ChromaDB knowledge base is not populated. Run python ingest.py first.")

    mock_llm = FakeLLM(response=MOCK_LLM_ADVISORY)

    result = answer_query(
        query="What natural botanical formulation can I spray to control caterpillar pests?",
        crop="Cotton",
        problem="Leaf eating caterpillars",
        preference="Natural farming only",
        retriever=retriever,
        llm=mock_llm,
        threshold=0.55,
    )

    # 1. Verification of Status & Content
    assert result.status != "INSUFFICIENT_EVIDENCE"
    assert result.evidence_status == "sufficient"
    assert result["possible_issue"] == "The symptoms may be consistent with insect pest pressure on cotton crops."
    assert len(result["evidence_based_practices"]) == 2
    assert "Neem Seed Kernel Extract" in result["evidence_based_practices"][0]

    # 2. Source citation verification: LLM's fake source MUST be replaced by verified retrieval metadata
    assert len(result["sources"]) > 0
    top_source = result["sources"][0]
    assert top_source["title"] != "Fabricated Title"
    assert top_source["page"] >= 1
    assert "section" in top_source

    # 3. Categorical confidence
    assert result["evidence_confidence"] in ["High", "Medium"]

    # 4. Delimiters verified in prompt sent to LLM
    prompt_sent = mock_llm.last_messages[1]["content"]
    assert "<RETRIEVED_CONTEXT>" in prompt_sent
    assert "</RETRIEVED_CONTEXT>" in prompt_sent
    assert "<USER_QUERY>" in prompt_sent
    assert "</USER_QUERY>" in prompt_sent
    assert "Target Crop: Cotton" in prompt_sent
    assert "Farming Preference: Natural farming only" in prompt_sent


def test_live_chromadb_insufficient_evidence_gating():
    retriever = ChromaRetriever()
    coll = retriever._get_collection()
    if coll is None or coll.count() == 0:
        pytest.skip("ChromaDB knowledge base is not populated. Run python ingest.py first.")

    class GuardedLLM:
        def generate(self, messages, temperature=0):
            raise AssertionError("LLM must NEVER be called when retrieval evidence is insufficient")

    result = answer_query(
        query="Explain cosmological inflation and the cosmic microwave background radiation",
        retriever=retriever,
        llm=GuardedLLM(),
        threshold=0.20,
    )

    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["evidence_confidence"] == "Low"
    assert result["sources"] == []
    assert result.evidence_status == "insufficient"
