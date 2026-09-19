"""Final evaluation tests for BioShield AI.

Verifies the 5 core testing requirements from the final engineering pass.
These are end-to-end integration tests that require a populated ChromaDB and a Granite API key.
"""

import os
import pytest

from src import config
from src.rag_pipeline import answer_query
from src.retriever import ChromaRetriever
from src.llm import OllamaProvider

pytestmark = pytest.mark.skipif(
    not config.OLLAMA_BASE_URL,
    reason="Ollama Base URL is required for final evaluation tests"
)

@pytest.fixture(scope="module")
def retriever():
    retriever = ChromaRetriever()
    coll = retriever._get_collection()
    if coll is None or coll.count() == 0:
        pytest.skip("ChromaDB knowledge base is not populated. Run python ingest.py first.")
    return retriever

@pytest.fixture(scope="module")
def llm():
    return OllamaProvider()

def test_1_supported_query(retriever, llm):
    """TEST 1 — SUPPORTED QUERY"""
    result = answer_query(
        query="What natural farming practices can help manage sucking pests?",
        retriever=retriever,
        llm=llm
    )
    
    assert result.evidence_status == "sufficient"
    assert result.status != "INSUFFICIENT_EVIDENCE"
    
    # Must have evidence-based practices and sources
    assert len(result["evidence_based_practices"]) > 0
    assert len(result["sources"]) > 0
    
    # Ensure source attribution is not fabricated
    for source in result["sources"]:
        assert "title" in source
        assert "page" in source
        
def test_2_crop_specific_query(retriever, llm):
    """TEST 2 — CROP-SPECIFIC QUERY"""
    result = answer_query(
        query="How to manage bollworms and aphids using botanical extracts in Cotton?",
        crop="Cotton",
        retriever=retriever,
        llm=llm
    )

    assert result.evidence_status == "sufficient"
    assert len(result["evidence_based_practices"]) > 0
    assert len(result["sources"]) > 0
    assert "cotton" in str(result).lower() or "bollworm" in str(result).lower() or "aphid" in str(result).lower()

def test_3_unsupported_query(retriever, llm):
    """TEST 3 — UNSUPPORTED QUERY"""
    result = answer_query(
        query="What is the best synthetic pesticide treatment for a crop not covered by natural farming?",
        preference="Conventional chemical",
        retriever=retriever,
        llm=llm,
        threshold=0.35
    )
    
    # Should safely fail the evidence threshold or fall back
    if result.evidence_status == "sufficient":
        # If somehow it passed threshold, it still shouldn't hallucinate a chemical
        assert len(result["sources"]) > 0
        answer_text = result.answer.lower()
        assert "synthetic" not in answer_text or "not recommend" in answer_text
    else:
        assert result.evidence_status == "insufficient"

def test_4_ambiguous_symptom(retriever, llm):
    """TEST 4 — AMBIGUOUS SYMPTOM"""
    result = answer_query(
        query="My crop leaves are turning yellow. What disease is it?",
        retriever=retriever,
        llm=llm
    )
    
    answer_text = result.answer.lower()
    
    # Must NOT make definitive diagnosis
    assert "your crop has" not in answer_text
    assert "this is definitely" not in answer_text
    
    # Check for non-definitive phrasing if evidence was found
    if result.evidence_status == "sufficient":
        assert "may be" in answer_text or "can be" in answer_text or "consult" in answer_text or "possible issue" in answer_text
    
def test_5_unsupported_dosage(retriever, llm):
    """TEST 5 — UNSUPPORTED DOSAGE / PREPARATION REQUEST"""
    result = answer_query(
        query="Give me the exact preparation ratio for an obscure natural pesticide 12345.",
        retriever=retriever,
        llm=llm
    )
    
    # Model should refuse to hallucinate an exact preparation ratio
    answer_text = result.answer.lower()
    if result.evidence_status == "sufficient":
        assert "12345" not in answer_text or "not mentioned" in answer_text or "consult" in answer_text
    else:
        assert result.evidence_status == "insufficient"
