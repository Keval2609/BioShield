"""End-to-end integration tests for BioShield AI retrieval."""

import pytest
from src import config
from src.retriever import ChromaRetriever


def test_e2e_retrieval_finds_natural_farming_advisory():
    retriever = ChromaRetriever()
    coll = retriever._get_collection()
    if coll is None or coll.count() == 0:
        pytest.skip("ChromaDB collection is empty. Run python ingest.py first.")

    result = retriever.retrieve("How to manage bollworms and aphids using botanical extracts?")
    assert len(result.evidence) > 0

    top_evidence = result.evidence[0]
    assert top_evidence.page >= 1
    assert top_evidence.document_title != ""
    assert top_evidence.farming_approach == "natural_farming"
    assert top_evidence.source_file != ""
    assert 0.0 <= top_evidence.distance <= 1.0
    assert 0.0 <= top_evidence.similarity_score <= 1.0


def test_e2e_retrieval_insufficient_evidence_for_irrelevant_query():
    retriever = ChromaRetriever()
    coll = retriever._get_collection()
    if coll is None or coll.count() == 0:
        pytest.skip("ChromaDB collection is empty. Run python ingest.py first.")

    # Extremely irrelevant query outside agriculture
    result = retriever.retrieve("Quantum chromodynamics gluon plasma hadronization in particle physics")
    # With a strict threshold (or checking distance), it should recognize weak/insufficient evidence
    filtered = result.filtered(threshold=0.25)
    assert len(filtered) == 0
