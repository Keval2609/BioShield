import pytest
from src.retriever import Evidence, RetrievalResult, has_sufficient_evidence


def test_has_sufficient_evidence_filters_by_distance_threshold():
    result = RetrievalResult(
        evidence=[
            Evidence(text="supported", metadata={"document_title": "Guide", "page": 3}, distance=0.2),
            Evidence(text="weak", metadata={"document_title": "Other", "page": 10}, distance=0.8),
        ]
    )

    assert has_sufficient_evidence(result, threshold=0.35)
    assert [item.text for item in result.filtered(0.35)] == ["supported"]


def test_no_evidence_is_insufficient():
    result = RetrievalResult(evidence=[])
    assert not has_sufficient_evidence(result, threshold=0.35)
    assert result.get_status(threshold=0.35) == "INSUFFICIENT_EVIDENCE"


def test_evidence_structure_exposes_source_fields():
    ev = Evidence(
        text="Neem cake application repels soil pests.",
        metadata={
            "document_id": "field_guide",
            "document_title": "Field Guide for Natural Farming",
            "source_file": "Field_Guide_for_Natural_Farming.pdf",
            "page": 45,
            "section": "Botanical Preparations",
            "source_type": "government_field_guide",
            "farming_approach": "natural_farming",
            "region": "India",
            "topic": "botanical_preparations",
        },
        distance=0.18,
    )
    assert ev.document_title == "Field Guide for Natural Farming"
    assert ev.source_file == "Field_Guide_for_Natural_Farming.pdf"
    assert ev.page == 45
    assert ev.section == "Botanical Preparations"
    assert ev.farming_approach == "natural_farming"
    assert ev.similarity_score == pytest.approx(0.82, 0.01)


def test_retriever_natural_farming_priority():
    result = RetrievalResult(
        evidence=[
            Evidence(
                text="Apply chemical insecticide spray",
                metadata={"document_title": "Conventional Manual", "farming_approach": "conventional/IPM", "page": 10},
                distance=0.15,
            ),
            Evidence(
                text="Apply Jeevamrit and botanical kashayam",
                metadata={"document_title": "Field Guide", "farming_approach": "natural_farming", "page": 20},
                distance=0.20,
            ),
        ]
    )

    prioritized = result.prioritized()
    # Natural farming evidence must come first despite slightly higher distance
    assert prioritized[0].farming_approach == "natural_farming"
    assert "Jeevamrit" in prioritized[0].text
