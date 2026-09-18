from src.retriever import Evidence, RetrievalResult, has_sufficient_evidence


def test_has_sufficient_evidence_filters_by_distance_threshold():
    result = RetrievalResult(
        evidence=[
            Evidence(text="supported", metadata={"title": "Guide"}, distance=0.2),
            Evidence(text="weak", metadata={"title": "Other"}, distance=0.8),
        ]
    )

    assert has_sufficient_evidence(result, threshold=0.35)
    assert [item.text for item in result.filtered(0.35)] == ["supported"]


def test_no_evidence_is_insufficient():
    result = RetrievalResult(evidence=[])
    assert not has_sufficient_evidence(result, threshold=0.35)
