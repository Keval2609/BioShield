"""Automated UI tests for BioShield AI Streamlit application using AppTest."""

from unittest.mock import patch
import pytest
from streamlit.testing.v1 import AppTest

from src.rag_pipeline import StructuredAdvisoryResult
from src.retriever import Evidence


@pytest.fixture
def mock_evidence():
    return [
        Evidence(
            text="Agniastra and Neem seed kernel extract effectively control sucking pests in cotton.",
            distance=0.18,
            metadata={
                "document_title": "Field_Guide_for_Natural_Farming.pdf",
                "page": 14,
                "section": "Botanical Preparations",
            },
        )
    ]


def test_app_ui_renders_initial_state():
    """Verify Streamlit UI renders header, disclaimer, sidebar, inputs, and demo buttons without error."""
    at = AppTest.from_file("../app.py", default_timeout=30)
    at.run()

    assert not at.exception
    # Verify title and subtitle in markdown
    markdown_texts = [str(m.value) for m in at.markdown]
    assert any("BioShield AI" in t for t in markdown_texts)
    assert any("Natural Farming Pest Advisory Assistant" in t for t in markdown_texts)

    # Verify prominent disclaimer
    warning_texts = [str(w.value) for w in at.warning]
    assert any("Important Disclaimer" in t for t in warning_texts)

    # Verify sidebar
    sidebar_headers = [str(h.value) for h in at.sidebar.header]
    assert any("Responsible AI" in h for h in sidebar_headers)

    # Verify 3 demo mode buttons exist
    assert len(at.button) >= 4  # 3 preset buttons + 1 submit button

    # Verify input fields
    assert len(at.text_input) >= 2  # crop, problem
    assert len(at.selectbox) >= 1   # farming approach
    assert len(at.text_area) >= 1   # question


def test_app_ui_example_preset_button_populates_inputs():
    """Verify clicking an example query preset populates the form fields."""
    at = AppTest.from_file("../app.py", default_timeout=30)
    at.run()

    # Click the first example button
    at.button[0].click().run()

    assert not at.exception
    assert at.session_state["crop_field"] == "Various"
    assert "pests" in at.session_state["problem_field"].lower()
    assert at.session_state["approach_field"] == "Natural farming"
    assert len(at.session_state["question_field"]) > 0


def test_app_ui_renders_structured_advisory_when_evidence_sufficient(mock_evidence):
    """Verify UI displays the 7 structured sections and retrieved evidence on successful generation."""
    mock_result_data = {
        "possible_issue": "Aphids and whiteflies infesting foliage",
        "evidence_based_practices": ["Neem Seed Kernel Extract (NSKE 5%)", "Agniastra spray"],
        "why_relevant": "Botanical extracts act as feeding deterrents and repellent against soft-bodied insects.",
        "precautions": ["Spray during evening hours", "Avoid excessive application in hot sun"],
        "sources": [
            {"title": "Field_Guide_for_Natural_Farming.pdf", "page": 14, "section": "Botanical Preparations"}
        ],
        "confidence": "High",
        "limitations": "Prototype decision-support advisory. Verify with local KVK experts.",
    }
    mock_result = StructuredAdvisoryResult(mock_result_data, retrieved_evidence=mock_evidence)

    with patch("src.rag_pipeline.answer_query", return_value=mock_result):
        at = AppTest.from_file("../app.py", default_timeout=30)
        at.run()

        # Fill inputs and submit
        at.text_input(key="crop_field").input("Cotton")
        at.text_input(key="problem_field").input("Insects on leaves")
        at.text_area(key="question_field").input("How to treat sustainably?")
        submit_btn = [b for b in at.button if "Generate" in str(b.label)][0]
        submit_btn.click().run()

        assert not at.exception
        markdown_texts = " ".join([str(m.value) for m in at.markdown])

        # Verify the 7 sections are rendered
        assert "1. Possible Issue" in markdown_texts
        assert "2. Evidence-Based Sustainable Practices" in markdown_texts
        assert "3. Why This May Be Relevant" in markdown_texts
        assert "4. Precautions" in markdown_texts
        assert "5. Sources" in markdown_texts
        assert "6. Confidence" in markdown_texts
        assert "7. Limitations" in markdown_texts
        assert "High Confidence" in markdown_texts
        assert any("Retrieved Evidence" in str(e.label) for e in at.expander)


def test_app_ui_renders_insufficient_evidence_safe_fallback():
    """Verify UI shows safe refusal notice when evidence is insufficient."""
    mock_insufficient_data = {
        "status": "INSUFFICIENT_EVIDENCE",
        "message": "I could not find sufficient verified information in the current knowledge base to answer this safely.",
        "sources": [],
        "confidence": "Low",
    }
    mock_result = StructuredAdvisoryResult(mock_insufficient_data, retrieved_evidence=[])

    with patch("src.rag_pipeline.answer_query", return_value=mock_result):
        at = AppTest.from_file("../app.py", default_timeout=30)
        at.run()

        # Fill inputs and submit
        at.text_input(key="crop_field").input("Tomato")
        at.text_input(key="problem_field").input("Chemical schedule")
        submit_btn = [b for b in at.button if "Generate" in str(b.label)][0]
        submit_btn.click().run()

        assert not at.exception
        warning_texts = [str(w.value) for w in at.warning]
        info_texts = [str(i.value) for i in at.info]

        assert any("could not find sufficient verified information" in t for t in warning_texts)
        assert any("consult a qualified agricultural expert" in t for t in info_texts)
