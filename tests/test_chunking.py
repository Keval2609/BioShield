from src.chunking import (
    chunk_page_text,
    chunk_text,
    clean_text,
    detect_section,
    detect_topic,
)


def test_clean_text_normalizes_whitespace():
    assert clean_text("  alpha\n\n beta\t gamma  ") == "alpha beta gamma"


def test_chunk_text_keeps_overlap_and_metadata():
    text = " ".join(f"word{i}" for i in range(20))
    chunks = chunk_text(text, chunk_size=8, overlap=2, metadata={"title": "Guide"})

    assert len(chunks) > 1
    assert chunks[0].metadata["title"] == "Guide"
    assert chunks[0].text.split()[-2:] == chunks[1].text.split()[:2]


def test_chunk_text_rejects_invalid_overlap():
    try:
        chunk_text("some text", chunk_size=4, overlap=4)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("Expected invalid overlap to raise ValueError")


def test_detect_section_identifies_headings_and_topics():
    assert detect_section("PREPARATION OF NEEM SEED KERNEL EXTRACT (NSKE)") == "Botanical Preparations"
    assert detect_section("Chapter 4: Pest and Disease Management in Cotton") == "Pest and Disease Management"
    assert detect_section("Biological Control using Trichogramma") == "Biological Management"
    assert detect_section("Regular sentence with multiple words discussing crops in fields.") is None


def test_chunk_page_text_preserves_page_and_attaches_section_metadata():
    page_text = """
    CULTURAL PRACTICES FOR INSECT MANAGEMENT
    Intercropping with marigold repels bollworms in cotton crops.
    Field sanitation and removal of crop residue reduces pupation in soil.
    Mulching preserves soil moisture and promotes beneficial microbes.
    """
    base_meta = {
        "document_id": "test_doc",
        "document_title": "Test Guide",
        "source_file": "test.pdf",
        "source_type": "government_field_guide",
        "farming_approach": "natural_farming",
        "region": "India",
        "topic": "cultural_practices",
    }
    chunks = chunk_page_text(
        page_text=page_text,
        page_number=12,
        base_metadata=base_meta,
        chunk_size=50,
        overlap=10,
    )
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.metadata["page"] == 12
        assert chunk.metadata["document_id"] == "test_doc"
        assert chunk.metadata["document_title"] == "Test Guide"
        assert chunk.metadata["source_file"] == "test.pdf"
        assert chunk.metadata["section"] == "Cultural Practices"
        assert chunk.metadata["farming_approach"] == "natural_farming"
        assert chunk.metadata["region"] == "India"
        assert chunk.metadata["topic"] == "cultural_practices"
        assert len(chunk.text) > 0
