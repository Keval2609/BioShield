from src.chunking import chunk_text, clean_text


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
