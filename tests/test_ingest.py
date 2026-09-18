"""Tests for ChromaDB ingestion and idempotency."""

from pathlib import Path
import pytest
from src.ingest import get_chroma_client, ingest_documents


def test_ingest_is_idempotent_and_preserves_metadata(tmp_path):
    chroma_dir = tmp_path / "chroma_test"
    test_files = ["GujaratNaturalFarmingScienceUniversityGujarat.pdf"]

    sample_pdf = Path("resources") / test_files[0]
    if not sample_pdf.exists():
        pytest.skip(f"Test resource {sample_pdf} does not exist")

    # First ingestion run (limited to first 2 pages for fast test execution)
    added_first = ingest_documents(
        resources_dir=Path("resources"),
        persist_dir=chroma_dir,
        collection_name="test_advisory",
        target_files=test_files,
        chunk_size=500,
        overlap=50,
        max_pages=2,
    )
    assert added_first > 0

    # Second ingestion run - must be idempotent (0 new chunks)
    added_second = ingest_documents(
        resources_dir=Path("resources"),
        persist_dir=chroma_dir,
        collection_name="test_advisory",
        target_files=test_files,
        chunk_size=500,
        overlap=50,
        max_pages=2,
    )
    assert added_second == 0

    # Verify persisted metadata in ChromaDB using standardized client
    client = get_chroma_client(str(chroma_dir))
    coll = client.get_collection("test_advisory")
    assert coll.count() == added_first

    sample = coll.get(limit=1, include=["metadatas", "documents"])
    assert len(sample["documents"]) == 1
    meta = sample["metadatas"][0]

    required_keys = [
        "document_id",
        "document_title",
        "source_file",
        "page",
        "section",
        "source_type",
        "farming_approach",
        "region",
        "topic",
    ]
    for key in required_keys:
        assert key in meta, f"Missing required metadata key: {key}"
        assert meta[key] is not None

    assert meta["farming_approach"] == "natural_farming"
    assert meta["source_file"] == test_files[0]
    assert isinstance(meta["page"], int)
    assert meta["page"] >= 1
