"""Tests for PDF extraction and error handling."""

from pathlib import Path
import pytest
from src.ingest import extract_pages_from_pdf


def test_extract_pages_handles_nonexistent_file():
    pages = extract_pages_from_pdf(Path("resources/nonexistent_file.pdf"))
    assert pages == []


def test_extract_pages_handles_non_pdf_file(tmp_path):
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("plain text", encoding="utf-8")
    pages = extract_pages_from_pdf(txt_file)
    assert pages == []


def test_extract_pages_handles_corrupted_file(tmp_path):
    corrupt_pdf = tmp_path / "corrupt.pdf"
    corrupt_pdf.write_bytes(b"%PDF-1.4\ncorrupted bytes that cannot be read")
    pages = extract_pages_from_pdf(corrupt_pdf)
    assert pages == []


def test_extract_pages_reads_real_pdf_and_returns_page_numbers():
    sample_pdf = Path("resources/GujaratNaturalFarmingScienceUniversityGujarat.pdf")
    if not sample_pdf.exists():
        pytest.skip("Sample PDF not found in resources")
    pages = extract_pages_from_pdf(sample_pdf)
    assert len(pages) > 0
    first_page_num, first_page_text = pages[0]
    assert first_page_num == 1
    assert isinstance(first_page_text, str)
    assert len(first_page_text.strip()) > 0
