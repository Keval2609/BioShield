"""Tests for document catalog and metadata schemas."""

from src.document_catalog import (
    CORE_DOCUMENTS,
    SECONDARY_DOCUMENTS,
    get_document_metadata,
    is_core_document,
)


def test_core_documents_contains_only_five_core_files():
    assert len(CORE_DOCUMENTS) == 5
    expected_files = {
        "Field_Guide_for_Natural_Farming.pdf",
        "Generic Protocols for Natural Farming.pdf",
        "GujaratNaturalFarmingScienceUniversityGujarat.pdf",
        "PEST and DISEASE MANAGEMENT.pdf",
        "Natural Farming Training Toolkit.pdf",
    }
    assert set(CORE_DOCUMENTS.keys()) == expected_files


def test_secondary_documents_are_excluded_from_core():
    excluded_files = [
        "Comprehensive_Training_manual_Organic_Farming.pdf",
        "ICAR-En-Kharif-Agro-Advisories-for-Farmers-2025.pdf",
        "Insect-PestManagementunderNaturalFarming.pdf",
    ]
    for filename in excluded_files:
        assert not is_core_document(filename)
        assert filename in SECONDARY_DOCUMENTS


def test_metadata_has_all_required_keys():
    required_keys = {
        "document_id",
        "document_title",
        "source_file",
        "source_type",
        "farming_approach",
        "region",
        "topic",
    }
    for filename, meta in CORE_DOCUMENTS.items():
        assert required_keys.issubset(meta.keys())
        assert meta["farming_approach"] == "natural_farming"
        assert meta["source_file"] == filename

    # Verify ICAR advisory is categorized as IPM/conventional, not natural farming
    icar_meta = get_document_metadata("ICAR-En-Kharif-Agro-Advisories-for-Farmers-2025.pdf")
    assert icar_meta["farming_approach"] in ["IPM", "conventional/IPM"]
