"""Document catalog and metadata registry for BioShield AI.

Defines the 5 core natural farming documents and explicitly handles
secondary / excluded documents.
"""

from typing import Any, Dict, Optional

# The 5 core documents representing the primary Natural Farming corpus
CORE_DOCUMENTS: Dict[str, Dict[str, Any]] = {
    "Field_Guide_for_Natural_Farming.pdf": {
        "document_id": "field_guide_natural_farming",
        "document_title": "Field Guide for Natural Farming",
        "source_file": "Field_Guide_for_Natural_Farming.pdf",
        "source_type": "government_field_guide",
        "farming_approach": "natural_farming",
        "region": "India",
        "topic": "natural_farming_practices",
    },
    "Generic Protocols for Natural Farming.pdf": {
        "document_id": "generic_protocols_natural_farming",
        "document_title": "Generic Protocols for Natural Farming",
        "source_file": "Generic Protocols for Natural Farming.pdf",
        "source_type": "natural_farming_protocol",
        "farming_approach": "natural_farming",
        "region": "India",
        "topic": "natural_farming_protocols",
    },
    "GujaratNaturalFarmingScienceUniversityGujarat.pdf": {
        "document_id": "gujarat_natural_farming_science_univ",
        "document_title": "Gujarat Natural Farming Science University Manual",
        "source_file": "GujaratNaturalFarmingScienceUniversityGujarat.pdf",
        "source_type": "agricultural_university_manual",
        "farming_approach": "natural_farming",
        "region": "Gujarat",
        "topic": "natural_farming_science_and_curriculum",
    },
    "PEST and DISEASE MANAGEMENT.pdf": {
        "document_id": "pest_and_disease_management",
        "document_title": "Pest and Disease Management",
        "source_file": "PEST and DISEASE MANAGEMENT.pdf",
        "source_type": "pest_management_manual",
        "farming_approach": "natural_farming",
        "region": "India",
        "topic": "pest_and_disease_management",
    },
    "Natural Farming Training Toolkit.pdf": {
        "document_id": "natural_farming_training_toolkit",
        "document_title": "Natural Farming Training Toolkit",
        "source_file": "Natural Farming Training Toolkit.pdf",
        "source_type": "government_training_manual",
        "farming_approach": "natural_farming",
        "region": "India",
        "topic": "training_toolkit",
    },
}

# Secondary documents that must NOT automatically be ingested into the core corpus
SECONDARY_DOCUMENTS: Dict[str, Dict[str, Any]] = {
    "Comprehensive_Training_manual_Organic_Farming.pdf": {
        "document_id": "organic_farming_training_manual",
        "document_title": "Comprehensive Training Manual on Organic Farming",
        "source_file": "Comprehensive_Training_manual_Organic_Farming.pdf",
        "source_type": "organic_training_manual",
        "farming_approach": "conventional/IPM",
        "region": "India",
        "topic": "organic_and_ipm_practices",
    },
    "ICAR-En-Kharif-Agro-Advisories-for-Farmers-2025.pdf": {
        "document_id": "icar_kharif_agro_advisories_2025",
        "document_title": "ICAR Kharif Agro-Advisories for Farmers 2025",
        "source_file": "ICAR-En-Kharif-Agro-Advisories-for-Farmers-2025.pdf",
        "source_type": "icar_advisory",
        "farming_approach": "conventional/IPM",
        "region": "India",
        "topic": "crop_advisories_and_ipm",
    },
    "Insect-PestManagementunderNaturalFarming.pdf": {
        "document_id": "insect_pest_management_natural_farming",
        "document_title": "Insect-Pest Management under Natural Farming",
        "source_file": "Insect-PestManagementunderNaturalFarming.pdf",
        "source_type": "research_paper",
        "farming_approach": "natural_farming",
        "region": "India",
        "topic": "pest_management_research",
    },
}


def is_core_document(filename: str) -> bool:
    """Check if a file belongs to the core natural farming corpus."""
    return filename in CORE_DOCUMENTS


def get_document_metadata(filename: str) -> Dict[str, Any]:
    """Retrieve default metadata for a recognized core or secondary document.
    
    If the document is not pre-registered, generates standardized fallback metadata.
    """
    if filename in CORE_DOCUMENTS:
        return dict(CORE_DOCUMENTS[filename])
    if filename in SECONDARY_DOCUMENTS:
        return dict(SECONDARY_DOCUMENTS[filename])

    # Fallback for arbitrary external files
    clean_stem = filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ")
    return {
        "document_id": filename.rsplit(".", 1)[0].lower().replace(" ", "_"),
        "document_title": clean_stem.title(),
        "source_file": filename,
        "source_type": "external_document",
        "farming_approach": "conventional/IPM",
        "region": "India",
        "topic": "general_agriculture",
    }
