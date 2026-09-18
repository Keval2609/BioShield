"""Tests for configuration settings and paths."""

from pathlib import Path
from src import config


def test_config_paths_and_defaults():
    assert config.EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.TOP_K >= 1
    assert 0.0 < config.SIMILARITY_THRESHOLD < 1.0
    assert config.CORE_RESOURCES_DIR.exists()
    assert config.DATA_SECONDARY_DIR.exists()
    assert config.CHROMA_COLLECTION_NAME == "agricultural_advisory"
