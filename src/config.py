"""Configuration loader for BioShield AI."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

def get_config_val(key: str, default: str = "") -> str:
    """Safely retrieve configuration from env vars or Streamlit secrets."""
    val = os.getenv(key)
    if val is not None:
        return val
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return default

# LLM Configuration
LLM_PROVIDER = get_config_val("LLM_PROVIDER", "ollama").lower()
LLM_TEMPERATURE = float(get_config_val("LLM_TEMPERATURE", "0.1"))

# Local Ollama Provider
OLLAMA_BASE_URL = get_config_val("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = get_config_val("OLLAMA_MODEL", "granite3.3:2b")

# Cloud Groq Provider
GROQ_API_KEY = get_config_val("GROQ_API_KEY", "")
GROQ_MODEL = get_config_val("GROQ_MODEL", "")

# Embeddings & Vector DB Configuration
EMBEDDING_MODEL = get_config_val("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(get_config_val("TOP_K", "4"))
SIMILARITY_THRESHOLD = float(get_config_val("SIMILARITY_THRESHOLD", "0.35"))

# Storage Directories
CHROMA_PERSIST_DIR = Path(get_config_val("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "chroma_db")))
DATA_RAW_DIR = Path(get_config_val("DATA_RAW_DIR", str(PROJECT_ROOT / "data" / "raw")))
DATA_PROCESSED_DIR = Path(get_config_val("DATA_PROCESSED_DIR", str(PROJECT_ROOT / "data" / "processed")))
CORE_RESOURCES_DIR = Path(get_config_val("CORE_RESOURCES_DIR", str(PROJECT_ROOT / "resources")))
DATA_SECONDARY_DIR = Path(get_config_val("DATA_SECONDARY_DIR", str(PROJECT_ROOT / "data" / "secondary")))

# Collection Name in ChromaDB
CHROMA_COLLECTION_NAME = "agricultural_advisory"
