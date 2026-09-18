"""Configuration loader for BioShield AI."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# LLM Configuration
GRANITE_MODEL = os.getenv("GRANITE_MODEL", "ibm/granite-3-8b-instruct")
GRANITE_BASE_URL = os.getenv("GRANITE_BASE_URL", "")
GRANITE_API_KEY = os.getenv("GRANITE_API_KEY", "")

# Embeddings & Vector DB Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "4"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))

# Storage Directories
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "chroma_db")))
DATA_RAW_DIR = Path(os.getenv("DATA_RAW_DIR", str(PROJECT_ROOT / "data" / "raw")))
DATA_PROCESSED_DIR = Path(os.getenv("DATA_PROCESSED_DIR", str(PROJECT_ROOT / "data" / "processed")))

# Collection Name in ChromaDB
CHROMA_COLLECTION_NAME = "agricultural_advisory"
