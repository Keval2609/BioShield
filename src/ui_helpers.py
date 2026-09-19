"""UI helper utilities, human-readable error formatting, and demo presets for BioShield AI."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src import config

EXAMPLE_QUERIES: Dict[str, Dict[str, str]] = {
    "1. Sucking Pests (Natural Farming)": {
        "crop": "Various",
        "problem": "Sucking pests and leaf-eating caterpillars",
        "approach": "Natural farming",
        "question": "How to manage sucking pests and leaf-eating caterpillars using Neemastra?",
    },
    "2. Seed/Soil Diseases (Natural Farming)": {
        "crop": "Various",
        "problem": "Seed and soil-borne diseases",
        "approach": "Natural farming",
        "question": "How to treat seed and soil-borne diseases with Beejamrit?",
    },
    "3. Tomato: Chemical Schedule (Unsupported Safe Refusal)": {
        "crop": "Tomato",
        "problem": "Severe fungal blight and weed invasion.",
        "approach": "Sustainable/IPM",
        "question": "Give me the chemical pesticide dosage schedule and synthetic tank mix to eradicate all pests.",
    },
}


def format_error_message(exc: Exception) -> str:
    """Translate low-level runtime exceptions into friendly, actionable guidance.
    
    Prevents raw Python stack traces from being displayed to agricultural users.
    """
    error_text = str(exc).lower()
    exc_type = type(exc).__name__

    if isinstance(exc, (ConnectionError, ConnectionRefusedError)) or "connection" in error_text or "10061" in error_text:
        return (
            "Unable to connect to the local inference service. "
            "Please verify that your model server (such as Ollama or vLLM) is running and accessible, "
            f"or ensure `OLLAMA_BASE_URL` ({getattr(config, 'OLLAMA_BASE_URL', 'not set')}) is properly configured in your `.env` file."
        )

    if isinstance(exc, TimeoutError) or "timed out" in error_text:
        return (
            "The advisory request timed out while waiting for Ollama to respond. "
            "This can happen if the local LLM is compiling or system load is high. "
            "Please try submitting your inquiry again."
        )

    if isinstance(exc, json.JSONDecodeError) or "malformed" in error_text:
        return (
            "The model response could not be parsed into the required structured advisory format. "
            "BioShield AI enforces strict schema validation for agricultural safety. "
            "Please refine your query or resubmit."
        )

    if "api key" in error_text or "401" in error_text or "unauthorized" in error_text:
        return (
            "Authentication failed for the LLM service. "
            "Please verify that `GRANITE_API_KEY` is provided in `.env` if using a secured enterprise endpoint."
        )

    if (
        isinstance(exc, FileNotFoundError)
        or "chroma" in error_text
        or "collection" in error_text
        or "empty" in error_text
    ):
        return (
            "The agricultural knowledge base is currently unavailable or unindexed. "
            "Please run `python ingest.py` in your terminal to extract and embed the core reference documents into ChromaDB."
        )

    return (
        f"An unexpected error ({exc_type}) occurred while processing your advisory. "
        "Please review your input or check the terminal output for configuration details."
    )


def check_kb_status(persist_path: Optional[Path] = None) -> Tuple[bool, int, str]:
    """Check whether the ChromaDB knowledge base is accessible and count indexed chunks.
    
    Returns:
        Tuple of (is_available: bool, chunk_count: int, message: str)
    """
    try:
        from src.retriever import ChromaRetriever

        retriever = ChromaRetriever(persist_path=persist_path)
        collection = retriever._get_collection()
        if collection is not None:
            count = collection.count()
            if count > 0:
                return True, count, f"{count} verified chunks indexed in ChromaDB"
            return False, 0, "ChromaDB collection is empty (run python ingest.py)"
    except Exception as exc:
        return False, 0, f"ChromaDB not accessible: {exc}"

    return False, 0, "ChromaDB collection not found (run python ingest.py)"
