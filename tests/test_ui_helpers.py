"""Tests for UI helper functions and human-readable error formatting."""

import json
from unittest.mock import MagicMock, patch

from src.ui_helpers import (
    EXAMPLE_QUERIES,
    check_kb_status,
    format_error_message,
)


def test_example_queries_contains_three_distinct_presets():
    """Verify the 3 required demo mode example queries exist with required fields."""
    assert len(EXAMPLE_QUERIES) == 3

    keys = list(EXAMPLE_QUERIES.keys())
    for key in keys:
        preset = EXAMPLE_QUERIES[key]
        assert "crop" in preset
        assert "problem" in preset
        assert "approach" in preset
        assert "question" in preset
        assert preset["approach"] in ["Natural farming", "Organic farming", "Sustainable/IPM"]


def test_format_error_message_handles_connection_refused():
    """Verify connection error is mapped to Granite configuration advice."""
    exc = ConnectionError("Failed to establish a new connection: [WinError 10061] No connection could be made")
    msg = format_error_message(exc)
    assert "Unable to connect to the IBM Granite inference service" in msg
    assert "GRANITE_BASE_URL" in msg
    assert "Traceback" not in msg


def test_format_error_message_handles_timeout():
    """Verify timeout is mapped to server delay guidance."""
    exc = TimeoutError("Request timed out after 45 seconds")
    msg = format_error_message(exc)
    assert "timed out" in msg.lower()
    assert "Traceback" not in msg


def test_format_error_message_handles_json_decode_error():
    """Verify malformed model response is mapped to friendly advice."""
    exc = json.JSONDecodeError("Expecting value", "invalid", 0)
    msg = format_error_message(exc)
    assert "structured advisory format" in msg.lower() or "malformed" in msg.lower()
    assert "Traceback" not in msg


def test_format_error_message_handles_chroma_or_empty_kb():
    """Verify missing or empty knowledge base returns helpful ingestion command."""
    exc = FileNotFoundError("Knowledge base directory does not exist or collection is empty")
    msg = format_error_message(exc)
    assert "python ingest.py" in msg
    assert "Traceback" not in msg


def test_check_kb_status_returns_counts():
    """Verify check_kb_status safely queries collection count."""
    with patch("src.retriever.ChromaRetriever") as mock_retriever_cls:
        mock_instance = MagicMock()
        mock_coll = MagicMock()
        mock_coll.count.return_value = 471
        mock_instance._get_collection.return_value = mock_coll
        mock_retriever_cls.return_value = mock_instance

        is_avail, count, msg = check_kb_status()
        assert is_avail is True
        assert count == 471
        assert "471 verified chunks" in msg
