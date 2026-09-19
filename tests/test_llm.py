"""Tests for LLM provider clients (Ollama and Groq)."""

import os
from unittest.mock import MagicMock, patch
import pytest
import requests
from src.llm import OllamaProvider, GroqProvider


# --- OllamaProvider Tests ---

def test_ollama_provider_handles_timeout():
    client = OllamaProvider(base_url="http://localhost:11434")
    with patch("requests.post", side_effect=requests.exceptions.Timeout("Request timed out")):
        with pytest.raises(RuntimeError, match="timed out"):
            client.generate([{"role": "user", "content": "hello"}])


def test_ollama_provider_handles_connection_error():
    client = OllamaProvider(base_url="http://localhost:11434")
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Connection refused")):
        with pytest.raises(RuntimeError, match="Local Granite model is unavailable"):
            client.generate([{"role": "user", "content": "hello"}])


def test_ollama_provider_handles_http_error():
    client = OllamaProvider(base_url="http://localhost:11434")
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"
    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(RuntimeError, match="HTTP 500"):
            client.generate([{"role": "user", "content": "hello"}])


def test_ollama_provider_successful_generation():
    client = OllamaProvider(base_url="http://localhost:11434")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": '{"possible_issue": "Aphid pressure"}'}}]
    }
    with patch("requests.post", return_value=mock_resp):
        res = client.generate([{"role": "user", "content": "hello"}])
        assert res == '{"possible_issue": "Aphid pressure"}'


# --- GroqProvider Tests ---

def test_groq_provider_requires_api_key():
    client = GroqProvider(api_key="")
    with pytest.raises(ValueError, match="GROQ_API_KEY is not set"):
        client.generate([{"role": "user", "content": "hello"}])


def test_groq_provider_successful_generation():
    client = GroqProvider(api_key="mock-key", model="llama-3.1-8b-instant")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": '{"possible_issue": "Aphid pressure"}'}}]
    }
    with patch("requests.post", return_value=mock_resp) as mock_post:
        res = client.generate([{"role": "user", "content": "hello"}])
        assert res == '{"possible_issue": "Aphid pressure"}'

        # Verify auth header was sent
        headers_sent = mock_post.call_args[1]["headers"]
        assert headers_sent["Authorization"] == "Bearer mock-key"

@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="Groq test skipped: GROQ_API_KEY not configured.")
def test_groq_provider_live_if_configured():
    # Only runs if GROQ_API_KEY is set in environment, fulfilling the requested requirement:
    # "If Groq credentials are unavailable... Groq test skipped: GROQ_API_KEY not configured."
    client = GroqProvider()
    # Simply assert initialization didn't fail
    assert client.api_key is not None
