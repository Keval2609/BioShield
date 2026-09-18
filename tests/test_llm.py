"""Tests for IBM Granite LLM client and error handling."""

from unittest.mock import MagicMock, patch
import pytest
import requests
from src.llm import GraniteLLM


def test_granite_llm_requires_base_url():
    client = GraniteLLM(base_url="")
    with pytest.raises(ValueError, match="GRANITE_BASE_URL is not set"):
        client.generate([{"role": "user", "content": "hello"}])


def test_granite_llm_handles_timeout():
    client = GraniteLLM(base_url="http://mock-granite:8000/v1")
    with patch("requests.post", side_effect=requests.exceptions.Timeout("Request timed out")):
        with pytest.raises(RuntimeError, match="timed out"):
            client.generate([{"role": "user", "content": "hello"}])


def test_granite_llm_handles_http_error():
    client = GraniteLLM(base_url="http://mock-granite:8000/v1")
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"
    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(RuntimeError, match="HTTP 500"):
            client.generate([{"role": "user", "content": "hello"}])


def test_granite_llm_handles_empty_choices():
    client = GraniteLLM(base_url="http://mock-granite:8000/v1")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": []}
    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(RuntimeError, match="No completion choices"):
            client.generate([{"role": "user", "content": "hello"}])


def test_granite_llm_successful_generation():
    client = GraniteLLM(base_url="http://mock-granite:8000/v1", api_key="secret-key")
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
        assert headers_sent["Authorization"] == "Bearer secret-key"
