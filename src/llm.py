"""LLM client layer supporting local (Ollama) and cloud (Groq) providers."""

import logging
from typing import Any, Dict, List, Optional, Protocol
import requests

from src import config

logger = logging.getLogger("bioshield.llm")


class BaseLLM(Protocol):
    """Protocol defining the inference interface for BioShield AI language models."""

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        """Generate a text response given a list of chat messages."""
        ...


class FakeLLM:
    """Mock LLM implementation for tests and offline development."""

    def __init__(self, response: str = "{}"):
        self.response = response
        self.last_messages: List[Dict[str, str]] = []

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        self.last_messages = messages
        return self.response


class OllamaProvider:
    """Client for local Ollama inference via OpenAI-compatible REST endpoints."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 45,
    ):
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or config.OLLAMA_MODEL
        self.timeout = timeout

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        """Call Ollama model with chat messages list at given temperature."""
        endpoint = f"{self.base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=self.timeout)
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"Ollama endpoint timed out after {self.timeout}s at {endpoint}."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                f"Local Granite model is unavailable. Please start Ollama and ensure {self.model} is installed. Error: {exc}"
            ) from exc

        if response.status_code != 200:
            raise RuntimeError(f"Ollama API returned HTTP {response.status_code}: {response.text}")

        try:
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError("No completion choices returned by Ollama endpoint.")
            content = choices[0].get("message", {}).get("content", "")
            return content.strip()
        except Exception as exc:
            raise RuntimeError(f"Failed to parse Ollama response: {exc}") from exc


class GroqProvider:
    """Client for Groq Cloud API inference via OpenAI-compatible REST endpoints."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 45,
    ):
        self.api_key = api_key or config.GROQ_API_KEY
        self.model = model or config.GROQ_MODEL
        self.timeout = timeout

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        """Call Groq API with chat messages list at given temperature."""
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Please configure GROQ_API_KEY in your .env file or Streamlit secrets."
            )

        endpoint = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=self.timeout)
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"Groq endpoint timed out after {self.timeout}s at {endpoint}."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                f"Failed to reach Groq API endpoint at {endpoint}. Error: {exc}"
            ) from exc

        if response.status_code != 200:
            raise RuntimeError(f"Groq API returned HTTP {response.status_code}: {response.text}")

        try:
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError("No completion choices returned by Groq endpoint.")
            content = choices[0].get("message", {}).get("content", "")
            return content.strip()
        except Exception as exc:
            raise RuntimeError(f"Failed to parse Groq response: {exc}") from exc
