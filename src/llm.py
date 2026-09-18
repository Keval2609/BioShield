"""LLM client for IBM Granite inference with isolated provider interface."""

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


class GraniteLLM:
    """Client for IBM Granite model inference via OpenAI-compatible REST endpoints.
    
    Supports local deployments (e.g. Ollama, vLLM) and remote endpoints (e.g. watsonx.ai).
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 45,
    ):
        self.model = model or config.GRANITE_MODEL
        self.base_url = (base_url or config.GRANITE_BASE_URL).rstrip("/")
        self.api_key = api_key or config.GRANITE_API_KEY
        self.timeout = timeout

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        """Call Granite model with chat messages list at temperature 0 (deterministic).
        
        Args:
            messages: List of message dictionaries, e.g. [{"role": "system", ...}, {"role": "user", ...}]
            temperature: Sampling temperature, defaults to 0.0 for strict grounding.

        Returns:
            The generated response string.

        Raises:
            ValueError: If endpoint is not configured.
            RuntimeError: If API call fails, times out, or returns non-200.
        """
        if not self.base_url:
            raise ValueError(
                "GRANITE_BASE_URL is not set. Please configure GRANITE_BASE_URL in your .env file "
                "(e.g., http://localhost:11434/v1 for local Ollama or your cloud endpoint)."
            )

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        endpoint = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=self.timeout)
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"IBM Granite endpoint timed out after {self.timeout}s at {endpoint}."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                f"Failed to reach IBM Granite endpoint at {endpoint}. Error: {exc}"
            ) from exc

        if response.status_code != 200:
            raise RuntimeError(
                f"IBM Granite API returned HTTP {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError("No completion choices returned by IBM Granite endpoint.")
            content = choices[0].get("message", {}).get("content", "")
            return content.strip()
        except Exception as exc:
            raise RuntimeError(f"Failed to parse IBM Granite response: {exc}") from exc
