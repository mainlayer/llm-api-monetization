"""
Upstream LLM client.

Supports any OpenAI-compatible API endpoint, including:
  - OpenAI (api.openai.com)
  - Anthropic (via the OpenAI-compatible messages adapter)
  - Groq, Together AI, Perplexity, Fireworks, OpenRouter, etc.
  - Local servers: Ollama, LM Studio, vLLM

Configure via environment variables (see .env.example):
  LLM_BASE_URL   — upstream API base URL
  LLM_API_KEY    — upstream API key
  LLM_MODEL      — default model to use if none specified in the request
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import httpx

from src.models import ChatCompletionResponse, ChatRequest, ChatResponseMessage
from src.models import ChatChoice, UsageInfo

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LLM_BASE_URL: str = os.getenv(
    "LLM_BASE_URL", "https://api.openai.com"
).rstrip("/")

LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")

LLM_DEFAULT_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

# Timeouts
_CONNECT_TIMEOUT: float = 10.0
_READ_TIMEOUT: float = 120.0  # LLMs can be slow for long completions


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LLMClientError(Exception):
    """Base exception for upstream LLM errors."""


class LLMAuthError(LLMClientError):
    """Raised when the upstream LLM rejects the API key."""


class LLMRateLimitError(LLMClientError):
    """Raised when the upstream LLM rate-limits the request."""


class LLMNetworkError(LLMClientError):
    """Raised when the upstream LLM is unreachable."""


class LLMResponseError(LLMClientError):
    """Raised when the upstream LLM returns an unexpected response."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class LLMClient:
    """
    Async client that forwards ChatRequest objects to an OpenAI-compatible API
    and returns a ChatCompletionResponse.

    Example::

        client = LLMClient()
        response = await client.complete(request)
        print(response.choices[0].message.content)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
    ) -> None:
        self._base_url = (base_url or LLM_BASE_URL).rstrip("/")
        self._api_key = api_key or LLM_API_KEY
        self._default_model = default_model or LLM_DEFAULT_MODEL
        self._timeout = httpx.Timeout(
            connect=_CONNECT_TIMEOUT,
            read=_READ_TIMEOUT,
            write=30.0,
            pool=10.0,
        )

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def complete(self, request: ChatRequest) -> ChatCompletionResponse:
        """
        Forward a ChatRequest to the upstream LLM and return the response.

        Args:
            request: The incoming ChatRequest (OpenAI-compatible schema).

        Returns:
            ChatCompletionResponse parsed from the upstream JSON.

        Raises:
            LLMAuthError:       Upstream rejected the API key (HTTP 401).
            LLMRateLimitError:  Upstream rate-limited the request (HTTP 429).
            LLMNetworkError:    Cannot reach the upstream endpoint.
            LLMResponseError:   Unexpected HTTP status or malformed JSON.
        """
        payload = self._build_payload(request)
        url = f"{self._base_url}/v1/chat/completions"

        logger.debug(
            "Forwarding request to %s model=%s messages=%d",
            url,
            payload.get("model"),
            len(request.messages),
        )

        try:
            async with httpx.AsyncClient(
                headers=self._headers, timeout=self._timeout
            ) as client:
                upstream_response = await client.post(url, json=payload)
        except httpx.ConnectError as exc:
            raise LLMNetworkError(
                f"Cannot connect to LLM at {self._base_url}: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMNetworkError(
                f"LLM request timed out after {_READ_TIMEOUT}s: {exc}"
            ) from exc

        return self._parse_response(upstream_response)

    async def check_health(self) -> bool:
        """
        Lightweight health check — fetches the model list from the upstream.

        Returns:
            True if the upstream is reachable and responding.
        """
        url = f"{self._base_url}/v1/models"
        try:
            async with httpx.AsyncClient(
                headers=self._headers,
                timeout=httpx.Timeout(5.0),
            ) as client:
                resp = await client.get(url)
            return resp.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_payload(self, request: ChatRequest) -> Dict[str, Any]:
        """Serialize the ChatRequest to a dict suitable for the upstream API."""
        payload: Dict[str, Any] = {
            "model": request.model or self._default_model,
            "messages": [
                self._serialize_message(msg) for msg in request.messages
            ],
        }

        # Optional fields — only include if explicitly set
        optionals = [
            "temperature", "top_p", "n", "stream", "stop",
            "max_tokens", "presence_penalty", "frequency_penalty",
            "logit_bias", "user", "tools", "tool_choice",
            "response_format", "seed",
        ]
        for field in optionals:
            value = getattr(request, field, None)
            if value is not None:
                payload[field] = (
                    value.model_dump() if hasattr(value, "model_dump") else value
                )

        # Streaming is not supported by this proxy (responses are buffered)
        payload["stream"] = False

        return payload

    @staticmethod
    def _serialize_message(msg) -> Dict[str, Any]:
        """Convert a ChatMessage to a plain dict for the upstream API."""
        data: Dict[str, Any] = {"role": msg.role}
        if msg.content is not None:
            data["content"] = msg.content
        if msg.name is not None:
            data["name"] = msg.name
        if msg.tool_calls is not None:
            data["tool_calls"] = msg.tool_calls
        if msg.tool_call_id is not None:
            data["tool_call_id"] = msg.tool_call_id
        return data

    def _parse_response(
        self, response: httpx.Response
    ) -> ChatCompletionResponse:
        """Parse the upstream HTTP response into a ChatCompletionResponse."""
        if response.status_code == 401:
            raise LLMAuthError(
                "Upstream LLM rejected the API key. Check LLM_API_KEY."
            )
        if response.status_code == 429:
            raise LLMRateLimitError(
                "Upstream LLM rate limit exceeded. Retry after a moment."
            )
        if response.status_code not in (200, 201):
            raise LLMResponseError(
                f"Upstream LLM returned HTTP {response.status_code}: "
                f"{response.text[:400]}"
            )

        try:
            data = response.json()
        except Exception as exc:
            raise LLMResponseError(
                f"Failed to parse upstream LLM response as JSON: {exc}"
            ) from exc

        # Parse usage
        usage_data = data.get("usage", {})
        usage = UsageInfo(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        # Parse choices
        choices = []
        for choice_data in data.get("choices", []):
            msg_data = choice_data.get("message", {})
            message = ChatResponseMessage(
                role=msg_data.get("role", "assistant"),
                content=msg_data.get("content"),
                tool_calls=msg_data.get("tool_calls"),
                function_call=msg_data.get("function_call"),
            )
            choices.append(
                ChatChoice(
                    index=choice_data.get("index", 0),
                    message=message,
                    finish_reason=choice_data.get("finish_reason"),
                    logprobs=choice_data.get("logprobs"),
                )
            )

        return ChatCompletionResponse(
            id=data.get("id", ""),
            object=data.get("object", "chat.completion"),
            created=data.get("created", 0),
            model=data.get("model", self._default_model),
            choices=choices,
            usage=usage,
            system_fingerprint=data.get("system_fingerprint"),
        )
