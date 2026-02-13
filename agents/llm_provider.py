"""LLM Provider abstraction layer for OpenAI, Anthropic, Cohere, and OpenRouter.

Provides a unified async interface to multiple LLM backends with
built-in retry logic, rate-limit handling, and token tracking.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMResponse:
    """Standardised response from any LLM provider."""

    text: str
    tokens_used: int
    model: str
    provider: str
    latency_ms: float
    raw: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """Provider-agnostic interface that all LLM backends implement."""

    name: str  # e.g. "openai", "anthropic", "cohere"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

        # Resolve API key: explicit > env var > raise
        self.api_key = api_key or os.getenv(api_key_env or "")
        if not self.api_key:
            raise ValueError(
                f"No API key for {self.name}. "
                f"Set {api_key_env!r} or pass api_key explicitly."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 500,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response with automatic retries on transient errors."""
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                start = time.perf_counter()
                response = await self._call_api(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                elapsed = (time.perf_counter() - start) * 1000
                response = LLMResponse(
                    text=response["text"],
                    tokens_used=response.get("tokens_used", 0),
                    model=self.model,
                    provider=self.name,
                    latency_ms=round(elapsed, 1),
                    raw=response.get("raw", {}),
                )
                logger.debug(
                    "[%s] %s responded (%d tokens, %.0f ms)",
                    self.name,
                    self.model,
                    response.tokens_used,
                    response.latency_ms,
                )
                return response
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                wait = min(2**attempt, 16)
                logger.warning(
                    "[%s] Attempt %d/%d failed (%s). Retrying in %ds …",
                    self.name,
                    attempt,
                    self.max_retries,
                    exc,
                    wait,
                )
                await asyncio.sleep(wait)

        raise RuntimeError(
            f"[{self.name}] All {self.max_retries} attempts failed"
        ) from last_exc

    # ------------------------------------------------------------------
    # Backend-specific implementation (override in subclasses)
    # ------------------------------------------------------------------

    @abstractmethod
    async def _call_api(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return ``{"text": ..., "tokens_used": ..., "raw": ...}``."""
        ...

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    """Async OpenAI provider using the ``openai>=1.0`` client."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o", **kwargs: Any) -> None:
        kwargs.setdefault("api_key_env", "OPENAI_API_KEY")
        super().__init__(model=model, **kwargs)
        import openai
        self._client = openai.AsyncOpenAI(
            api_key=self.api_key,
            timeout=self.timeout,
        )

    async def _call_api(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        choice = response.choices[0]
        usage = response.usage
        return {
            "text": choice.message.content or "",
            "tokens_used": usage.total_tokens if usage else 0,
            "raw": response.model_dump(),
        }


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    """Async Anthropic provider using the ``anthropic>=0.18`` client."""

    name = "anthropic"

    def __init__(
        self, model: str = "claude-sonnet-4-5", **kwargs: Any
    ) -> None:
        kwargs.setdefault("api_key_env", "ANTHROPIC_API_KEY")
        super().__init__(model=model, **kwargs)
        import anthropic
        self._client = anthropic.AsyncAnthropic(
            api_key=self.api_key,
            timeout=self.timeout,
        )

    async def _call_api(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Anthropic uses a separate system parameter
        system_msg = ""
        api_messages: list[dict[str, str]] = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                api_messages.append(msg)

        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }
        if system_msg:
            create_kwargs["system"] = system_msg

        response = await self._client.messages.create(**create_kwargs)
        text_block = response.content[0].text if response.content else ""
        tokens = (response.usage.input_tokens + response.usage.output_tokens) if response.usage else 0
        return {
            "text": text_block,
            "tokens_used": tokens,
            "raw": response.model_dump() if hasattr(response, "model_dump") else {},
        }


# ---------------------------------------------------------------------------
# Cohere
# ---------------------------------------------------------------------------

class CohereProvider(LLMProvider):
    """Async Cohere provider using the ``cohere>=5.0`` client."""

    name = "cohere"

    def __init__(self, model: str = "command-r-plus", **kwargs: Any) -> None:
        kwargs.setdefault("api_key_env", "COHERE_API_KEY")
        super().__init__(model=model, **kwargs)
        import cohere
        self._client = cohere.AsyncClientV2(api_key=self.api_key)

    async def _call_api(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        response = await self._client.chat(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        text = response.message.content[0].text if response.message and response.message.content else ""
        tokens = 0
        if response.usage and response.usage.tokens:
            tokens = (
                (response.usage.tokens.input_tokens or 0)
                + (response.usage.tokens.output_tokens or 0)
            )
        return {
            "text": text,
            "tokens_used": tokens,
            "raw": response.model_dump() if hasattr(response, "model_dump") else {},
        }


# ---------------------------------------------------------------------------
# OpenRouter
# ---------------------------------------------------------------------------

class OpenRouterProvider(LLMProvider):
    """Async OpenRouter provider using OpenAI-compatible API.

    OpenRouter provides unified access to multiple LLM providers
    (OpenAI, Anthropic, Cohere, Meta, Google, etc.) through a single
    API key and OpenAI-compatible interface.

    Model names should use OpenRouter's format, e.g.:
    - "openai/gpt-4o"
    - "anthropic/claude-sonnet-4.5"
    - "cohere/command-r-plus"
    - "meta-llama/llama-3.1-70b-instruct"
    """

    name = "openrouter"

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("api_key_env", "OPENROUTER_API_KEY")
        super().__init__(model=model, **kwargs)

        import openai
        self._client = openai.AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            timeout=self.timeout,
        )

    async def _call_api(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        choice = response.choices[0]
        usage = response.usage
        return {
            "text": choice.message.content or "",
            "tokens_used": usage.total_tokens if usage else 0,
            "raw": response.model_dump(),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROVIDERS: dict[str, type[LLMProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "cohere": CohereProvider,
    "openrouter": OpenRouterProvider,
}


def create_provider(name: str, **kwargs: Any) -> LLMProvider:
    """Instantiate an LLM provider by its short name.

    >>> provider = create_provider("openai", model="gpt-4o")
    """
    cls = _PROVIDERS.get(name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown provider {name!r}. Choose from {list(_PROVIDERS)}"
        )
    return cls(**kwargs)
