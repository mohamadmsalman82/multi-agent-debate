"""Tests for agents.llm_provider module."""

from __future__ import annotations

import pytest

from agents.llm_provider import LLMResponse, create_provider
from tests.conftest import MockProvider


class TestMockProvider:
    """Verify the mock provider works correctly for downstream tests."""

    @pytest.mark.asyncio
    async def test_generate_returns_llm_response(self, mock_provider: MockProvider):
        resp = await mock_provider.generate(
            [{"role": "user", "content": "Hello"}],
            temperature=0.5,
            max_tokens=100,
        )
        assert isinstance(resp, LLMResponse)
        assert resp.provider == "mock"
        assert resp.model == "mock-v1"
        assert len(resp.text) > 0
        assert resp.tokens_used > 0
        assert resp.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_multiple_responses_cycle(self):
        provider = MockProvider(responses=["first", "second", "third"])
        r1 = await provider.generate([{"role": "user", "content": "a"}])
        r2 = await provider.generate([{"role": "user", "content": "b"}])
        r3 = await provider.generate([{"role": "user", "content": "c"}])
        r4 = await provider.generate([{"role": "user", "content": "d"}])
        assert r1.text == "first"
        assert r2.text == "second"
        assert r3.text == "third"
        assert r4.text == "first"  # cycles

    @pytest.mark.asyncio
    async def test_call_log_records_parameters(self, mock_provider: MockProvider):
        await mock_provider.generate(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
            temperature=0.9,
            max_tokens=200,
        )
        assert len(mock_provider.call_log) == 1
        log = mock_provider.call_log[0]
        assert log["temperature"] == 0.9
        assert log["max_tokens"] == 200
        assert len(log["messages"]) == 2


class TestCreateProvider:
    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider("nonexistent", api_key="k")

    def test_missing_api_key_raises(self):
        with pytest.raises(ValueError, match="No API key"):
            create_provider("openai")  # no key, no env var

    def test_openrouter_provider_created(self):
        """Test that OpenRouter provider can be instantiated."""
        provider = create_provider("openrouter", api_key="test-key", model="openai/gpt-4")
        assert provider.name == "openrouter"
        assert provider.model == "openai/gpt-4"


class TestLLMResponse:
    def test_frozen_dataclass(self):
        resp = LLMResponse(
            text="hi", tokens_used=5, model="m", provider="p", latency_ms=10.0
        )
        assert resp.text == "hi"
        with pytest.raises(AttributeError):
            resp.text = "modified"  # type: ignore[misc]
