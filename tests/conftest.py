"""Shared fixtures for the test suite.

Provides a MockProvider that simulates LLM responses without network calls,
plus pre-built agents and database instances for integration tests.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest
import pytest_asyncio

from agents.base import AgentResponse, AgentRole, BaseAgent, DebateState
from agents.llm_provider import LLMProvider
from data.database import DebateDatabase


# ---------------------------------------------------------------------------
# Mock LLM provider
# ---------------------------------------------------------------------------

class MockProvider(LLMProvider):
    """Deterministic mock provider for testing – no network calls."""

    name = "mock"

    def __init__(
        self,
        model: str = "mock-v1",
        responses: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        # Bypass API-key validation
        self.model = model
        self.timeout = kwargs.get("timeout", 30)
        self.max_retries = kwargs.get("max_retries", 1)
        self.api_key = "mock-key"

        self._responses = responses or [
            "This is a mock response about the debate topic. "
            "It contains evidence: studies show 75% effectiveness. "
            "For example, recent research demonstrates clear results."
        ]
        self._call_count = 0
        self.call_log: list[dict[str, Any]] = []

    async def _call_api(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        text = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        self.call_log.append(
            {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        )
        return {"text": text, "tokens_used": len(text.split()) * 2, "raw": {}}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_provider() -> MockProvider:
    return MockProvider()


@pytest.fixture
def varied_provider() -> MockProvider:
    """Provider that returns different responses on successive calls."""
    return MockProvider(
        responses=[
            "I strongly argue that regulation is necessary for AI safety. "
            "Studies show 80% of experts agree. This evidence demonstrates the need.",
            "I disagree with this position. The argument contains a logical fallacy. "
            "The data cited is misleading and lacks proper context.",
            "Fact check: The claim about 80% expert agreement is rated MEDIUM confidence. "
            "The actual figure from the 2023 survey was 72%.",
            "The debate has been productive so far. Both sides have presented valid points. "
            "The key area of disagreement centres on implementation timelines.",
            "FINAL VERDICT: The proposer presented stronger evidence overall. "
            "Score: Proposer 7/10, Critic 6/10. The debate was well-structured.",
        ]
    )


@pytest.fixture
def mock_agents(mock_provider: MockProvider) -> list[BaseAgent]:
    """A minimal set of agents using the mock provider."""
    from agents import Proposer, Critic, Judge

    return [
        Proposer(provider=mock_provider),
        Critic(provider=mock_provider),
        Judge(provider=mock_provider),
    ]


@pytest.fixture
def all_mock_agents(varied_provider: MockProvider) -> list[BaseAgent]:
    """All five agent types using the varied mock provider."""
    from agents import Proposer, Critic, FactChecker, Moderator, Judge

    return [
        Proposer(provider=MockProvider(responses=[varied_provider._responses[0]])),
        Critic(provider=MockProvider(responses=[varied_provider._responses[1]])),
        FactChecker(provider=MockProvider(responses=[varied_provider._responses[2]])),
        Moderator(provider=MockProvider(responses=[varied_provider._responses[3]])),
        Judge(provider=MockProvider(responses=[varied_provider._responses[4]])),
    ]


@pytest.fixture
def sample_state() -> DebateState:
    return DebateState(
        topic="Should AI development be regulated?",
        protocol="round_robin",
        current_turn=1,
        max_turns=6,
        history=[],
    )


@pytest.fixture
def sample_history() -> list[AgentResponse]:
    """A small debate history for testing metrics and visualization."""
    return [
        AgentResponse(
            agent_id="proposer_001",
            role=AgentRole.PROPOSER,
            content=(
                "AI regulation is essential. Studies show 80% of experts agree. "
                "For instance, the EU AI Act demonstrates feasibility. "
                "Data from 2023 surveys indicate growing public support at 65%."
            ),
            turn_number=1,
            tokens_used=120,
            provider="openai",
            model="gpt-4",
            latency_ms=450.0,
        ),
        AgentResponse(
            agent_id="critic_001",
            role=AgentRole.CRITIC,
            content=(
                "The argument contains a logical fallacy. The 80% figure is misleading "
                "because the survey sample was biased. Furthermore, regulation could "
                "stifle innovation, as demonstrated by historical precedents in biotech."
            ),
            turn_number=2,
            tokens_used=110,
            provider="anthropic",
            model="claude-3",
            latency_ms=380.0,
        ),
        AgentResponse(
            agent_id="fact_checker_001",
            role=AgentRole.FACT_CHECKER,
            content=(
                "Claim: 80% expert agreement. Confidence: MEDIUM. "
                "The actual survey (N=500) reported 72%. "
                "Claim: EU AI Act feasibility. Confidence: HIGH."
            ),
            turn_number=3,
            tokens_used=85,
            provider="cohere",
            model="command-r",
            latency_ms=290.0,
        ),
        AgentResponse(
            agent_id="judge_001",
            role=AgentRole.JUDGE,
            content=(
                "VERDICT: The proposer presented a reasonable case but overstated "
                "the expert consensus. The critic raised valid concerns about "
                "innovation impacts. Score: Proposer 7/10, Critic 6/10."
            ),
            turn_number=4,
            tokens_used=100,
            provider="openai",
            model="gpt-4",
            latency_ms=500.0,
        ),
    ]


@pytest_asyncio.fixture
async def test_db(tmp_path) -> DebateDatabase:
    """In-memory SQLite database for testing."""
    db = DebateDatabase(db_path=tmp_path / "test_debates.db")
    await db.connect()
    yield db
    await db.close()
