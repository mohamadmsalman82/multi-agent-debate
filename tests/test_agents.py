"""Tests for agent implementations (base + specialised)."""

from __future__ import annotations

import pytest

from agents.base import AgentResponse, AgentRole, BaseAgent, DebateState
from agents import Proposer, Critic, FactChecker, Moderator, Judge
from tests.conftest import MockProvider


class TestBaseAgent:
    @pytest.mark.asyncio
    async def test_generate_response_returns_agent_response(self, mock_provider):
        agent = BaseAgent(
            role=AgentRole.PROPOSER,
            provider=mock_provider,
            system_prompt="You are a test agent.",
        )
        resp = await agent.generate_response("context here", "prompt here")
        assert isinstance(resp, AgentResponse)
        assert resp.role == AgentRole.PROPOSER
        assert resp.provider == "mock"
        assert len(resp.content) > 0

    @pytest.mark.asyncio
    async def test_process_turn(self, mock_provider, sample_state):
        agent = BaseAgent(
            role=AgentRole.CRITIC,
            provider=mock_provider,
        )
        resp = await agent.process_turn(sample_state)
        assert resp.turn_number == sample_state.current_turn
        assert resp.role == AgentRole.CRITIC

    @pytest.mark.asyncio
    async def test_memory_accumulates(self, mock_provider):
        agent = BaseAgent(role=AgentRole.PROPOSER, provider=mock_provider)
        await agent.generate_response("ctx", "p1")
        await agent.generate_response("ctx", "p2")
        # 2 user + 2 assistant messages
        assert len(agent._conversation_history) == 4

    @pytest.mark.asyncio
    async def test_clear_memory(self, mock_provider):
        agent = BaseAgent(role=AgentRole.PROPOSER, provider=mock_provider)
        await agent.generate_response("ctx", "p1")
        agent.clear_memory()
        assert len(agent._conversation_history) == 0

    @pytest.mark.asyncio
    async def test_total_tokens_tracked(self, mock_provider):
        agent = BaseAgent(role=AgentRole.PROPOSER, provider=mock_provider)
        await agent.generate_response("ctx", "p")
        assert agent.total_tokens_used > 0

    def test_agent_id_auto_generated(self, mock_provider):
        agent = BaseAgent(role=AgentRole.PROPOSER, provider=mock_provider)
        assert agent.agent_id.startswith("proposer_")
        assert len(agent.agent_id) > len("proposer_")

    def test_agent_id_custom(self, mock_provider):
        agent = BaseAgent(
            role=AgentRole.PROPOSER, provider=mock_provider, agent_id="my_agent"
        )
        assert agent.agent_id == "my_agent"

    def test_repr(self, mock_provider):
        agent = BaseAgent(role=AgentRole.PROPOSER, provider=mock_provider)
        r = repr(agent)
        assert "BaseAgent" in r
        assert "proposer" in r


class TestProposer:
    @pytest.mark.asyncio
    async def test_opening_argument(self, mock_provider, sample_state):
        agent = Proposer(provider=mock_provider)
        resp = await agent.process_turn(sample_state)
        assert resp.role == AgentRole.PROPOSER

    @pytest.mark.asyncio
    async def test_responds_to_criticism(self, mock_provider, sample_history):
        agent = Proposer(provider=mock_provider)
        state = DebateState(
            topic="AI regulation",
            protocol="round_robin",
            current_turn=3,
            max_turns=6,
            history=sample_history,
        )
        resp = await agent.process_turn(state)
        assert resp.role == AgentRole.PROPOSER


class TestCritic:
    @pytest.mark.asyncio
    async def test_critique_with_history(self, mock_provider, sample_history):
        agent = Critic(provider=mock_provider)
        state = DebateState(
            topic="AI regulation",
            protocol="round_robin",
            current_turn=2,
            max_turns=6,
            history=sample_history[:1],  # just proposer's message
        )
        resp = await agent.process_turn(state)
        assert resp.role == AgentRole.CRITIC

    @pytest.mark.asyncio
    async def test_critique_without_history(self, mock_provider):
        agent = Critic(provider=mock_provider)
        state = DebateState(
            topic="AI regulation",
            protocol="round_robin",
            current_turn=1,
            max_turns=6,
            history=[],
        )
        resp = await agent.process_turn(state)
        assert resp.role == AgentRole.CRITIC


class TestFactChecker:
    @pytest.mark.asyncio
    async def test_fact_check(self, mock_provider, sample_history):
        agent = FactChecker(provider=mock_provider)
        state = DebateState(
            topic="AI regulation",
            protocol="round_robin",
            current_turn=3,
            max_turns=6,
            history=sample_history[:2],
        )
        resp = await agent.process_turn(state)
        assert resp.role == AgentRole.FACT_CHECKER

    @pytest.mark.asyncio
    async def test_no_claims_to_check(self, mock_provider):
        agent = FactChecker(provider=mock_provider)
        state = DebateState(
            topic="AI regulation",
            protocol="round_robin",
            current_turn=1,
            max_turns=6,
            history=[],
        )
        resp = await agent.process_turn(state)
        assert resp.role == AgentRole.FACT_CHECKER


class TestModerator:
    @pytest.mark.asyncio
    async def test_opening_moderation(self, mock_provider, sample_state):
        agent = Moderator(provider=mock_provider)
        resp = await agent.process_turn(sample_state)
        assert resp.role == AgentRole.MODERATOR

    @pytest.mark.asyncio
    async def test_final_summary(self, mock_provider, sample_history):
        agent = Moderator(provider=mock_provider)
        state = DebateState(
            topic="AI regulation",
            protocol="round_robin",
            current_turn=6,
            max_turns=6,
            history=sample_history,
        )
        resp = await agent.process_turn(state)
        assert resp.role == AgentRole.MODERATOR


class TestJudge:
    @pytest.mark.asyncio
    async def test_interim_assessment(self, mock_provider, sample_history):
        agent = Judge(provider=mock_provider)
        state = DebateState(
            topic="AI regulation",
            protocol="round_robin",
            current_turn=3,
            max_turns=6,
            history=sample_history,
        )
        resp = await agent.process_turn(state)
        assert resp.role == AgentRole.JUDGE

    @pytest.mark.asyncio
    async def test_final_verdict(self, mock_provider, sample_history):
        agent = Judge(provider=mock_provider)
        state = DebateState(
            topic="AI regulation",
            protocol="round_robin",
            current_turn=6,
            max_turns=6,
            history=sample_history,
        )
        resp = await agent.process_turn(state)
        assert resp.role == AgentRole.JUDGE
