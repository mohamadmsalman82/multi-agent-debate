"""Tests for the DebateManager orchestration."""

from __future__ import annotations

import pytest

from agents import Proposer, Critic, Judge
from data.database import DebateDatabase
from orchestration.debate_manager import DebateManager, DebateResult
from tests.conftest import MockProvider


class TestDebateManagerTransient:
    """Tests without database (in-memory only)."""

    @pytest.mark.asyncio
    async def test_run_basic_debate(self, mock_agents):
        manager = DebateManager(agents=mock_agents, protocol="round_robin")
        result = await manager.run_debate(
            topic="Test topic",
            max_turns=3,
        )
        assert isinstance(result, DebateResult)
        assert result.topic == "Test topic"
        assert result.protocol == "round_robin"
        assert result.status == "completed"
        assert len(result.history) > 0

    @pytest.mark.asyncio
    async def test_debate_result_metrics(self, mock_agents):
        manager = DebateManager(agents=mock_agents, protocol="round_robin")
        result = await manager.run_debate(topic="Metrics test", max_turns=2)
        assert "total_tokens" in result.metrics
        assert "total_messages" in result.metrics
        assert result.metrics["total_messages"] == len(result.history)

    @pytest.mark.asyncio
    async def test_adversarial_protocol(self, mock_agents):
        manager = DebateManager(agents=mock_agents, protocol="adversarial")
        result = await manager.run_debate(topic="Adversarial test", max_turns=4)
        assert result.status == "completed"
        assert len(result.history) > 0

    @pytest.mark.asyncio
    async def test_collaborative_protocol(self, all_mock_agents):
        manager = DebateManager(agents=all_mock_agents, protocol="collaborative")
        result = await manager.run_debate(topic="Collaborative test", max_turns=3)
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_debate_result_to_dict(self, mock_agents):
        manager = DebateManager(agents=mock_agents, protocol="round_robin")
        result = await manager.run_debate(topic="Dict test", max_turns=2)
        d = result.to_dict()
        assert d["topic"] == "Dict test"
        assert d["protocol"] == "round_robin"
        assert d["status"] == "completed"
        assert isinstance(d["total_messages"], int)


class TestDebateManagerWithDB:
    """Tests with SQLite persistence."""

    @pytest.mark.asyncio
    async def test_debate_persisted(self, mock_agents, test_db: DebateDatabase):
        manager = DebateManager(
            agents=mock_agents, protocol="round_robin", db=test_db
        )
        result = await manager.run_debate(topic="Persist test", max_turns=2)
        assert result.debate_id > 0

        # Verify in DB
        debate = await test_db.get_debate(result.debate_id)
        assert debate is not None
        assert debate.topic == "Persist test"
        assert debate.status == "completed"

    @pytest.mark.asyncio
    async def test_messages_persisted(self, mock_agents, test_db: DebateDatabase):
        manager = DebateManager(
            agents=mock_agents, protocol="round_robin", db=test_db
        )
        result = await manager.run_debate(topic="Msg persist", max_turns=2)

        messages = await test_db.get_messages(result.debate_id)
        assert len(messages) == len(result.history)

    @pytest.mark.asyncio
    async def test_evaluations_persisted(self, mock_agents, test_db: DebateDatabase):
        manager = DebateManager(
            agents=mock_agents, protocol="round_robin", db=test_db
        )
        result = await manager.run_debate(topic="Eval persist", max_turns=2)

        evals = await test_db.get_evaluations(result.debate_id)
        metric_names = [e.metric_name for e in evals]
        assert "total_tokens" in metric_names
        assert "total_messages" in metric_names

    @pytest.mark.asyncio
    async def test_agent_configs_persisted(self, mock_agents, test_db: DebateDatabase):
        manager = DebateManager(
            agents=mock_agents, protocol="round_robin", db=test_db
        )
        result = await manager.run_debate(topic="Config persist", max_turns=2)

        configs = await test_db.get_agent_configs(result.debate_id)
        assert len(configs) == len(mock_agents)
