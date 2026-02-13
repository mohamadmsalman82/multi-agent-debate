"""Tests for the data layer (database + models)."""

from __future__ import annotations

import pytest

from data.database import DebateDatabase
from data.models import (
    AgentConfigRecord,
    DebateRecord,
    EvaluationRecord,
    MessageRecord,
)


class TestDebateRecord:
    def test_metadata_property(self):
        rec = DebateRecord(topic="t", protocol="p", metadata_json='{"key": "val"}')
        assert rec.metadata == {"key": "val"}

    def test_default_status(self):
        rec = DebateRecord(topic="t", protocol="p")
        assert rec.status == "pending"


class TestDatabase:
    @pytest.mark.asyncio
    async def test_create_and_get_debate(self, test_db: DebateDatabase):
        rec = DebateRecord(topic="Test topic", protocol="round_robin")
        debate_id = await test_db.create_debate(rec)
        assert debate_id is not None
        assert debate_id > 0

        fetched = await test_db.get_debate(debate_id)
        assert fetched is not None
        assert fetched.topic == "Test topic"
        assert fetched.protocol == "round_robin"
        assert fetched.status == "running" or fetched.status == "pending"

    @pytest.mark.asyncio
    async def test_update_debate_status(self, test_db: DebateDatabase):
        debate_id = await test_db.create_debate(
            DebateRecord(topic="t", protocol="p")
        )
        await test_db.update_debate_status(debate_id, "completed")
        fetched = await test_db.get_debate(debate_id)
        assert fetched is not None
        assert fetched.status == "completed"

    @pytest.mark.asyncio
    async def test_list_debates(self, test_db: DebateDatabase):
        for i in range(5):
            await test_db.create_debate(
                DebateRecord(topic=f"Topic {i}", protocol="p")
            )
        debates = await test_db.list_debates(limit=3)
        assert len(debates) == 3
        # Most recent first
        assert debates[0].topic == "Topic 4"

    @pytest.mark.asyncio
    async def test_get_nonexistent_debate(self, test_db: DebateDatabase):
        assert await test_db.get_debate(999) is None

    @pytest.mark.asyncio
    async def test_save_and_get_messages(self, test_db: DebateDatabase):
        debate_id = await test_db.create_debate(
            DebateRecord(topic="t", protocol="p")
        )
        for i in range(3):
            await test_db.save_message(
                MessageRecord(
                    debate_id=debate_id,
                    agent_id=f"agent_{i}",
                    role="proposer",
                    content=f"Message {i}",
                    turn_number=i + 1,
                    tokens_used=50,
                    provider="mock",
                    model="m",
                )
            )

        messages = await test_db.get_messages(debate_id)
        assert len(messages) == 3
        assert messages[0].content == "Message 0"
        assert messages[2].turn_number == 3

    @pytest.mark.asyncio
    async def test_save_and_get_evaluations(self, test_db: DebateDatabase):
        debate_id = await test_db.create_debate(
            DebateRecord(topic="t", protocol="p")
        )
        await test_db.save_evaluation(
            EvaluationRecord(
                debate_id=debate_id, metric_name="coherence", metric_value=0.85
            )
        )
        await test_db.save_evaluation(
            EvaluationRecord(
                debate_id=debate_id, metric_name="relevance", metric_value=0.72
            )
        )

        evals = await test_db.get_evaluations(debate_id)
        assert len(evals) == 2
        assert evals[0].metric_name == "coherence"
        assert evals[0].metric_value == 0.85

    @pytest.mark.asyncio
    async def test_save_and_get_agent_configs(self, test_db: DebateDatabase):
        debate_id = await test_db.create_debate(
            DebateRecord(topic="t", protocol="p")
        )
        await test_db.save_agent_config(
            AgentConfigRecord(
                debate_id=debate_id,
                agent_id="proposer_001",
                provider="openai",
                model="gpt-4",
                config_json='{"temperature": 0.7}',
            )
        )

        configs = await test_db.get_agent_configs(debate_id)
        assert len(configs) == 1
        assert configs[0].provider == "openai"
        assert configs[0].config == {"temperature": 0.7}

    @pytest.mark.asyncio
    async def test_get_debate_stats(self, test_db: DebateDatabase):
        debate_id = await test_db.create_debate(
            DebateRecord(topic="t", protocol="p")
        )
        for role in ["proposer", "proposer", "critic"]:
            await test_db.save_message(
                MessageRecord(
                    debate_id=debate_id,
                    agent_id=f"{role}_001",
                    role=role,
                    content="msg",
                    turn_number=1,
                    tokens_used=100,
                    provider="mock",
                    model="m",
                    latency_ms=200.0,
                )
            )

        stats = await test_db.get_debate_stats(debate_id)
        assert stats["total_messages"] == 3
        assert stats["total_tokens"] == 300
        assert stats["avg_latency_ms"] == 200.0
        assert stats["participation"]["proposer"] == 2
        assert stats["participation"]["critic"] == 1
