"""Tests for evaluation validators."""

from __future__ import annotations

import pytest

from agents.base import AgentResponse, AgentRole, DebateState
from evaluation.validators import DebateValidator


@pytest.fixture
def validator():
    return DebateValidator()


class TestDebateValidator:
    def test_valid_response(self, validator: DebateValidator, sample_state):
        resp = AgentResponse(
            agent_id="a", role=AgentRole.PROPOSER,
            content="This is a valid response with enough content to pass.",
            turn_number=1, tokens_used=50, provider="p", model="m", latency_ms=0,
        )
        result = validator.validate_response(resp, sample_state)
        assert result.valid
        assert len(result.issues) == 0

    def test_too_short(self, validator: DebateValidator, sample_state):
        resp = AgentResponse(
            agent_id="a", role=AgentRole.PROPOSER,
            content="Short",
            turn_number=1, tokens_used=5, provider="p", model="m", latency_ms=0,
        )
        result = validator.validate_response(resp, sample_state)
        assert not result.valid
        assert any("too short" in issue for issue in result.issues)

    def test_empty_content(self, validator: DebateValidator, sample_state):
        resp = AgentResponse(
            agent_id="a", role=AgentRole.PROPOSER,
            content="   ",
            turn_number=1, tokens_used=0, provider="p", model="m", latency_ms=0,
        )
        result = validator.validate_response(resp, sample_state)
        assert not result.valid

    def test_too_long(self, sample_state):
        validator = DebateValidator(max_response_length=50)
        resp = AgentResponse(
            agent_id="a", role=AgentRole.PROPOSER,
            content="x" * 100,
            turn_number=1, tokens_used=50, provider="p", model="m", latency_ms=0,
        )
        result = validator.validate_response(resp, sample_state)
        assert not result.valid
        assert any("too long" in issue for issue in result.issues)

    def test_duplicate_detection(self, validator: DebateValidator):
        existing = AgentResponse(
            agent_id="b", role=AgentRole.CRITIC,
            content="This is a previously stated argument.",
            turn_number=1, tokens_used=50, provider="p", model="m", latency_ms=0,
        )
        state = DebateState(
            topic="t", protocol="p", current_turn=2, max_turns=6, history=[existing],
        )
        resp = AgentResponse(
            agent_id="a", role=AgentRole.PROPOSER,
            content="This is a previously stated argument.",
            turn_number=2, tokens_used=50, provider="p", model="m", latency_ms=0,
        )
        result = validator.validate_response(resp, state)
        assert not result.valid
        assert any("Duplicate" in issue for issue in result.issues)

    def test_validate_debate_config_valid(self, validator: DebateValidator):
        result = validator.validate_debate_config(3, 6, "round_robin")
        assert result.valid

    def test_validate_debate_config_too_few_agents(self, validator: DebateValidator):
        result = validator.validate_debate_config(1, 6, "round_robin")
        assert not result.valid

    def test_validate_debate_config_bad_protocol(self, validator: DebateValidator):
        result = validator.validate_debate_config(3, 6, "unknown")
        assert not result.valid

    def test_validate_debate_config_too_many_turns(self, validator: DebateValidator):
        result = validator.validate_debate_config(3, 100, "round_robin")
        assert not result.valid
