"""Tests for debate protocols."""

from __future__ import annotations

import pytest

from agents.base import AgentRole
from agents import Proposer, Critic, FactChecker, Moderator, Judge
from orchestration.protocols import (
    AdversarialProtocol,
    CollaborativeProtocol,
    RoundRobinProtocol,
    create_protocol,
)
from tests.conftest import MockProvider


@pytest.fixture
def five_agents():
    p = MockProvider()
    return [
        Proposer(provider=p),
        Critic(provider=p),
        FactChecker(provider=p),
        Moderator(provider=p),
        Judge(provider=p),
    ]


@pytest.fixture
def three_agents():
    p = MockProvider()
    return [
        Proposer(provider=p),
        Critic(provider=p),
        Judge(provider=p),
    ]


class TestRoundRobinProtocol:
    def test_non_final_turn_excludes_judge(self, five_agents):
        proto = RoundRobinProtocol()
        order = proto.get_turn_order(five_agents, turn=1, max_turns=6)
        roles = [a.role for a in order]
        assert AgentRole.JUDGE not in roles
        assert AgentRole.PROPOSER in roles
        assert AgentRole.CRITIC in roles

    def test_final_turn_includes_judge(self, five_agents):
        proto = RoundRobinProtocol()
        order = proto.get_turn_order(five_agents, turn=6, max_turns=6)
        roles = [a.role for a in order]
        assert AgentRole.JUDGE in roles

    def test_ordering(self, five_agents):
        proto = RoundRobinProtocol()
        order = proto.get_turn_order(five_agents, turn=6, max_turns=6)
        roles = [a.role for a in order]
        # Proposer before Critic before Judge
        assert roles.index(AgentRole.PROPOSER) < roles.index(AgentRole.CRITIC)
        assert roles.index(AgentRole.CRITIC) < roles.index(AgentRole.JUDGE)


class TestAdversarialProtocol:
    def test_odd_turn_proposer(self, five_agents):
        proto = AdversarialProtocol()
        order = proto.get_turn_order(five_agents, turn=1, max_turns=6)
        assert order[0].role == AgentRole.PROPOSER

    def test_even_turn_critic(self, five_agents):
        proto = AdversarialProtocol()
        order = proto.get_turn_order(five_agents, turn=2, max_turns=6)
        assert order[0].role == AgentRole.CRITIC

    def test_fact_checker_every_3rd(self, five_agents):
        proto = AdversarialProtocol()
        order = proto.get_turn_order(five_agents, turn=3, max_turns=6)
        roles = [a.role for a in order]
        assert AgentRole.FACT_CHECKER in roles

    def test_judge_final_only(self, five_agents):
        proto = AdversarialProtocol()
        mid = proto.get_turn_order(five_agents, turn=3, max_turns=6)
        final = proto.get_turn_order(five_agents, turn=6, max_turns=6)
        assert AgentRole.JUDGE not in [a.role for a in mid]
        assert AgentRole.JUDGE in [a.role for a in final]


class TestCollaborativeProtocol:
    def test_moderator_leads(self, five_agents):
        proto = CollaborativeProtocol()
        order = proto.get_turn_order(five_agents, turn=1, max_turns=6)
        assert order[0].role == AgentRole.MODERATOR

    def test_fact_checker_every_other_turn(self, five_agents):
        proto = CollaborativeProtocol()
        even = proto.get_turn_order(five_agents, turn=2, max_turns=6)
        odd = proto.get_turn_order(five_agents, turn=3, max_turns=6)
        even_roles = [a.role for a in even]
        odd_roles = [a.role for a in odd]
        assert AgentRole.FACT_CHECKER in even_roles
        assert AgentRole.FACT_CHECKER not in odd_roles

    def test_judge_on_final(self, five_agents):
        proto = CollaborativeProtocol()
        final = proto.get_turn_order(five_agents, turn=6, max_turns=6)
        assert AgentRole.JUDGE in [a.role for a in final]


class TestCreateProtocol:
    def test_known_protocols(self):
        for name in ("round_robin", "adversarial", "collaborative"):
            proto = create_protocol(name)
            assert proto.name == name

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown protocol"):
            create_protocol("nonexistent")
