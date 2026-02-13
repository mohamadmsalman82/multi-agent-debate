"""Debate protocols that control turn-taking order.

Each protocol decides which agent speaks next given the current state.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from agents.base import AgentRole, BaseAgent

logger = logging.getLogger(__name__)


class DebateProtocol(ABC):
    """Base class for turn-taking strategies."""

    name: str

    @abstractmethod
    def get_turn_order(
        self,
        agents: list[BaseAgent],
        turn: int,
        max_turns: int,
    ) -> list[BaseAgent]:
        """Return the ordered list of agents that should speak this turn."""
        ...


class RoundRobinProtocol(DebateProtocol):
    """Every agent speaks once per turn in a fixed order.

    Order: Proposer -> Critic -> FactChecker -> Moderator -> Judge (if present).
    The Judge only speaks on the final turn.
    """

    name = "round_robin"

    _ROLE_ORDER = [
        AgentRole.PROPOSER,
        AgentRole.CRITIC,
        AgentRole.FACT_CHECKER,
        AgentRole.MODERATOR,
        AgentRole.JUDGE,
    ]

    def get_turn_order(
        self,
        agents: list[BaseAgent],
        turn: int,
        max_turns: int,
    ) -> list[BaseAgent]:
        by_role: dict[AgentRole, BaseAgent] = {a.role: a for a in agents}
        ordered: list[BaseAgent] = []

        for role in self._ROLE_ORDER:
            agent = by_role.get(role)
            if agent is None:
                continue
            # Judge only speaks on the final turn
            if role == AgentRole.JUDGE and turn < max_turns:
                continue
            ordered.append(agent)

        return ordered


class AdversarialProtocol(DebateProtocol):
    """Proposer and Critic alternate, with periodic FactChecker and final Judge.

    Turn pattern:
      odd  turns → Proposer
      even turns → Critic
      every 3rd turn → FactChecker
      final turn  → Judge
    """

    name = "adversarial"

    def get_turn_order(
        self,
        agents: list[BaseAgent],
        turn: int,
        max_turns: int,
    ) -> list[BaseAgent]:
        by_role: dict[AgentRole, BaseAgent] = {a.role: a for a in agents}
        ordered: list[BaseAgent] = []

        # Primary speakers
        if turn % 2 == 1 and AgentRole.PROPOSER in by_role:
            ordered.append(by_role[AgentRole.PROPOSER])
        elif turn % 2 == 0 and AgentRole.CRITIC in by_role:
            ordered.append(by_role[AgentRole.CRITIC])

        # FactChecker every 3rd turn
        if turn % 3 == 0 and AgentRole.FACT_CHECKER in by_role:
            ordered.append(by_role[AgentRole.FACT_CHECKER])

        # Judge on final turn
        if turn == max_turns and AgentRole.JUDGE in by_role:
            ordered.append(by_role[AgentRole.JUDGE])

        return ordered


class CollaborativeProtocol(DebateProtocol):
    """All non-judge agents work together, building on each other.

    Moderator opens each turn, then participants respond.
    Judge provides interim feedback every 3 turns and final verdict.
    """

    name = "collaborative"

    def get_turn_order(
        self,
        agents: list[BaseAgent],
        turn: int,
        max_turns: int,
    ) -> list[BaseAgent]:
        by_role: dict[AgentRole, BaseAgent] = {a.role: a for a in agents}
        ordered: list[BaseAgent] = []

        # Moderator leads
        if AgentRole.MODERATOR in by_role:
            ordered.append(by_role[AgentRole.MODERATOR])

        # Then proposer and critic collaborate
        if AgentRole.PROPOSER in by_role:
            ordered.append(by_role[AgentRole.PROPOSER])
        if AgentRole.CRITIC in by_role:
            ordered.append(by_role[AgentRole.CRITIC])

        # FactChecker every other turn
        if turn % 2 == 0 and AgentRole.FACT_CHECKER in by_role:
            ordered.append(by_role[AgentRole.FACT_CHECKER])

        # Judge interim feedback every 3 turns or final
        if (turn % 3 == 0 or turn == max_turns) and AgentRole.JUDGE in by_role:
            ordered.append(by_role[AgentRole.JUDGE])

        return ordered


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROTOCOLS: dict[str, type[DebateProtocol]] = {
    "round_robin": RoundRobinProtocol,
    "adversarial": AdversarialProtocol,
    "collaborative": CollaborativeProtocol,
}


def create_protocol(name: str) -> DebateProtocol:
    """Instantiate a protocol by name."""
    cls = _PROTOCOLS.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown protocol {name!r}. Choose from {list(_PROTOCOLS)}")
    return cls()
