"""DebateManager – orchestrates multi-agent debates end-to-end.

Coordinates turn-taking, persists messages to SQLite, and collects metrics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from agents.base import AgentResponse, BaseAgent, DebateState
from data.database import DebateDatabase
from data.models import (
    AgentConfigRecord,
    DebateRecord,
    EvaluationRecord,
    MessageRecord,
)
from orchestration.protocols import DebateProtocol, create_protocol

logger = logging.getLogger(__name__)


@dataclass
class DebateResult:
    """Aggregated outcome of a completed debate."""

    debate_id: int
    topic: str
    protocol: str
    turns_completed: int
    history: list[AgentResponse]
    metrics: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    status: str = "completed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "debate_id": self.debate_id,
            "topic": self.topic,
            "protocol": self.protocol,
            "turns_completed": self.turns_completed,
            "total_messages": len(self.history),
            "metrics": self.metrics,
            "summary": self.summary,
            "status": self.status,
        }


class DebateManager:
    """High-level controller that runs a debate to completion.

    Parameters
    ----------
    agents : list[BaseAgent]
        Participating agents (order does not matter – the protocol decides).
    protocol : str | DebateProtocol
        Turn-taking strategy.
    db : DebateDatabase | None
        Optional database for persistence; if *None* a transient in-memory
        debate is run.
    """

    def __init__(
        self,
        agents: list[BaseAgent],
        protocol: str | DebateProtocol = "round_robin",
        db: DebateDatabase | None = None,
    ) -> None:
        self.agents = agents
        self.protocol: DebateProtocol = (
            create_protocol(protocol) if isinstance(protocol, str) else protocol
        )
        self.db = db

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_debate(
        self,
        topic: str,
        max_turns: int = 10,
        **kwargs: Any,
    ) -> DebateResult:
        """Execute the full debate loop and return results."""
        debate_id = await self._init_debate(topic)
        history: list[AgentResponse] = []
        status = "completed"

        logger.info(
            "Starting debate #%d: '%s' [%s, %d turns, %d agents]",
            debate_id,
            topic,
            self.protocol.name,
            max_turns,
            len(self.agents),
        )

        try:
            for turn in range(1, max_turns + 1):
                speakers = self.protocol.get_turn_order(
                    self.agents, turn, max_turns
                )
                for agent in speakers:
                    state = DebateState(
                        topic=topic,
                        protocol=self.protocol.name,
                        current_turn=turn,
                        max_turns=max_turns,
                        history=history,
                    )
                    response = await agent.process_turn(state)
                    history.append(response)
                    await self._persist_message(debate_id, response)
                    logger.info(
                        "[Turn %d] %s (%s/%s): %s",
                        turn,
                        agent.role.value,
                        response.provider,
                        response.model,
                        response.content[:80] + "…" if len(response.content) > 80 else response.content,
                    )
        except Exception:
            status = "failed"
            logger.exception("Debate #%d failed", debate_id)
            raise
        finally:
            if self.db:
                await self.db.update_debate_status(debate_id, status)

        # Build summary from judge's last message (if present)
        judge_msgs = [r for r in history if r.role.value == "judge"]
        summary = judge_msgs[-1].content if judge_msgs else ""

        # Basic metrics
        total_tokens = sum(r.tokens_used for r in history)
        avg_latency = (
            sum(r.latency_ms for r in history) / len(history) if history else 0
        )
        metrics = {
            "total_tokens": total_tokens,
            "avg_latency_ms": round(avg_latency, 1),
            "total_messages": len(history),
            "turns_completed": max_turns,
            "agents_used": {
                a.agent_id: {"provider": a.provider.name, "model": a.provider.model}
                for a in self.agents
            },
        }

        result = DebateResult(
            debate_id=debate_id,
            topic=topic,
            protocol=self.protocol.name,
            turns_completed=max_turns,
            history=history,
            metrics=metrics,
            summary=summary,
            status=status,
        )

        # Persist metrics
        if self.db:
            for name, value in [
                ("total_tokens", total_tokens),
                ("avg_latency_ms", avg_latency),
                ("total_messages", len(history)),
            ]:
                await self.db.save_evaluation(
                    EvaluationRecord(
                        debate_id=debate_id,
                        metric_name=name,
                        metric_value=float(value),
                    )
                )

        logger.info(
            "Debate #%d completed: %d messages, %d tokens",
            debate_id,
            len(history),
            total_tokens,
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _init_debate(self, topic: str) -> int:
        """Create the debate record in the DB (or return a dummy id)."""
        if self.db is None:
            return 0  # transient debate

        debate_id = await self.db.create_debate(
            DebateRecord(topic=topic, protocol=self.protocol.name, status="running")
        )

        # Persist agent configs
        for agent in self.agents:
            await self.db.save_agent_config(
                AgentConfigRecord(
                    debate_id=debate_id,
                    agent_id=agent.agent_id,
                    provider=agent.provider.name,
                    model=agent.provider.model,
                    config_json=json.dumps(
                        {
                            "temperature": agent.temperature,
                            "max_tokens": agent.max_tokens,
                            "role": agent.role.value,
                        }
                    ),
                )
            )

        return debate_id

    async def _persist_message(self, debate_id: int, response: AgentResponse) -> None:
        """Save a single agent response to the database."""
        if self.db is None:
            return
        await self.db.save_message(
            MessageRecord(
                debate_id=debate_id,
                agent_id=response.agent_id,
                role=response.role.value,
                content=response.content,
                turn_number=response.turn_number,
                tokens_used=response.tokens_used,
                provider=response.provider,
                model=response.model,
                latency_ms=response.latency_ms,
            )
        )
