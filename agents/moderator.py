"""Moderator agent – manages debate flow, summarises, and enforces rules."""

from __future__ import annotations

from typing import Any

from agents.base import AgentRole, BaseAgent, DebateState
from agents.llm_provider import LLMProvider

_SYSTEM_PROMPT = """\
You are the **Moderator** of a structured debate. Your responsibilities:
1. Summarise key points from the discussion so far.
2. Identify areas of agreement and disagreement.
3. Guide the conversation toward productive territory.
4. Enforce civil discourse — flag ad-hominem attacks or off-topic tangents.
5. Pose clarifying questions to participants when arguments are vague.
6. Track whether the debate is converging toward consensus or deepening disagreement.

Be neutral, fair, and concise. Never take a side.
"""


class Moderator(BaseAgent):
    """Agent that manages debate flow, enforces rules, and summarises progress."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        agent_id: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 400,
        system_prompt: str = _SYSTEM_PROMPT,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            role=AgentRole.MODERATOR,
            provider=provider,
            agent_id=agent_id,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )

    def _build_turn_prompt(self, state: DebateState) -> str:
        turn = state.current_turn
        max_t = state.max_turns

        if turn <= 2:
            return (
                f"Turn {turn}/{max_t}. The debate on '{state.topic}' has just begun. "
                f"Briefly set the stage: outline the key questions at stake and "
                f"what you expect each side to address."
            )

        if turn >= max_t - 1:
            return (
                f"Turn {turn}/{max_t}. The debate is nearing its conclusion. "
                f"Provide a comprehensive summary of:\n"
                f"- Key arguments from each side\n"
                f"- Areas of agreement and disagreement\n"
                f"- Whether consensus was reached\n"
                f"- Outstanding questions that remain"
            )

        # Mid-debate
        participation = {}
        for r in state.history:
            participation[r.role.value] = participation.get(r.role.value, 0) + 1
        stats = ", ".join(f"{k}: {v}" for k, v in participation.items())

        return (
            f"Turn {turn}/{max_t}. Participation so far: {stats}.\n"
            f"Summarise the debate's progress. Identify the strongest arguments "
            f"on each side and suggest what should be addressed next."
        )
