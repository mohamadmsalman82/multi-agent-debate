"""Critic agent – challenges arguments and identifies logical fallacies."""

from __future__ import annotations

from typing import Any

from agents.base import AgentRole, BaseAgent, DebateState
from agents.llm_provider import LLMProvider

_SYSTEM_PROMPT = """\
You are a rigorous debate **Critic**. Your responsibilities:
1. Carefully analyse arguments presented by other participants.
2. Identify logical fallacies, unsupported claims, and weak reasoning.
3. Provide specific, constructive counter-arguments.
4. Challenge assumptions and demand stronger evidence.
5. Be intellectually honest — acknowledge strong points even as you critique.

Structure your critique clearly: state what you are challenging, why it is
problematic, and what would strengthen the argument.
"""


class Critic(BaseAgent):
    """Agent that challenges arguments, identifies fallacies, and raises objections."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        agent_id: str | None = None,
        temperature: float = 0.8,
        max_tokens: int = 500,
        system_prompt: str = _SYSTEM_PROMPT,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            role=AgentRole.CRITIC,
            provider=provider,
            agent_id=agent_id,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )

    def _build_turn_prompt(self, state: DebateState) -> str:
        # Find the most recent proposer argument to critique
        proposer_msgs = [
            r for r in state.history if r.role == AgentRole.PROPOSER
        ]
        if proposer_msgs:
            latest = proposer_msgs[-1]
            return (
                f"Turn {state.current_turn}/{state.max_turns}. "
                f"Critically analyse the following argument from the Proposer:\n\n"
                f"\"{latest.content[:600]}\"\n\n"
                f"Identify weaknesses, logical fallacies, unsupported claims, or "
                f"missing evidence. Provide specific counter-arguments."
            )

        return (
            f"Turn {state.current_turn}/{state.max_turns}. "
            f"The debate topic is '{state.topic}'. Prepare a critical perspective "
            f"challenging common arguments on this topic."
        )
