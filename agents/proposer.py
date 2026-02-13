"""Proposer agent – generates initial arguments with supporting evidence."""

from __future__ import annotations

from typing import Any

from agents.base import AgentRole, BaseAgent, DebateState
from agents.llm_provider import LLMProvider

_SYSTEM_PROMPT = """\
You are a skilled debate **Proposer**. Your responsibilities:
1. Formulate clear, well-structured arguments in favour of a position.
2. Support every claim with evidence, examples, or logical reasoning.
3. Anticipate counter-arguments and pre-emptively address them.
4. Be persuasive yet intellectually honest.
5. Structure your response with a clear thesis, supporting points, and a conclusion.

Keep responses focused and concise.
"""


class Proposer(BaseAgent):
    """Agent that proposes and defends positions with evidence-backed arguments."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        agent_id: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        system_prompt: str = _SYSTEM_PROMPT,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            role=AgentRole.PROPOSER,
            provider=provider,
            agent_id=agent_id,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )

    def _build_turn_prompt(self, state: DebateState) -> str:
        if state.current_turn == 1:
            return (
                f"Present a well-structured opening argument on the topic: "
                f"'{state.topic}'. Include a clear thesis statement, at least "
                f"three supporting points with evidence, and a brief conclusion."
            )

        # Subsequent turns: respond to criticism
        recent_criticisms = [
            r for r in state.history[-5:]
            if r.role in (AgentRole.CRITIC, AgentRole.FACT_CHECKER)
        ]
        if recent_criticisms:
            points = "\n".join(f"- {c.role.value}: {c.content[:200]}" for c in recent_criticisms)
            return (
                f"Turn {state.current_turn}/{state.max_turns}. "
                f"Address the following challenges to your position:\n{points}\n\n"
                f"Strengthen your argument with additional evidence or refined reasoning."
            )

        return (
            f"Turn {state.current_turn}/{state.max_turns}. "
            f"Continue building your case on '{state.topic}'. "
            f"Add new supporting evidence or refine previous arguments."
        )
