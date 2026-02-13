"""Judge agent – evaluates arguments and determines debate outcomes."""

from __future__ import annotations

from typing import Any

from agents.base import AgentRole, BaseAgent, DebateState
from agents.llm_provider import LLMProvider

_SYSTEM_PROMPT = """\
You are the **Judge** of a structured debate. Your responsibilities:
1. Evaluate the quality of arguments from all participants.
2. Assess evidence strength, logical coherence, and persuasiveness.
3. Determine a final verdict with clear reasoning.
4. Score each participant's performance (1-10) across these dimensions:
   - Argument Quality (clarity, structure, depth)
   - Evidence Strength (data, examples, citations)
   - Logical Coherence (reasoning, fallacy avoidance)
   - Persuasiveness (rhetorical effectiveness)
   - Responsiveness (addressing counter-arguments)

Be fair, thorough, and justify every score. Structure your verdict clearly.
"""


class Judge(BaseAgent):
    """Agent that evaluates arguments and delivers the final verdict."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        agent_id: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 600,
        system_prompt: str = _SYSTEM_PROMPT,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            role=AgentRole.JUDGE,
            provider=provider,
            agent_id=agent_id,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )

    def _build_turn_prompt(self, state: DebateState) -> str:
        turn = state.current_turn
        max_t = state.max_turns

        if turn < max_t:
            # Interim assessment
            return (
                f"Turn {turn}/{max_t}. Provide an interim assessment of the "
                f"debate on '{state.topic}'.\n"
                f"Rate the arguments presented so far and highlight:\n"
                f"- Which arguments are strongest and why\n"
                f"- Which arguments need more evidence\n"
                f"- Current direction of the debate"
            )

        # Final verdict
        args_summary = "\n".join(
            f"[Turn {r.turn_number}] {r.role.value}: {r.content[:300]}"
            for r in state.history
        )
        return (
            f"The debate on '{state.topic}' has concluded after {max_t} turns.\n\n"
            f"Full debate transcript:\n{args_summary}\n\n"
            f"Deliver your FINAL VERDICT. Include:\n"
            f"1. Overall winner (or declare a draw) with justification\n"
            f"2. Scores for each participant (1-10) on: Argument Quality, "
            f"Evidence Strength, Logical Coherence, Persuasiveness, Responsiveness\n"
            f"3. Key moments that influenced your decision\n"
            f"4. Recommendations for improvement"
        )
