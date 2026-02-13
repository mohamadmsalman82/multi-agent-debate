"""FactChecker agent – verifies claims and rates confidence."""

from __future__ import annotations

from typing import Any

from agents.base import AgentRole, BaseAgent, DebateState
from agents.llm_provider import LLMProvider

_SYSTEM_PROMPT = """\
You are an impartial **Fact Checker** in a structured debate. Your responsibilities:
1. Evaluate factual claims made by other participants.
2. Rate your confidence in each claim (HIGH / MEDIUM / LOW / UNVERIFIABLE).
3. Provide reasoning for each rating.
4. Flag misinformation or misleading statistics.
5. Suggest corrections or more accurate data where possible.

Format your response as a list of claims followed by your assessment.
Be concise and precise.
"""


class FactChecker(BaseAgent):
    """Agent that verifies claims and rates confidence in factual accuracy."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        agent_id: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 300,
        system_prompt: str = _SYSTEM_PROMPT,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            role=AgentRole.FACT_CHECKER,
            provider=provider,
            agent_id=agent_id,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )

    def _build_turn_prompt(self, state: DebateState) -> str:
        # Collect recent claims from proposer and critic
        recent = [
            r for r in state.history[-6:]
            if r.role in (AgentRole.PROPOSER, AgentRole.CRITIC)
        ]
        if recent:
            claims_text = "\n\n".join(
                f"[{r.role.value.upper()} – Turn {r.turn_number}]: {r.content[:400]}"
                for r in recent
            )
            return (
                f"Turn {state.current_turn}/{state.max_turns}. "
                f"Review the following statements for factual accuracy:\n\n"
                f"{claims_text}\n\n"
                f"For each major factual claim, provide:\n"
                f"- The claim\n"
                f"- Confidence rating (HIGH/MEDIUM/LOW/UNVERIFIABLE)\n"
                f"- Brief justification"
            )

        return (
            f"Turn {state.current_turn}/{state.max_turns}. "
            f"No new claims to check. Summarise the factual landscape of "
            f"the debate on '{state.topic}' so far."
        )
