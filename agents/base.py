"""Base agent class with provider-agnostic interface.

Every debate participant inherits from ``BaseAgent`` which provides:
- Conversation history / memory management
- Provider-agnostic ``generate_response`` and ``process_turn``
- Automatic token tracking and structured response types
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from agents.llm_provider import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & data containers
# ---------------------------------------------------------------------------

class AgentRole(str, Enum):
    """Well-known roles in the debate system."""

    PROPOSER = "proposer"
    CRITIC = "critic"
    FACT_CHECKER = "fact_checker"
    MODERATOR = "moderator"
    JUDGE = "judge"


@dataclass
class AgentResponse:
    """Structured response emitted by an agent on each turn."""

    agent_id: str
    role: AgentRole
    content: str
    turn_number: int
    tokens_used: int
    provider: str
    model: str
    latency_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DebateState:
    """Snapshot of the debate visible to an agent when it takes a turn."""

    topic: str
    protocol: str
    current_turn: int
    max_turns: int
    history: list[AgentResponse]
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------

class BaseAgent:
    """Provider-agnostic base class for every debate agent.

    Parameters
    ----------
    role : AgentRole
        The functional role this agent plays in the debate.
    provider : LLMProvider
        The LLM backend used for generation.
    agent_id : str | None
        Unique identifier; auto-generated if not supplied.
    temperature : float
        Sampling temperature forwarded to the provider.
    max_tokens : int
        Max output tokens forwarded to the provider.
    system_prompt : str
        The root system prompt that defines agent behaviour.
    """

    role: AgentRole

    def __init__(
        self,
        *,
        role: AgentRole,
        provider: LLMProvider,
        agent_id: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        system_prompt: str = "",
    ) -> None:
        self.agent_id = agent_id or f"{role.value}_{uuid.uuid4().hex[:8]}"
        self.role = role
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

        # Internal memory
        self._conversation_history: list[dict[str, str]] = []
        self._total_tokens_used: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_response(
        self,
        context: str,
        prompt: str,
        **kwargs: Any,
    ) -> AgentResponse:
        """Send *prompt* (with *context*) to the LLM and return a structured response.

        The full message list sent to the provider is:
        ``[system_prompt, ...conversation_history, user(context + prompt)]``
        """
        messages = self._build_messages(context, prompt)
        llm_resp: LLMResponse = await self.provider.generate(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )

        # Update memory
        self._conversation_history.append({"role": "user", "content": f"{context}\n{prompt}"})
        self._conversation_history.append({"role": "assistant", "content": llm_resp.text})
        self._total_tokens_used += llm_resp.tokens_used

        return AgentResponse(
            agent_id=self.agent_id,
            role=self.role,
            content=llm_resp.text,
            turn_number=0,  # Will be set by caller
            tokens_used=llm_resp.tokens_used,
            provider=llm_resp.provider,
            model=llm_resp.model,
            latency_ms=llm_resp.latency_ms,
        )

    async def process_turn(self, state: DebateState) -> AgentResponse:
        """High-level entry point called by the orchestrator each turn.

        Subclasses *should* override ``_build_turn_prompt`` to customise
        what the agent sees each turn.
        """
        context = self._build_context(state)
        prompt = self._build_turn_prompt(state)
        response = await self.generate_response(context, prompt)
        response.turn_number = state.current_turn
        return response

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    def _build_turn_prompt(self, state: DebateState) -> str:
        """Return the user-facing prompt for this turn. Override in subclasses."""
        return (
            f"It is turn {state.current_turn}/{state.max_turns} of the debate on "
            f"'{state.topic}'. Please provide your response as {self.role.value}."
        )

    def _build_context(self, state: DebateState) -> str:
        """Build context string from debate history."""
        if not state.history:
            return f"Debate topic: {state.topic}\nProtocol: {state.protocol}"

        lines = [f"Debate topic: {state.topic}", f"Protocol: {state.protocol}", ""]
        for resp in state.history[-10:]:  # keep last 10 for context window
            lines.append(f"[Turn {resp.turn_number}] {resp.role.value}: {resp.content}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def _build_messages(self, context: str, prompt: str) -> list[dict[str, str]]:
        """Assemble the full message list for the provider."""
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        # Include recent conversation history (sliding window)
        messages.extend(self._conversation_history[-20:])
        messages.append({"role": "user", "content": f"{context}\n\n{prompt}"})
        return messages

    def clear_memory(self) -> None:
        """Reset conversation history."""
        self._conversation_history.clear()

    @property
    def total_tokens_used(self) -> int:
        return self._total_tokens_used

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(id={self.agent_id!r}, "
            f"role={self.role.value!r}, provider={self.provider})"
        )
