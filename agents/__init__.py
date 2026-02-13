"""Multi-agent debate system - Agent definitions."""

from agents.base import BaseAgent
from agents.proposer import Proposer
from agents.critic import Critic
from agents.fact_checker import FactChecker
from agents.moderator import Moderator
from agents.judge import Judge
from agents.llm_provider import (
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    CohereProvider,
    OpenRouterProvider,
    create_provider,
)

__all__ = [
    "BaseAgent",
    "Proposer",
    "Critic",
    "FactChecker",
    "Moderator",
    "Judge",
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "CohereProvider",
    "OpenRouterProvider",
    "create_provider",
]
