"""Validators for debate responses and configurations."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from agents.base import AgentResponse, DebateState

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Outcome of a validation check."""

    valid: bool
    issues: list[str]

    def __bool__(self) -> bool:
        return self.valid


class DebateValidator:
    """Validates debate responses against configurable rules."""

    def __init__(
        self,
        min_response_length: int = 20,
        max_response_length: int = 5000,
        require_unique_content: bool = True,
    ) -> None:
        self.min_response_length = min_response_length
        self.max_response_length = max_response_length
        self.require_unique_content = require_unique_content

    def validate_response(
        self,
        response: AgentResponse,
        state: DebateState,
    ) -> ValidationResult:
        """Check a single response against debate rules."""
        issues: list[str] = []

        # Length checks
        content_len = len(response.content.strip())
        if content_len < self.min_response_length:
            issues.append(
                f"Response too short ({content_len} chars, "
                f"minimum {self.min_response_length})"
            )
        if content_len > self.max_response_length:
            issues.append(
                f"Response too long ({content_len} chars, "
                f"maximum {self.max_response_length})"
            )

        # Empty content
        if not response.content.strip():
            issues.append("Response is empty")

        # Duplicate detection
        if self.require_unique_content:
            for prev in state.history:
                if (
                    prev.content.strip() == response.content.strip()
                    and prev.agent_id != response.agent_id
                ):
                    issues.append(
                        f"Duplicate of {prev.role.value}'s response from turn {prev.turn_number}"
                    )
                    break

        if issues:
            logger.warning(
                "Validation failed for %s (turn %d): %s",
                response.role.value,
                response.turn_number,
                "; ".join(issues),
            )

        return ValidationResult(valid=len(issues) == 0, issues=issues)

    def validate_debate_config(
        self,
        agents_count: int,
        max_turns: int,
        protocol: str,
    ) -> ValidationResult:
        """Validate debate configuration before starting."""
        issues: list[str] = []

        if agents_count < 2:
            issues.append("At least 2 agents are required for a debate")
        if max_turns < 1:
            issues.append("max_turns must be at least 1")
        if max_turns > 50:
            issues.append("max_turns exceeds reasonable limit (50)")

        valid_protocols = {"round_robin", "adversarial", "collaborative"}
        if protocol not in valid_protocols:
            issues.append(f"Unknown protocol '{protocol}'. Valid: {valid_protocols}")

        return ValidationResult(valid=len(issues) == 0, issues=issues)
