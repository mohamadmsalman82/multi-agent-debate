"""Orchestration layer – debate management and protocols."""

from orchestration.debate_manager import DebateManager, DebateResult
from orchestration.protocols import (
    DebateProtocol,
    RoundRobinProtocol,
    AdversarialProtocol,
    CollaborativeProtocol,
    create_protocol,
)

__all__ = [
    "DebateManager",
    "DebateResult",
    "DebateProtocol",
    "RoundRobinProtocol",
    "AdversarialProtocol",
    "CollaborativeProtocol",
    "create_protocol",
]
