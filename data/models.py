"""Pydantic models mirroring the SQLite schema."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DebateRecord(BaseModel):
    """Row in the ``debates`` table."""

    id: int | None = None
    topic: str
    protocol: str
    created_at: datetime = Field(default_factory=_utcnow)
    status: str = "pending"  # pending | running | completed | failed
    metadata_json: str = "{}"

    @property
    def metadata(self) -> dict[str, Any]:
        return json.loads(self.metadata_json)

    @metadata.setter
    def metadata(self, value: dict[str, Any]) -> None:
        self.metadata_json = json.dumps(value)


class MessageRecord(BaseModel):
    """Row in the ``messages`` table."""

    id: int | None = None
    debate_id: int
    agent_id: str
    role: str
    content: str
    turn_number: int
    tokens_used: int = 0
    provider: str = ""
    model: str = ""
    latency_ms: float = 0.0
    created_at: datetime = Field(default_factory=_utcnow)


class EvaluationRecord(BaseModel):
    """Row in the ``evaluations`` table."""

    id: int | None = None
    debate_id: int
    metric_name: str
    metric_value: float
    details_json: str = "{}"

    @property
    def details(self) -> dict[str, Any]:
        return json.loads(self.details_json)


class AgentConfigRecord(BaseModel):
    """Row in the ``agent_configs`` table."""

    id: int | None = None
    debate_id: int
    agent_id: str
    provider: str
    model: str
    config_json: str = "{}"

    @property
    def config(self) -> dict[str, Any]:
        return json.loads(self.config_json)
