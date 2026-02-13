"""Async SQLite database layer using aiosqlite.

Handles schema creation, CRUD for debates / messages / evaluations,
and helper queries used by the CLI and visualisation modules.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import aiosqlite

from data.models import (
    AgentConfigRecord,
    DebateRecord,
    EvaluationRecord,
    MessageRecord,
)

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS debates (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    topic       TEXT    NOT NULL,
    protocol    TEXT    NOT NULL,
    created_at  TEXT    NOT NULL,
    status      TEXT    NOT NULL DEFAULT 'pending',
    metadata    TEXT    NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    debate_id   INTEGER NOT NULL REFERENCES debates(id),
    agent_id    TEXT    NOT NULL,
    role        TEXT    NOT NULL,
    content     TEXT    NOT NULL,
    turn_number INTEGER NOT NULL,
    tokens_used INTEGER NOT NULL DEFAULT 0,
    provider    TEXT    NOT NULL DEFAULT '',
    model       TEXT    NOT NULL DEFAULT '',
    latency_ms  REAL    NOT NULL DEFAULT 0.0,
    created_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    debate_id    INTEGER NOT NULL REFERENCES debates(id),
    metric_name  TEXT    NOT NULL,
    metric_value REAL    NOT NULL,
    details      TEXT    NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS agent_configs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    debate_id   INTEGER NOT NULL REFERENCES debates(id),
    agent_id    TEXT    NOT NULL,
    provider    TEXT    NOT NULL,
    model       TEXT    NOT NULL,
    config      TEXT    NOT NULL DEFAULT '{}'
);
"""


class DebateDatabase:
    """Async wrapper around an SQLite database for debate persistence."""

    def __init__(self, db_path: str | Path = "data/debates.db") -> None:
        self.db_path = Path(db_path)
        self._conn: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open connection and ensure schema exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self.db_path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(_SCHEMA)
        await self._conn.commit()
        logger.info("Database connected: %s", self.db_path)

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._conn

    # ------------------------------------------------------------------
    # Debates
    # ------------------------------------------------------------------

    async def create_debate(self, record: DebateRecord) -> int:
        """Insert a new debate and return its id."""
        cur = await self.conn.execute(
            "INSERT INTO debates (topic, protocol, created_at, status, metadata) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                record.topic,
                record.protocol,
                record.created_at.isoformat(),
                record.status,
                record.metadata_json,
            ),
        )
        await self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    async def update_debate_status(self, debate_id: int, status: str) -> None:
        await self.conn.execute(
            "UPDATE debates SET status = ? WHERE id = ?", (status, debate_id)
        )
        await self.conn.commit()

    async def get_debate(self, debate_id: int) -> DebateRecord | None:
        cur = await self.conn.execute("SELECT * FROM debates WHERE id = ?", (debate_id,))
        row = await cur.fetchone()
        if row is None:
            return None
        return DebateRecord(
            id=row["id"],
            topic=row["topic"],
            protocol=row["protocol"],
            created_at=row["created_at"],
            status=row["status"],
            metadata_json=row["metadata"],
        )

    async def list_debates(self, limit: int = 20) -> list[DebateRecord]:
        cur = await self.conn.execute(
            "SELECT * FROM debates ORDER BY id DESC LIMIT ?", (limit,)
        )
        rows = await cur.fetchall()
        return [
            DebateRecord(
                id=r["id"],
                topic=r["topic"],
                protocol=r["protocol"],
                created_at=r["created_at"],
                status=r["status"],
                metadata_json=r["metadata"],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    async def save_message(self, record: MessageRecord) -> int:
        cur = await self.conn.execute(
            "INSERT INTO messages "
            "(debate_id, agent_id, role, content, turn_number, tokens_used, "
            " provider, model, latency_ms, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.debate_id,
                record.agent_id,
                record.role,
                record.content,
                record.turn_number,
                record.tokens_used,
                record.provider,
                record.model,
                record.latency_ms,
                record.created_at.isoformat(),
            ),
        )
        await self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    async def get_messages(self, debate_id: int) -> list[MessageRecord]:
        cur = await self.conn.execute(
            "SELECT * FROM messages WHERE debate_id = ? ORDER BY turn_number, id",
            (debate_id,),
        )
        rows = await cur.fetchall()
        return [
            MessageRecord(
                id=r["id"],
                debate_id=r["debate_id"],
                agent_id=r["agent_id"],
                role=r["role"],
                content=r["content"],
                turn_number=r["turn_number"],
                tokens_used=r["tokens_used"],
                provider=r["provider"],
                model=r["model"],
                latency_ms=r["latency_ms"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Evaluations
    # ------------------------------------------------------------------

    async def save_evaluation(self, record: EvaluationRecord) -> int:
        cur = await self.conn.execute(
            "INSERT INTO evaluations (debate_id, metric_name, metric_value, details) "
            "VALUES (?, ?, ?, ?)",
            (record.debate_id, record.metric_name, record.metric_value, record.details_json),
        )
        await self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    async def get_evaluations(self, debate_id: int) -> list[EvaluationRecord]:
        cur = await self.conn.execute(
            "SELECT * FROM evaluations WHERE debate_id = ? ORDER BY id", (debate_id,)
        )
        rows = await cur.fetchall()
        return [
            EvaluationRecord(
                id=r["id"],
                debate_id=r["debate_id"],
                metric_name=r["metric_name"],
                metric_value=r["metric_value"],
                details_json=r["details"],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Agent configs
    # ------------------------------------------------------------------

    async def save_agent_config(self, record: AgentConfigRecord) -> int:
        cur = await self.conn.execute(
            "INSERT INTO agent_configs (debate_id, agent_id, provider, model, config) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                record.debate_id,
                record.agent_id,
                record.provider,
                record.model,
                record.config_json,
            ),
        )
        await self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    async def get_agent_configs(self, debate_id: int) -> list[AgentConfigRecord]:
        cur = await self.conn.execute(
            "SELECT * FROM agent_configs WHERE debate_id = ? ORDER BY id", (debate_id,)
        )
        rows = await cur.fetchall()
        return [
            AgentConfigRecord(
                id=r["id"],
                debate_id=r["debate_id"],
                agent_id=r["agent_id"],
                provider=r["provider"],
                model=r["model"],
                config_json=r["config"],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Aggregate helpers
    # ------------------------------------------------------------------

    async def get_debate_stats(self, debate_id: int) -> dict[str, Any]:
        """Return aggregate statistics for a given debate."""
        msg_cur = await self.conn.execute(
            "SELECT COUNT(*) as cnt, SUM(tokens_used) as total_tokens, "
            "AVG(latency_ms) as avg_latency FROM messages WHERE debate_id = ?",
            (debate_id,),
        )
        msg_row = await msg_cur.fetchone()

        role_cur = await self.conn.execute(
            "SELECT role, COUNT(*) as cnt FROM messages "
            "WHERE debate_id = ? GROUP BY role",
            (debate_id,),
        )
        role_rows = await role_cur.fetchall()

        return {
            "total_messages": msg_row["cnt"] if msg_row else 0,
            "total_tokens": msg_row["total_tokens"] or 0 if msg_row else 0,
            "avg_latency_ms": round(msg_row["avg_latency"] or 0, 1) if msg_row else 0,
            "participation": {r["role"]: r["cnt"] for r in role_rows},
        }
