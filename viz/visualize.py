"""Debate visualization – charts and formatted text reports.

Generates matplotlib charts for metrics, participation, token usage,
and exports a pretty-printed text transcript.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from agents.base import AgentResponse
from evaluation.metrics import DebateMetrics

logger = logging.getLogger(__name__)

# Colour palette per role
_ROLE_COLOURS: dict[str, str] = {
    "proposer": "#4CAF50",
    "critic": "#F44336",
    "fact_checker": "#2196F3",
    "moderator": "#FF9800",
    "judge": "#9C27B0",
}


class DebateVisualizer:
    """Generate charts and reports from debate data."""

    def __init__(self, output_dir: str | Path = "viz/output") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_all(
        self,
        debate_id: int,
        history: list[AgentResponse],
        metrics: DebateMetrics,
        topic: str = "",
    ) -> list[Path]:
        """Generate all charts and the text transcript. Returns file paths."""
        paths: list[Path] = []
        paths.append(self.plot_participation(debate_id, history))
        paths.append(self.plot_token_usage(debate_id, history))
        paths.append(self.plot_quality_radar(debate_id, metrics))
        paths.append(self.plot_response_lengths(debate_id, history))
        paths.append(self.export_transcript(debate_id, history, topic))
        return paths

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------

    def plot_participation(
        self, debate_id: int, history: list[AgentResponse]
    ) -> Path:
        """Bar chart of messages per agent role."""
        counts: dict[str, int] = {}
        for r in history:
            counts[r.role.value] = counts.get(r.role.value, 0) + 1

        fig, ax = plt.subplots(figsize=(8, 4))
        roles = list(counts.keys())
        values = list(counts.values())
        colours = [_ROLE_COLOURS.get(r, "#607D8B") for r in roles]

        ax.barh(roles, values, color=colours)
        ax.set_xlabel("Messages")
        ax.set_title(f"Debate #{debate_id} – Agent Participation")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.tight_layout()

        path = self.output_dir / f"debate_{debate_id}_participation.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", path)
        return path

    def plot_token_usage(
        self, debate_id: int, history: list[AgentResponse]
    ) -> Path:
        """Stacked bar chart of token usage per turn."""
        turns: dict[int, dict[str, int]] = {}
        for r in history:
            turns.setdefault(r.turn_number, {})
            turns[r.turn_number][r.role.value] = (
                turns[r.turn_number].get(r.role.value, 0) + r.tokens_used
            )

        fig, ax = plt.subplots(figsize=(10, 5))
        turn_nums = sorted(turns)
        roles_seen = sorted({role for t in turns.values() for role in t})
        bottom = [0] * len(turn_nums)

        for role in roles_seen:
            values = [turns[t].get(role, 0) for t in turn_nums]
            ax.bar(
                turn_nums,
                values,
                bottom=bottom,
                label=role,
                color=_ROLE_COLOURS.get(role, "#607D8B"),
            )
            bottom = [b + v for b, v in zip(bottom, values)]

        ax.set_xlabel("Turn")
        ax.set_ylabel("Tokens")
        ax.set_title(f"Debate #{debate_id} – Token Usage per Turn")
        ax.legend()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.tight_layout()

        path = self.output_dir / f"debate_{debate_id}_tokens.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", path)
        return path

    def plot_quality_radar(
        self, debate_id: int, metrics: DebateMetrics
    ) -> Path:
        """Radar (spider) chart of quality metrics."""
        import numpy as np

        categories = ["Coherence", "Evidence", "Relevance", "Diversity", "Consensus"]
        values = [
            metrics.coherence,
            metrics.evidence_strength,
            metrics.relevance,
            metrics.argument_diversity,
            metrics.consensus_level,
        ]
        values += values[:1]  # close the polygon

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
        ax.plot(angles, values, "o-", linewidth=2, color="#1976D2")
        ax.fill(angles, values, alpha=0.25, color="#1976D2")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title(f"Debate #{debate_id} – Quality Metrics", y=1.08)
        plt.tight_layout()

        path = self.output_dir / f"debate_{debate_id}_quality.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", path)
        return path

    def plot_response_lengths(
        self, debate_id: int, history: list[AgentResponse]
    ) -> Path:
        """Line chart of response lengths over time."""
        fig, ax = plt.subplots(figsize=(10, 4))

        # Group by role
        role_data: dict[str, tuple[list[int], list[int]]] = {}
        for r in history:
            role_data.setdefault(r.role.value, ([], []))
            role_data[r.role.value][0].append(r.turn_number)
            role_data[r.role.value][1].append(len(r.content))

        for role, (turns, lengths) in role_data.items():
            ax.plot(
                turns,
                lengths,
                "o-",
                label=role,
                color=_ROLE_COLOURS.get(role, "#607D8B"),
                markersize=5,
            )

        ax.set_xlabel("Turn")
        ax.set_ylabel("Response Length (chars)")
        ax.set_title(f"Debate #{debate_id} – Response Lengths")
        ax.legend()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.tight_layout()

        path = self.output_dir / f"debate_{debate_id}_lengths.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", path)
        return path

    # ------------------------------------------------------------------
    # Text transcript
    # ------------------------------------------------------------------

    def export_transcript(
        self,
        debate_id: int,
        history: list[AgentResponse],
        topic: str = "",
    ) -> Path:
        """Export a pretty-printed text transcript."""
        lines = [
            f"{'=' * 72}",
            f"  DEBATE #{debate_id} TRANSCRIPT",
            f"  Topic: {topic}",
            f"{'=' * 72}",
            "",
        ]
        current_turn = -1
        for r in history:
            if r.turn_number != current_turn:
                current_turn = r.turn_number
                lines.append(f"--- Turn {current_turn} {'─' * 50}")
            lines.append(f"  [{r.role.value.upper()}] ({r.provider}/{r.model})")
            lines.append(f"  Tokens: {r.tokens_used} | Latency: {r.latency_ms:.0f} ms")
            lines.append("")
            for paragraph in r.content.split("\n"):
                lines.append(f"    {paragraph}")
            lines.append("")

        lines.append(f"{'=' * 72}")
        lines.append(f"  END OF TRANSCRIPT")
        lines.append(f"{'=' * 72}")

        path = self.output_dir / f"debate_{debate_id}_transcript.txt"
        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Saved %s", path)
        return path
