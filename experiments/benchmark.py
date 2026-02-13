"""Benchmark runner for systematic debate experiments.

Reads experiment configs and runs multiple debates with varying parameters,
collecting aggregated metrics for comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agents.base import BaseAgent
from data.database import DebateDatabase
from evaluation.metrics import DebateMetrics, compute_all_metrics
from orchestration.debate_manager import DebateManager, DebateResult

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Aggregated results from a benchmark run."""

    name: str
    debates: list[DebateResult] = field(default_factory=list)
    metrics_per_debate: list[DebateMetrics] = field(default_factory=list)
    aggregate_metrics: dict[str, float] = field(default_factory=dict)

    def summarise(self) -> dict[str, Any]:
        if not self.metrics_per_debate:
            return {"name": self.name, "debates_run": 0}

        n = len(self.metrics_per_debate)
        return {
            "name": self.name,
            "debates_run": n,
            "avg_coherence": round(sum(m.coherence for m in self.metrics_per_debate) / n, 3),
            "avg_evidence": round(sum(m.evidence_strength for m in self.metrics_per_debate) / n, 3),
            "avg_relevance": round(sum(m.relevance for m in self.metrics_per_debate) / n, 3),
            "avg_diversity": round(sum(m.argument_diversity for m in self.metrics_per_debate) / n, 3),
            "avg_consensus": round(sum(m.consensus_level for m in self.metrics_per_debate) / n, 3),
            "total_tokens": sum(m.total_tokens for m in self.metrics_per_debate),
        }


class BenchmarkRunner:
    """Run a suite of debates defined in an experiment YAML config.

    Parameters
    ----------
    agents : list[BaseAgent]
        The pool of agents to use in each debate.
    db : DebateDatabase | None
        Optional persistence layer.
    """

    def __init__(
        self,
        agents: list[BaseAgent],
        db: DebateDatabase | None = None,
    ) -> None:
        self.agents = agents
        self.db = db

    async def run_from_config(self, config_path: str | Path) -> list[BenchmarkResult]:
        """Parse a YAML experiment config and execute all benchmarks."""
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        benchmarks = cfg.get("benchmarks", [])
        results: list[BenchmarkResult] = []

        for bench_cfg in benchmarks:
            result = await self._run_benchmark(bench_cfg)
            results.append(result)

        return results

    async def _run_benchmark(self, cfg: dict[str, Any]) -> BenchmarkResult:
        """Execute a single benchmark block."""
        name = cfg.get("name", "unnamed")
        topics = cfg.get("topics", [])
        protocols = cfg.get("protocols", ["round_robin"])
        max_turns = cfg.get("max_turns", 6)
        repetitions = cfg.get("repetitions", 1)

        result = BenchmarkResult(name=name)
        total_runs = len(topics) * len(protocols) * repetitions

        logger.info(
            "Benchmark '%s': %d topics x %d protocols x %d reps = %d runs",
            name, len(topics), len(protocols), repetitions, total_runs,
        )

        run_idx = 0
        for topic in topics:
            for protocol in protocols:
                for rep in range(repetitions):
                    run_idx += 1
                    logger.info(
                        "[%d/%d] topic=%r, protocol=%s, rep=%d",
                        run_idx, total_runs, topic, protocol, rep + 1,
                    )
                    # Clear agent memory between runs
                    for agent in self.agents:
                        agent.clear_memory()

                    manager = DebateManager(
                        agents=self.agents,
                        protocol=protocol,
                        db=self.db,
                    )

                    try:
                        debate_result = await manager.run_debate(
                            topic=topic, max_turns=max_turns
                        )
                        metrics = compute_all_metrics(
                            debate_result.history,
                            topic=topic,
                            max_turns=max_turns,
                        )
                        result.debates.append(debate_result)
                        result.metrics_per_debate.append(metrics)
                    except Exception:
                        logger.exception(
                            "Benchmark run failed: topic=%r, protocol=%s", topic, protocol
                        )

        # Aggregate
        result.aggregate_metrics = result.summarise()
        return result
