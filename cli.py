#!/usr/bin/env python3
"""Command-line interface for the multi-agent debate system.

Usage examples:
    python cli.py debate --topic "AI safety" --protocol round_robin
    python cli.py debate --topic "UBI policy" --protocol adversarial --agents proposer,critic,judge --max-turns 6
    python cli.py run-experiment --config config/experiment.yaml
    python cli.py visualize --debate-id 1
    python cli.py list-debates
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import click
import yaml

from agents import Proposer, Critic, FactChecker, Moderator, Judge, create_provider
from agents.base import AgentResponse, BaseAgent
from data.database import DebateDatabase
from evaluation.metrics import compute_all_metrics
from orchestration.debate_manager import DebateManager
from viz.visualize import DebateVisualizer


# ---------------------------------------------------------------------------
# Live debate display
# ---------------------------------------------------------------------------

# Role labels and ANSI colour codes for terminal output
_ROLE_STYLES: dict[str, tuple[str, str]] = {
    # role -> (label, ANSI colour code)
    "proposer":     ("PROPOSER",      "\033[1;34m"),   # bold blue
    "critic":       ("CRITIC",        "\033[1;31m"),   # bold red
    "fact_checker": ("FACT CHECKER",  "\033[1;33m"),   # bold yellow
    "moderator":    ("MODERATOR",     "\033[1;35m"),   # bold magenta
    "judge":        ("JUDGE",         "\033[1;32m"),   # bold green
}
_RESET = "\033[0m"
_DIM = "\033[2m"


def _print_response(turn: int, resp: AgentResponse) -> None:
    """Pretty-print a single agent response to the terminal."""
    role_name = resp.role.value
    label, colour = _ROLE_STYLES.get(role_name, (role_name.upper(), "\033[1m"))

    # Header bar
    click.echo(f"\n{colour}{'─' * 60}")
    click.echo(f"  [{label}]  Turn {turn}  •  {resp.model}")
    click.echo(f"{'─' * 60}{_RESET}")

    # Body – wrap long lines for readability
    content = resp.content.strip()
    for paragraph in content.split("\n"):
        # Indent each paragraph for visual separation
        click.echo(f"  {paragraph}")

    # Footer – tokens and latency
    click.echo(
        f"{_DIM}  [{resp.tokens_used} tokens  •  {resp.latency_ms:.0f} ms]{_RESET}"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: str = "config/default.yaml") -> dict[str, Any]:
    """Load and return the YAML config."""
    p = Path(config_path)
    if not p.exists():
        click.echo(f"Config not found: {p}. Using defaults.", err=True)
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


_AGENT_CLASSES: dict[str, type[BaseAgent]] = {
    "proposer": Proposer,
    "critic": Critic,
    "fact_checker": FactChecker,
    "moderator": Moderator,
    "judge": Judge,
}


def _build_agents(
    agent_names: list[str],
    cfg: dict[str, Any],
) -> list[BaseAgent]:
    """Instantiate agents based on config and requested names."""
    api_cfg = cfg.get("api", {})
    agent_cfgs = cfg.get("agents", {})
    timeout = api_cfg.get("timeout", 30)
    max_retries = api_cfg.get("max_retries", 3)

    agents: list[BaseAgent] = []
    for name in agent_names:
        cls = _AGENT_CLASSES.get(name)
        if cls is None:
            raise click.BadParameter(
                f"Unknown agent '{name}'. Choose from: {list(_AGENT_CLASSES)}"
            )

        acfg = agent_cfgs.get(name, {})
        provider_name = acfg.get("provider", "openai")
        provider_api_cfg = api_cfg.get(provider_name, {})

        # Model resolution order:
        #   1. Agent-level model (agents.proposer.model)
        #   2. Provider-level model (api.anthropic.model)
        #   3. Provider class default
        model = acfg.get("model") or provider_api_cfg.get("model")

        provider_kwargs: dict[str, Any] = {
            "api_key_env": provider_api_cfg.get("api_key_env"),
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if model:
            provider_kwargs["model"] = model

        provider = create_provider(provider_name, **provider_kwargs)

        agent = cls(
            provider=provider,
            temperature=acfg.get("temperature", 0.7),
            max_tokens=acfg.get("max_tokens", 500),
        )
        agents.append(agent)

    return agents


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
@click.option("--config", default="config/default.yaml", help="Path to YAML config")
@click.option("--log-level", default="INFO", help="Logging level")
@click.pass_context
def cli(ctx: click.Context, config: str, log_level: str) -> None:
    """Multi-Agent Debate System – run LLM debates from the command line."""
    _setup_logging(log_level)
    ctx.ensure_object(dict)
    ctx.obj["config"] = _load_config(config)
    ctx.obj["config_path"] = config


# ---- debate ---------------------------------------------------------------

@cli.command()
@click.option("--topic", required=True, help="Debate topic")
@click.option(
    "--protocol",
    type=click.Choice(["round_robin", "adversarial", "collaborative"]),
    default="round_robin",
    help="Turn-taking protocol",
)
@click.option(
    "--agents",
    default="proposer,critic,judge",
    help="Comma-separated agent names",
)
@click.option("--max-turns", default=6, type=int, help="Maximum debate turns")
@click.option("--no-db", is_flag=True, help="Skip database persistence")
@click.pass_context
def debate(
    ctx: click.Context,
    topic: str,
    protocol: str,
    agents: str,
    max_turns: int,
    no_db: bool,
) -> None:
    """Run a structured debate on a given topic."""
    cfg = ctx.obj["config"]
    agent_names = [a.strip() for a in agents.split(",")]

    # Print debate header
    click.echo(f"\n\033[1m{'=' * 60}")
    click.echo(f"  DEBATE: {topic}")
    click.echo(f"{'=' * 60}\033[0m")
    click.echo(f"  Protocol : {protocol}")
    click.echo(f"  Agents   : {', '.join(agent_names)}")
    click.echo(f"  Max turns: {max_turns}")

    agent_objs = _build_agents(agent_names, cfg)

    async def _run() -> None:
        db: DebateDatabase | None = None
        if not no_db:
            db_path = cfg.get("database", {}).get("path", "data/debates.db")
            db = DebateDatabase(db_path)
            await db.connect()

        try:
            manager = DebateManager(agents=agent_objs, protocol=protocol, db=db)
            result = await manager.run_debate(
                topic=topic,
                max_turns=max_turns,
                on_response=_print_response,
            )

            # Compute and display metrics
            metrics = compute_all_metrics(result.history, topic=topic, max_turns=max_turns)

            click.echo(f"\n{'=' * 60}")
            click.echo(f"  DEBATE COMPLETE – #{result.debate_id}")
            click.echo(f"{'=' * 60}")
            click.echo(f"  Status    : {result.status}")
            click.echo(f"  Messages  : {len(result.history)}")
            click.echo(f"  Tokens    : {metrics.total_tokens}")
            click.echo()

            click.echo("  Quality Metrics:")
            for k, v in metrics.to_dict()["quality"].items():
                click.echo(f"    {k:25s}: {v:.3f}")

            click.echo()
            click.echo("  Participation:")
            for role, count in metrics.agent_participation.items():
                click.echo(f"    {role:25s}: {count} messages")

            if result.summary:
                click.echo(f"\n  Judge's Summary:\n")
                for line in result.summary.split("\n"):
                    click.echo(f"    {line}")

            # Generate visualizations
            viz = DebateVisualizer()
            paths = viz.generate_all(
                result.debate_id, result.history, metrics, topic=topic
            )
            click.echo(f"\n  Outputs saved to: {viz.output_dir}/")
            for p in paths:
                click.echo(f"    - {p.name}")

        finally:
            if db:
                await db.close()

    asyncio.run(_run())


# ---- run-experiment -------------------------------------------------------

@cli.command("run-experiment")
@click.option(
    "--benchmark-config",
    "bench_config",
    default="config/experiment.yaml",
    help="Experiment config YAML",
)
@click.pass_context
def run_experiment(ctx: click.Context, bench_config: str) -> None:
    """Run a benchmark experiment from a YAML config."""
    cfg = ctx.obj["config"]
    all_agent_names = list(_AGENT_CLASSES.keys())
    agent_objs = _build_agents(all_agent_names, cfg)

    async def _run() -> None:
        db_path = cfg.get("database", {}).get("path", "data/debates.db")
        db = DebateDatabase(db_path)
        await db.connect()

        try:
            from experiments.benchmark import BenchmarkRunner

            runner = BenchmarkRunner(agents=agent_objs, db=db)
            results = await runner.run_from_config(bench_config)

            click.echo(f"\n{'=' * 60}")
            click.echo("  EXPERIMENT RESULTS")
            click.echo(f"{'=' * 60}")
            for r in results:
                click.echo(f"\n  Benchmark: {r.name}")
                summary = r.summarise()
                for k, v in summary.items():
                    click.echo(f"    {k:25s}: {v}")
        finally:
            await db.close()

    asyncio.run(_run())


# ---- visualize ------------------------------------------------------------

@cli.command()
@click.option("--debate-id", required=True, type=int, help="Debate ID to visualize")
@click.pass_context
def visualize(ctx: click.Context, debate_id: int) -> None:
    """Generate visualizations for a completed debate."""
    cfg = ctx.obj["config"]

    async def _run() -> None:
        db_path = cfg.get("database", {}).get("path", "data/debates.db")
        db = DebateDatabase(db_path)
        await db.connect()

        try:
            debate_rec = await db.get_debate(debate_id)
            if debate_rec is None:
                click.echo(f"Debate #{debate_id} not found.", err=True)
                return

            messages = await db.get_messages(debate_id)
            if not messages:
                click.echo(f"No messages found for debate #{debate_id}.", err=True)
                return

            # Convert MessageRecords to AgentResponses for viz
            from agents.base import AgentResponse, AgentRole

            history: list[AgentResponse] = []
            for m in messages:
                history.append(
                    AgentResponse(
                        agent_id=m.agent_id,
                        role=AgentRole(m.role),
                        content=m.content,
                        turn_number=m.turn_number,
                        tokens_used=m.tokens_used,
                        provider=m.provider,
                        model=m.model,
                        latency_ms=m.latency_ms,
                    )
                )

            metrics = compute_all_metrics(
                history, topic=debate_rec.topic
            )
            viz = DebateVisualizer()
            paths = viz.generate_all(
                debate_id, history, metrics, topic=debate_rec.topic
            )

            click.echo(f"Generated {len(paths)} files in {viz.output_dir}/:")
            for p in paths:
                click.echo(f"  - {p.name}")
        finally:
            await db.close()

    asyncio.run(_run())


# ---- list-debates ---------------------------------------------------------

@cli.command("list-debates")
@click.option("--limit", default=20, type=int, help="Number of debates to list")
@click.pass_context
def list_debates(ctx: click.Context, limit: int) -> None:
    """List recent debates stored in the database."""
    cfg = ctx.obj["config"]

    async def _run() -> None:
        db_path = cfg.get("database", {}).get("path", "data/debates.db")
        db = DebateDatabase(db_path)
        await db.connect()

        try:
            debates = await db.list_debates(limit=limit)
            if not debates:
                click.echo("No debates found.")
                return

            click.echo(f"{'ID':>5}  {'Status':<12} {'Protocol':<15} {'Topic'}")
            click.echo(f"{'─' * 5}  {'─' * 12} {'─' * 15} {'─' * 40}")
            for d in debates:
                click.echo(
                    f"{d.id:>5}  {d.status:<12} {d.protocol:<15} {d.topic[:40]}"
                )
        finally:
            await db.close()

    asyncio.run(_run())


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
