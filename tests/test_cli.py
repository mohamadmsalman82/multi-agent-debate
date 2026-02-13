"""Tests for the CLI interface."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from cli import cli


@pytest.fixture
def runner():
    return CliRunner()


class TestCLI:
    def test_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Multi-Agent Debate System" in result.output

    def test_debate_help(self, runner):
        result = runner.invoke(cli, ["debate", "--help"])
        assert result.exit_code == 0
        assert "--topic" in result.output
        assert "--protocol" in result.output

    def test_list_debates_empty(self, runner, tmp_path):
        """List debates on a fresh database."""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(
            f"database:\n  path: {tmp_path / 'test.db'}\n"
            "api: {}\nagents: {}\n"
        )
        result = runner.invoke(cli, ["--config", str(config_path), "list-debates"])
        assert result.exit_code == 0
        assert "No debates found" in result.output

    def test_visualize_not_found(self, runner, tmp_path):
        """Visualize a non-existent debate."""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(
            f"database:\n  path: {tmp_path / 'test.db'}\n"
            "api: {}\nagents: {}\n"
        )
        result = runner.invoke(
            cli, ["--config", str(config_path), "visualize", "--debate-id", "999"]
        )
        assert result.exit_code == 0
        assert "not found" in result.output

    def test_run_experiment_help(self, runner):
        result = runner.invoke(cli, ["run-experiment", "--help"])
        assert result.exit_code == 0
        assert "--benchmark-config" in result.output
