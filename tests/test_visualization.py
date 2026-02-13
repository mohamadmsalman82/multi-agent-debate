"""Tests for the visualization module."""

from __future__ import annotations

import pytest

from evaluation.metrics import compute_all_metrics
from viz.visualize import DebateVisualizer


class TestDebateVisualizer:
    @pytest.fixture
    def viz(self, tmp_path):
        return DebateVisualizer(output_dir=tmp_path / "viz_output")

    def test_plot_participation(self, viz, sample_history):
        path = viz.plot_participation(1, sample_history)
        assert path.exists()
        assert path.suffix == ".png"

    def test_plot_token_usage(self, viz, sample_history):
        path = viz.plot_token_usage(1, sample_history)
        assert path.exists()

    def test_plot_quality_radar(self, viz, sample_history):
        metrics = compute_all_metrics(sample_history, topic="AI", max_turns=4)
        path = viz.plot_quality_radar(1, metrics)
        assert path.exists()

    def test_plot_response_lengths(self, viz, sample_history):
        path = viz.plot_response_lengths(1, sample_history)
        assert path.exists()

    def test_export_transcript(self, viz, sample_history):
        path = viz.export_transcript(1, sample_history, topic="AI regulation")
        assert path.exists()
        content = path.read_text()
        assert "DEBATE #1 TRANSCRIPT" in content
        assert "PROPOSER" in content
        assert "CRITIC" in content

    def test_generate_all(self, viz, sample_history):
        metrics = compute_all_metrics(sample_history, topic="AI", max_turns=4)
        paths = viz.generate_all(1, sample_history, metrics, topic="AI")
        assert len(paths) == 5
        for p in paths:
            assert p.exists()
