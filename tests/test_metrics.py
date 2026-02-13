"""Tests for evaluation metrics."""

from __future__ import annotations

import pytest

from agents.base import AgentResponse, AgentRole
from evaluation.metrics import (
    DebateMetrics,
    argument_diversity_score,
    coherence_score,
    compute_all_metrics,
    evidence_strength_score,
    relevance_score,
)


class TestCoherenceScore:
    def test_empty_history(self):
        assert coherence_score([]) == 0.0

    def test_single_message(self, sample_history):
        score = coherence_score(sample_history[:1])
        assert 0.0 <= score <= 1.0

    def test_consistent_lengths_score_higher(self):
        uniform = [
            AgentResponse(
                agent_id="a", role=AgentRole.PROPOSER, content="x" * 100,
                turn_number=i, tokens_used=50, provider="p", model="m", latency_ms=0,
            )
            for i in range(5)
        ]
        varied = [
            AgentResponse(
                agent_id="a", role=AgentRole.PROPOSER, content="x" * (50 * (i + 1)),
                turn_number=i, tokens_used=50, provider="p", model="m", latency_ms=0,
            )
            for i in range(5)
        ]
        assert coherence_score(uniform) >= coherence_score(varied)

    def test_structured_content_bonus(self):
        structured = [
            AgentResponse(
                agent_id="a", role=AgentRole.PROPOSER,
                content="Point one.\n\nPoint two.\n- Detail.\n1. Numbered.",
                turn_number=1, tokens_used=50, provider="p", model="m", latency_ms=0,
            )
        ]
        flat = [
            AgentResponse(
                agent_id="a", role=AgentRole.PROPOSER,
                content="Point one. Point two. Detail. Numbered.",
                turn_number=1, tokens_used=50, provider="p", model="m", latency_ms=0,
            )
        ]
        assert coherence_score(structured) >= coherence_score(flat)


class TestEvidenceStrengthScore:
    def test_empty(self):
        assert evidence_strength_score([]) == 0.0

    def test_evidence_rich_content(self, sample_history):
        score = evidence_strength_score(sample_history)
        assert score > 0.0

    def test_evidence_keywords_boost(self):
        rich = [
            AgentResponse(
                agent_id="a", role=AgentRole.PROPOSER,
                content="According to research, studies show 80% of data from surveys and analysis.",
                turn_number=1, tokens_used=50, provider="p", model="m", latency_ms=0,
            )
        ]
        poor = [
            AgentResponse(
                agent_id="a", role=AgentRole.PROPOSER,
                content="I think this is true because it makes sense to me personally.",
                turn_number=1, tokens_used=50, provider="p", model="m", latency_ms=0,
            )
        ]
        assert evidence_strength_score(rich) > evidence_strength_score(poor)


class TestRelevanceScore:
    def test_no_topic(self, sample_history):
        assert relevance_score(sample_history) == 0.5

    def test_on_topic(self, sample_history):
        score = relevance_score(sample_history, topic="AI regulation safety")
        assert score > 0.0

    def test_empty(self):
        assert relevance_score([], "topic") == 0.5


class TestArgumentDiversity:
    def test_empty(self):
        assert argument_diversity_score([]) == 0.0

    def test_repetitive_lower_than_diverse(self):
        repetitive = [
            AgentResponse(
                agent_id="a", role=AgentRole.PROPOSER,
                content="word word word word word word",
                turn_number=1, tokens_used=10, provider="p", model="m", latency_ms=0,
            )
        ]
        diverse = [
            AgentResponse(
                agent_id="a", role=AgentRole.PROPOSER,
                content="alpha bravo charlie delta echo foxtrot",
                turn_number=1, tokens_used=10, provider="p", model="m", latency_ms=0,
            )
        ]
        assert argument_diversity_score(diverse) > argument_diversity_score(repetitive)


class TestComputeAllMetrics:
    def test_full_metrics(self, sample_history):
        metrics = compute_all_metrics(
            sample_history, topic="AI regulation", max_turns=4
        )
        assert isinstance(metrics, DebateMetrics)
        assert metrics.total_messages == 4
        assert metrics.turns_completed == 4
        assert metrics.total_tokens > 0
        assert 0.0 <= metrics.coherence <= 1.0
        assert 0.0 <= metrics.evidence_strength <= 1.0
        assert 0.0 <= metrics.relevance <= 1.0
        assert 0.0 <= metrics.argument_diversity <= 1.0

    def test_to_dict(self, sample_history):
        metrics = compute_all_metrics(sample_history, topic="t", max_turns=4)
        d = metrics.to_dict()
        assert "quality" in d
        assert "process" in d
        assert "outcome" in d
        assert "coherence" in d["quality"]
