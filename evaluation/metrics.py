"""Debate quality metrics and scoring.

Provides both heuristic metrics (computed locally) and an optional
LLM-as-judge evaluation path for deeper analysis.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from agents.base import AgentResponse, AgentRole


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class DebateMetrics:
    """Collection of quality and process metrics for a debate."""

    # Quality
    coherence: float = 0.0
    evidence_strength: float = 0.0
    relevance: float = 0.0
    argument_diversity: float = 0.0

    # Process
    turns_completed: int = 0
    total_messages: int = 0
    agent_participation: dict[str, int] = field(default_factory=dict)
    avg_response_length: float = 0.0
    total_tokens: int = 0

    # Outcome
    consensus_level: float = 0.0
    final_confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality": {
                "coherence": round(self.coherence, 3),
                "evidence_strength": round(self.evidence_strength, 3),
                "relevance": round(self.relevance, 3),
                "argument_diversity": round(self.argument_diversity, 3),
            },
            "process": {
                "turns_completed": self.turns_completed,
                "total_messages": self.total_messages,
                "agent_participation": self.agent_participation,
                "avg_response_length": round(self.avg_response_length, 1),
                "total_tokens": self.total_tokens,
            },
            "outcome": {
                "consensus_level": round(self.consensus_level, 3),
                "final_confidence": round(self.final_confidence, 3),
            },
        }


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def coherence_score(history: list[AgentResponse]) -> float:
    """Estimate coherence from response length consistency and structure.

    A score of 0–1 where higher means more coherent (well-structured,
    consistent-length responses with clear paragraph breaks).
    """
    if not history:
        return 0.0

    lengths = [len(r.content) for r in history]
    mean_len = sum(lengths) / len(lengths)
    if mean_len == 0:
        return 0.0

    # Coefficient of variation (lower = more consistent = more coherent)
    std = math.sqrt(sum((x - mean_len) ** 2 for x in lengths) / len(lengths))
    cv = std / mean_len

    # Structure bonus: reward responses with paragraph breaks
    structured_count = sum(
        1 for r in history if "\n\n" in r.content or "\n-" in r.content or "\n1." in r.content
    )
    structure_bonus = structured_count / len(history) * 0.2

    score = max(0.0, 1.0 - cv) + structure_bonus
    return min(1.0, score)


def evidence_strength_score(history: list[AgentResponse]) -> float:
    """Heuristic estimate of evidence use across the debate.

    Looks for numeric references, citation-like patterns, examples,
    and keywords signalling evidence.
    """
    if not history:
        return 0.0

    evidence_keywords = {
        "study", "research", "data", "evidence", "according",
        "percent", "%", "statistic", "report", "survey",
        "example", "for instance", "specifically", "demonstrates",
        "analysis", "findings", "experiment", "source",
    }

    total_score = 0.0
    for resp in history:
        text_lower = resp.content.lower()
        keyword_hits = sum(1 for kw in evidence_keywords if kw in text_lower)
        # Numeric references
        num_hits = len(re.findall(r"\b\d+\.?\d*%?\b", resp.content))
        msg_score = min(1.0, (keyword_hits * 0.08 + min(num_hits, 5) * 0.04))
        total_score += msg_score

    return min(1.0, total_score / len(history))


def relevance_score(history: list[AgentResponse], topic: str = "") -> float:
    """Estimate how on-topic the discussion stayed.

    Uses keyword overlap between topic and responses.
    """
    if not history or not topic:
        return 0.5  # neutral if no topic provided

    topic_words = set(topic.lower().split())
    # Remove common stop words
    stop = {"the", "a", "an", "is", "are", "be", "should", "of", "in", "to", "and", "or", "for"}
    topic_words -= stop

    if not topic_words:
        return 0.5

    scores: list[float] = []
    for resp in history:
        words = set(resp.content.lower().split())
        overlap = len(topic_words & words)
        scores.append(min(1.0, overlap / max(len(topic_words), 1)))

    return sum(scores) / len(scores)


def argument_diversity_score(history: list[AgentResponse]) -> float:
    """Measure diversity of arguments by unique vocabulary ratio.

    A higher score indicates that participants introduced varied vocabulary
    rather than repeating the same phrases.
    """
    if not history:
        return 0.0

    all_words: list[str] = []
    for resp in history:
        all_words.extend(resp.content.lower().split())

    if not all_words:
        return 0.0

    unique = set(all_words)
    # Type-token ratio (capped to 0-1)
    ttr = len(unique) / len(all_words)
    return min(1.0, ttr * 2)  # Scale so 0.5 TTR → 1.0


def _consensus_heuristic(history: list[AgentResponse]) -> float:
    """Rough consensus estimator based on agreement language."""
    if not history:
        return 0.0

    agreement = {"agree", "consensus", "valid point", "fair", "concede", "acknowledge", "correct"}
    disagreement = {"disagree", "incorrect", "wrong", "flawed", "fallacy", "reject", "deny"}

    agree_count = 0
    disagree_count = 0
    for resp in history:
        text = resp.content.lower()
        agree_count += sum(1 for w in agreement if w in text)
        disagree_count += sum(1 for w in disagreement if w in text)

    total = agree_count + disagree_count
    if total == 0:
        return 0.5

    return agree_count / total


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def compute_all_metrics(
    history: list[AgentResponse],
    topic: str = "",
    max_turns: int = 0,
) -> DebateMetrics:
    """Compute the full suite of debate metrics."""
    participation: dict[str, int] = {}
    total_tokens = 0
    for r in history:
        participation[r.role.value] = participation.get(r.role.value, 0) + 1
        total_tokens += r.tokens_used

    avg_len = sum(len(r.content) for r in history) / max(len(history), 1)

    return DebateMetrics(
        coherence=coherence_score(history),
        evidence_strength=evidence_strength_score(history),
        relevance=relevance_score(history, topic),
        argument_diversity=argument_diversity_score(history),
        turns_completed=max_turns or (max(r.turn_number for r in history) if history else 0),
        total_messages=len(history),
        agent_participation=participation,
        avg_response_length=avg_len,
        total_tokens=total_tokens,
        consensus_level=_consensus_heuristic(history),
        final_confidence=0.0,  # set by judge if available
    )
