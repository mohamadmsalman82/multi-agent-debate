# Multi-Agent Debate System

A Python-based multi-agent debate framework where LLM agents engage in structured debates using multiple providers — **OpenAI GPT**, **Anthropic Claude**, **Cohere Command-R**, and **OpenRouter**.

Each agent can be backed by a **different LLM provider**, enabling cross-model comparisons and heterogeneous debates where GPT-4 argues against Claude while Cohere fact-checks.

**NEW:** Use **OpenRouter** with a single API key to access all models from OpenAI, Anthropic, Cohere, Meta, Google, and more!

---

## Features

- **Multi-provider support** — OpenAI, Anthropic, Cohere, and OpenRouter behind a unified async interface with automatic retries and rate-limit handling
- **5 specialised agents** — Proposer, Critic, Fact Checker, Moderator, Judge — each with role-specific prompting strategies
- **3 debate protocols** — Round Robin, Adversarial, Collaborative — controlling turn-taking dynamics
- **SQLite persistence** — full debate history, agent configs, and evaluation metrics stored automatically
- **Evaluation framework** — coherence, evidence strength, relevance, diversity, and consensus scoring
- **Visualization** — participation charts, token usage breakdown, quality radar, and formatted transcripts
- **Benchmark runner** — systematic experiments across topics, protocols, and repetitions via YAML config
- **CLI** — run debates, experiments, and visualizations from the terminal
- **93% test coverage** — 89 tests across all modules using a zero-cost MockProvider

---

## Project Structure

```
multi-agent-debate/
├── agents/
│   ├── __init__.py
│   ├── llm_provider.py       # LLM provider abstraction (OpenAI, Anthropic, Cohere)
│   ├── base.py               # BaseAgent class with memory management
│   ├── proposer.py           # Generates arguments with evidence
│   ├── critic.py             # Challenges arguments, identifies fallacies
│   ├── fact_checker.py       # Verifies claims, rates confidence
│   ├── moderator.py          # Manages flow, summarises progress
│   └── judge.py              # Evaluates quality, delivers verdict
├── orchestration/
│   ├── __init__.py
│   ├── debate_manager.py     # DebateManager — runs debates end-to-end
│   └── protocols.py          # Turn-taking strategies
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py            # Quality, process, and outcome metrics
│   └── validators.py         # Response and config validation
├── data/
│   ├── __init__.py
│   ├── database.py           # Async SQLite layer (aiosqlite)
│   └── models.py             # Pydantic models for DB records
├── experiments/
│   ├── __init__.py
│   └── benchmark.py          # Benchmark runner for systematic experiments
├── viz/
│   ├── __init__.py
│   └── visualize.py          # Matplotlib charts + text transcripts
├── config/
│   ├── default.yaml          # Default agent/provider/protocol configuration
│   └── experiment.yaml       # Benchmark experiment definitions
├── tests/                    # 89 tests — 93% coverage
│   ├── conftest.py           # MockProvider + shared fixtures
│   ├── test_llm_provider.py
│   ├── test_agents.py
│   ├── test_database.py
│   ├── test_debate_manager.py
│   ├── test_protocols.py
│   ├── test_metrics.py
│   ├── test_validators.py
│   ├── test_visualization.py
│   └── test_cli.py
├── cli.py                    # Click-based CLI
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/multi-agent-debate.git
cd multi-agent-debate
pip install -r requirements.txt
```

### 2. Set API keys

You need at least **one** provider key. Export whichever you have:

```bash
# Option A: Use individual provider keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export COHERE_API_KEY="..."

# Option B: Use OpenRouter (recommended - one key for all models)
export OPENROUTER_API_KEY="sk-or-..."
```

### 3. Run a debate

```bash
# Simple 3-agent debate (uses config/default.yaml)
python cli.py debate \
  --topic "Should AI development be regulated?" \
  --protocol round_robin \
  --agents proposer,critic,judge \
  --max-turns 4

# Use OpenRouter for all models (if using OpenRouter)
python cli.py --config config/openrouter.yaml debate \
  --topic "Should AI development be regulated?" \
  --protocol round_robin \
  --agents proposer,critic,judge \
  --max-turns 4

# Full 5-agent adversarial debate
python cli.py debate \
  --topic "Is universal basic income viable?" \
  --protocol adversarial \
  --agents proposer,critic,fact_checker,moderator,judge \
  --max-turns 6

# Collaborative protocol
python cli.py debate \
  --topic "Should autonomous weapons be banned?" \
  --protocol collaborative \
  --agents proposer,critic,fact_checker,moderator,judge \
  --max-turns 6
```

### 4. View and visualize results

```bash
# List all saved debates
python cli.py list-debates

# Generate charts and transcript for a specific debate
python cli.py visualize --debate-id 1
```

### 5. Run benchmark experiments

```bash
python cli.py run-experiment --benchmark-config config/experiment.yaml
```

---

## Python API

### Using individual providers

```python
import asyncio
from orchestration import DebateManager
from agents import Proposer, Critic, Judge
from agents.llm_provider import OpenAIProvider, AnthropicProvider, CohereProvider

# Each agent uses a different LLM provider
proposer = Proposer(provider=OpenAIProvider(model="gpt-4-turbo-preview"))
critic   = Critic(provider=AnthropicProvider(model="claude-3-5-sonnet-20241022"))
judge    = Judge(provider=CohereProvider(model="command-r-plus"))

manager = DebateManager(
    agents=[proposer, critic, judge],
    protocol="round_robin",
)

async def main():
    result = await manager.run_debate(
        topic="Should AI development be regulated?",
        max_turns=6,
    )
    print(result.summary)
    print(result.metrics)
    print(f"Proposer used: {proposer.provider.name}")   # openai
    print(f"Critic used:   {critic.provider.name}")      # anthropic
    print(f"Judge used:    {judge.provider.name}")        # cohere

asyncio.run(main())
```

### Using OpenRouter (one API key for all models)

```python
import asyncio
from orchestration import DebateManager
from agents import Proposer, Critic, Judge
from agents.llm_provider import OpenRouterProvider

# All agents use OpenRouter with different models
proposer = Proposer(provider=OpenRouterProvider(model="openai/gpt-4-turbo"))
critic   = Critic(provider=OpenRouterProvider(model="anthropic/claude-3.5-sonnet"))
judge    = Judge(provider=OpenRouterProvider(model="cohere/command-r-plus"))

manager = DebateManager(
    agents=[proposer, critic, judge],
    protocol="round_robin",
)

async def main():
    result = await manager.run_debate(
        topic="Should AI development be regulated?",
        max_turns=6,
    )
    print(result.summary)
    print(f"All agents used OpenRouter with different models")

asyncio.run(main())
```

---

## Debate Protocols

| Protocol | How it works |
|---|---|
| **round_robin** | Every agent speaks once per turn in a fixed order. Judge speaks only on the final turn. |
| **adversarial** | Proposer and Critic alternate turns. Fact Checker intervenes every 3rd turn. Judge delivers the final verdict. |
| **collaborative** | Moderator opens each turn. All agents contribute ideas. Judge gives interim feedback every 3 turns plus a final verdict. |

## Agent Roles

| Agent | Responsibility | Default Provider |
|---|---|---|
| **Proposer** | Formulates arguments with evidence and examples | OpenAI |
| **Critic** | Challenges arguments, identifies fallacies and weak reasoning | Anthropic |
| **Fact Checker** | Verifies claims with HIGH / MEDIUM / LOW / UNVERIFIABLE ratings | Cohere |
| **Moderator** | Manages flow, summarises progress, enforces civil discourse | Anthropic |
| **Judge** | Scores participants (1–10) and delivers a final verdict | OpenAI |

---

## Evaluation Metrics

**Quality** (0–1 scale):
| Metric | What it measures |
|---|---|
| Coherence | Response length consistency and structural quality |
| Evidence Strength | Use of data, citations, statistics, and examples |
| Relevance | How closely responses track the debate topic |
| Argument Diversity | Vocabulary richness (type-token ratio) |

**Process:**
- Turns completed, total messages, per-agent participation, total token usage, average latency

**Outcome:**
- Consensus level (agreement vs. disagreement language ratio)
- Final confidence score

---

## Configuration

### Using individual providers (config/default.yaml)

```yaml
api:
  openai:
    model: gpt-4-turbo-preview
    api_key_env: OPENAI_API_KEY
  anthropic:
    model: claude-3-5-sonnet-20241022
    api_key_env: ANTHROPIC_API_KEY
  cohere:
    model: command-r-plus
    api_key_env: COHERE_API_KEY
  openrouter:
    model: openai/gpt-4-turbo
    api_key_env: OPENROUTER_API_KEY
  timeout: 30
  max_retries: 3

agents:
  proposer:
    provider: openai
    temperature: 0.7
    max_tokens: 500
  critic:
    provider: anthropic
    temperature: 0.8
    max_tokens: 500
  # ... etc.
```

### Using OpenRouter (config/openrouter.yaml)

```yaml
agents:
  proposer:
    provider: openrouter
    model: openai/gpt-4-turbo      # Access GPT-4 via OpenRouter
    temperature: 0.7
  critic:
    provider: openrouter
    model: anthropic/claude-3.5-sonnet  # Access Claude via OpenRouter
    temperature: 0.8
  fact_checker:
    provider: openrouter
    model: cohere/command-r-plus   # Access Cohere via OpenRouter
    temperature: 0.3
  # ... etc.
```

Then run:
```bash
python cli.py --config config/openrouter.yaml debate --topic "..."
```

### OpenRouter Model Names

OpenRouter uses a `provider/model` format:
- `openai/gpt-4-turbo`, `openai/gpt-4`, `openai/gpt-3.5-turbo`
- `anthropic/claude-3.5-sonnet`, `anthropic/claude-3-opus`
- `cohere/command-r-plus`, `cohere/command-r`
- `meta-llama/llama-3.1-70b-instruct`
- `google/gemini-pro-1.5`
- And many more — see [OpenRouter docs](https://openrouter.ai/docs)

---

## Testing

```bash
# Run all 89 tests
python -m pytest tests/ -v

# With coverage report
python -m pytest tests/ \
  --cov=agents --cov=orchestration --cov=evaluation --cov=data --cov=viz \
  --cov-report=term-missing
```

All tests use a `MockProvider` — no API keys or network calls needed.

---

## Tech Stack

| Dependency | Purpose |
|---|---|
| `openai >= 1.0` | OpenAI GPT async client (also used for OpenRouter) |
| `anthropic >= 0.18` | Anthropic Claude async client |
| `cohere >= 5.0` | Cohere Command-R async client |
| `pydantic >= 2.0` | Data validation and models |
| `aiosqlite >= 0.19` | Async SQLite persistence |
| `pyyaml >= 6.0` | YAML config parsing |
| `click >= 8.1` | CLI framework |
| `matplotlib >= 3.7` | Visualization charts |
| `pytest >= 7.4` | Testing framework |
| `pytest-asyncio >= 0.21` | Async test support |

**Requires Python 3.11+**

---

## License

MIT
