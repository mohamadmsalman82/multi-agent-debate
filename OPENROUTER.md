# Using OpenRouter with Multi-Agent Debate

OpenRouter provides unified access to multiple AI providers through a single API key. This is the **simplest way** to run debates using different models without managing multiple API keys.

## Why OpenRouter?

- ✅ **One API key** for all models (OpenAI, Anthropic, Cohere, Meta, Google, etc.)
- ✅ **Unified billing** — one invoice instead of multiple
- ✅ **Flexible routing** — use the best model for each agent role
- ✅ **Cost optimization** — compare prices across providers
- ✅ **No provider lock-in** — switch models anytime

## Quick Start

### 1. Get an OpenRouter API key

Sign up at [openrouter.ai](https://openrouter.ai) and create an API key.

### 2. Set the environment variable

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

Or add to your `~/.zshrc` (macOS) or `~/.bashrc` (Linux):

```bash
echo 'export OPENROUTER_API_KEY="sk-or-v1-..."' >> ~/.zshrc
source ~/.zshrc
```

### 3. Run a debate using the OpenRouter config

```bash
python cli.py --config config/openrouter.yaml debate \
  --topic "Should AI development be regulated?" \
  --protocol round_robin \
  --agents proposer,critic,judge \
  --max-turns 4
```

That's it! The system will use:
- GPT-4 Turbo for the Proposer
- Claude 3.5 Sonnet for the Critic
- Command-R Plus for the Judge

All through your single OpenRouter API key.

## Available Models

OpenRouter supports 100+ models. Popular choices:

### OpenAI
- `openai/gpt-4o` (recommended)
- `openai/gpt-4o-mini`
- `openai/gpt-3.5-turbo`

### Anthropic
- `anthropic/claude-sonnet-4.5` (recommended)
- `anthropic/claude-haiku-4.5`
- `anthropic/claude-opus-4.6`

### Cohere
- `cohere/command-r-plus`
- `cohere/command-r`

### Meta
- `meta-llama/llama-3.1-70b-instruct`
- `meta-llama/llama-3.1-8b-instruct`

### Google
- `google/gemini-2.5-pro-preview`
- `google/gemini-2.5-flash-preview`

### Other
- `mistralai/mistral-large`
- `perplexity/llama-3.1-sonar-large-128k-online` (web search!)

See the full list at [openrouter.ai/docs](https://openrouter.ai/docs)

## Customizing Your Setup

Edit `config/openrouter.yaml` to change which models each agent uses:

```yaml
agents:
  proposer:
    provider: openrouter
    model: openai/gpt-4o                # Strong reasoning
    temperature: 0.7
    max_tokens: 500
  
  critic:
    provider: openrouter
    model: anthropic/claude-sonnet-4.5 # Excellent at critique
    temperature: 0.8
    max_tokens: 500
  
  fact_checker:
    provider: openrouter
    model: perplexity/llama-3.1-sonar-large-128k-online  # Has web search!
    temperature: 0.3
    max_tokens: 300
  
  moderator:
    provider: openrouter
    model: anthropic/claude-sonnet-4.5 # Best for summarization
    temperature: 0.5
    max_tokens: 400
  
  judge:
    provider: openrouter
    model: openai/gpt-4o               # Analytical scoring
    temperature: 0.4
    max_tokens: 600
```

## Cost Comparison

OpenRouter shows real-time pricing for each model. Example costs per 1M tokens (as of 2024):

| Model | Input | Output |
|---|---|---|
| GPT-4 Turbo | $10 | $30 |
| Claude 3.5 Sonnet | $3 | $15 |
| Command-R Plus | $3 | $15 |
| Llama 3.1 70B | $0.88 | $0.88 |

A typical 6-turn debate with 3 agents uses ~5,000-10,000 tokens total, costing **$0.05-0.30** depending on model choice.

## Fallback Models

OpenRouter supports automatic fallbacks. Add this to your model names:

```yaml
model: "openai/gpt-4o,anthropic/claude-sonnet-4.5"
```

If GPT-4 is unavailable or rate-limited, OpenRouter automatically tries Claude.

## Python API

```python
from agents import Proposer, Critic, Judge
from agents.llm_provider import OpenRouterProvider
from orchestration import DebateManager

# All agents use OpenRouter with different models
proposer = Proposer(
    provider=OpenRouterProvider(model="openai/gpt-4o")
)
critic = Critic(
    provider=OpenRouterProvider(model="anthropic/claude-sonnet-4.5")
)
judge = Judge(
    provider=OpenRouterProvider(model="cohere/command-r-plus")
)

manager = DebateManager(
    agents=[proposer, critic, judge],
    protocol="round_robin",
)

result = await manager.run_debate(
    topic="Your debate topic here",
    max_turns=6,
)
```

## Troubleshooting

### "No API key" error

Make sure you've exported the key:
```bash
echo $OPENROUTER_API_KEY  # Should print your key
```

If empty, export it again:
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

### Model not found

Check that the model name uses OpenRouter's format (`provider/model-name`):
- ✅ `openai/gpt-4o`
- ❌ `gpt-4o`

### Rate limits

OpenRouter has per-model rate limits. If you hit limits:
1. Add a fallback model (see above)
2. Reduce `max_turns` to lower total requests
3. Increase `timeout` in the config if requests are slow

## Benefits over Direct API Keys

| Feature | OpenRouter | Direct APIs |
|---|---|---|
| API keys needed | 1 | 3+ |
| Billing accounts | 1 | 3+ |
| Model switching | Instant | Requires code changes |
| Fallback routing | Built-in | Manual |
| Cost comparison | Real-time | Manual research |
| Web access models | Yes (Perplexity) | No |

---

**Questions?** Check the [OpenRouter docs](https://openrouter.ai/docs) or open an issue on GitHub.
