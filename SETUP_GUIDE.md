# Setup Guide: Getting Started with Multi-Agent Debate

This guide will help you set up and run your first debate in under 5 minutes.

## Option 1: Using OpenRouter (Recommended - Easiest)

OpenRouter gives you access to **all models** through one API key.

### Step 1: Get OpenRouter API Key

1. Go to [openrouter.ai](https://openrouter.ai)
2. Sign up and create an API key
3. Copy your key (starts with `sk-or-v1-...`)

### Step 2: Set Environment Variable

```bash
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"
```

To make it permanent:
```bash
echo 'export OPENROUTER_API_KEY="sk-or-v1-..."' >> ~/.zshrc
source ~/.zshrc
```

### Step 3: Run Your First Debate

```bash
cd ~/Desktop/AI_Debate

python3 cli.py --config config/openrouter.yaml debate \
  --topic "Should AI development be regulated?" \
  --agents proposer,critic,judge \
  --max-turns 3
```

**That's it!** The debate will use:
- GPT-4 Turbo for Proposer
- Claude 3.5 Sonnet for Critic  
- Command-R Plus for Judge

All through your single OpenRouter key.

---

## Option 2: Using Individual Provider Keys

If you already have OpenAI/Anthropic/Cohere keys.

### Step 1: Set API Keys

Set **at least one** (or all three):

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export COHERE_API_KEY="..."
```

### Step 2: Edit Config (if using only one provider)

If you only have **one key**, edit `config/default.yaml` to use that provider for all agents:

```yaml
agents:
  proposer:
    provider: openai     # Change to your provider
  critic:
    provider: openai     # Change to your provider
  judge:
    provider: openai     # Change to your provider
```

### Step 3: Run Debate

```bash
python3 cli.py debate \
  --topic "Should AI development be regulated?" \
  --agents proposer,critic,judge \
  --max-turns 3
```

---

## After Your First Debate

### View Saved Debates

```bash
python3 cli.py list-debates
```

### Generate Visualizations

```bash
python3 cli.py visualize --debate-id 1
```

Charts and transcript saved to `viz/output/`

### Customize Your Setup

Edit the YAML config to:
- Change which models each agent uses
- Adjust temperature (creativity) settings
- Modify max tokens per response
- Change debate protocols

---

## Quick Tips

### Start Small
- Use `--max-turns 3` for testing (cheaper, faster)
- Increase to 6-10 for full debates

### Cost Management
With OpenRouter, check [pricing](https://openrouter.ai/docs) for each model:
- GPT-4 Turbo: ~$0.03 per 1K tokens
- Claude 3.5 Sonnet: ~$0.015 per 1K tokens  
- Command-R Plus: ~$0.015 per 1K tokens
- Llama 3.1 70B: ~$0.001 per 1K tokens (cheapest!)

A 3-turn debate typically uses 3,000-5,000 tokens = **$0.01-0.15**

### Model Recommendations

| Agent Role | Best Models |
|---|---|
| Proposer | GPT-4 Turbo, Claude 3.5 Sonnet |
| Critic | Claude 3.5 Sonnet, GPT-4 |
| Fact Checker | Perplexity models (have web search!), Command-R Plus |
| Moderator | Claude 3 Opus (best at summarization) |
| Judge | GPT-4 Turbo, Claude 3.5 Sonnet |

### Protocols

- **round_robin**: Balanced, everyone speaks equally
- **adversarial**: Proposer vs Critic debate
- **collaborative**: All agents work together

---

## Troubleshooting

### "No API key" error

Check if the variable is set:
```bash
echo $OPENROUTER_API_KEY  # or OPENAI_API_KEY, etc.
```

If empty, export it again.

### Tests Failing

Run the test suite to verify everything works:
```bash
python3 -m pytest tests/ -v
```

All 90 tests should pass (no API keys needed for tests).

### Module Not Found

Make sure you installed dependencies:
```bash
python3 -m pip install -r requirements.txt
```

---

## Next Steps

1. **Read OPENROUTER.md** for advanced OpenRouter features
2. **Experiment with protocols** - try `adversarial` and `collaborative`
3. **Run benchmarks** - `python3 cli.py run-experiment`
4. **Explore the Python API** - see README.md for examples

---

## Getting Help

- Check **README.md** for full documentation
- See **OPENROUTER.md** for OpenRouter-specific details
- Review example configs in `config/`
- Run `python3 cli.py --help` for CLI commands

Happy debating! 🎯
