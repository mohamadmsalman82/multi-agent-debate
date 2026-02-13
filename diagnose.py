#!/usr/bin/env python3
"""Diagnostic script to test ALL API keys and find valid model names."""

import asyncio
import os
import sys
import httpx

print("=" * 60)
print("  MULTI-AGENT DEBATE - API DIAGNOSTICS")
print("=" * 60)

# ── Check environment variables ──────────────────────────────
print("\n[1] ENVIRONMENT VARIABLES\n")
keys = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
    "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
    "COHERE_API_KEY": os.getenv("COHERE_API_KEY", ""),
    "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY", ""),
}
for name, val in keys.items():
    if val:
        print(f"  OK {name} is set (length={len(val)}, prefix={val[:12]}...)")
    else:
        print(f"  -- {name} is NOT set")

available_keys = {k: v for k, v in keys.items() if v}
if not available_keys:
    print("\n  ERROR: No API keys found! Set at least one.")
    sys.exit(1)


async def test_anthropic():
    """Test Anthropic API key and list available models."""
    key = keys["ANTHROPIC_API_KEY"]
    if not key:
        print("  SKIP: No ANTHROPIC_API_KEY set")
        return None

    print(f"\n  Testing Anthropic key ({key[:12]}...)...")

    async with httpx.AsyncClient() as client:
        # First, try listing models via the API
        try:
            resp = await client.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                },
                timeout=15,
            )
            print(f"  Models endpoint: HTTP {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("data", [])
                print(f"  Available models ({len(models)}):")
                for m in models[:20]:
                    model_id = m.get("id", "unknown")
                    print(f"    - {model_id}")
                if models:
                    # Return a good default - prefer sonnet for cost/speed balance
                    model_ids = [m.get("id", "") for m in models]
                    # Try to find a sonnet model first
                    for mid in model_ids:
                        if "sonnet" in mid:
                            return mid
                    return model_ids[0]
            else:
                print(f"  Response: {resp.text[:300]}")
        except Exception as e:
            print(f"  Models list failed: {e}")

        # Fallback: try specific model names
        test_models = [
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5-20251001",
            "claude-opus-4-6",
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-latest",
            "claude-3-haiku-20240307",
        ]
        print(f"\n  Testing {len(test_models)} model names...")
        for model in test_models:
            try:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": model,
                        "max_tokens": 5,
                        "messages": [{"role": "user", "content": "Say hi"}],
                    },
                    timeout=20,
                )
                if resp.status_code == 200:
                    print(f"  OK Model '{model}' works!")
                    return model
                else:
                    error = resp.json().get("error", {}).get("message", resp.text[:100])
                    print(f"  -- Model '{model}': {resp.status_code} - {error}")
            except Exception as e:
                print(f"  -- Model '{model}': {e}")

    return None


async def test_openrouter():
    """Test OpenRouter API key."""
    key = keys["OPENROUTER_API_KEY"]
    if not key:
        print("  SKIP: No OPENROUTER_API_KEY set")
        return None

    print(f"\n  Testing OpenRouter key ({key[:12]}...)...")

    async with httpx.AsyncClient() as client:
        # Test with a simple completion first
        test_models = [
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "anthropic/claude-sonnet-4.5",
            "openai/gpt-3.5-turbo",
        ]
        for model in test_models:
            try:
                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "max_tokens": 5,
                        "messages": [{"role": "user", "content": "Say hi"}],
                    },
                    timeout=20,
                )
                if resp.status_code == 200:
                    print(f"  OK Model '{model}' works!")
                    return model
                elif resp.status_code == 401:
                    print(f"  -- AUTH FAILED (401): {resp.text[:200]}")
                    print("  -> Your OpenRouter key may be invalid or account has no credits")
                    return None
                else:
                    print(f"  -- Model '{model}': {resp.status_code} - {resp.text[:150]}")
            except Exception as e:
                print(f"  -- Model '{model}': {e}")

    return None


async def test_openai():
    """Test OpenAI API key."""
    key = keys["OPENAI_API_KEY"]
    if not key:
        print("  SKIP: No OPENAI_API_KEY set")
        return None

    print(f"\n  Testing OpenAI key ({key[:12]}...)...")

    async with httpx.AsyncClient() as client:
        test_models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
        for model in test_models:
            try:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "max_tokens": 5,
                        "messages": [{"role": "user", "content": "Say hi"}],
                    },
                    timeout=20,
                )
                if resp.status_code == 200:
                    print(f"  OK Model '{model}' works!")
                    return model
                else:
                    print(f"  -- Model '{model}': {resp.status_code} - {resp.text[:150]}")
            except Exception as e:
                print(f"  -- Model '{model}': {e}")

    return None


async def test_cohere():
    """Test Cohere API key."""
    key = keys["COHERE_API_KEY"]
    if not key:
        print("  SKIP: No COHERE_API_KEY set")
        return None

    print(f"\n  Testing Cohere key ({key[:12]}...)...")

    async with httpx.AsyncClient() as client:
        test_models = ["command-r-plus", "command-r", "command"]
        for model in test_models:
            try:
                resp = await client.post(
                    "https://api.cohere.com/v2/chat",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "max_tokens": 5,
                        "messages": [{"role": "user", "content": "Say hi"}],
                    },
                    timeout=20,
                )
                if resp.status_code == 200:
                    print(f"  OK Model '{model}' works!")
                    return model
                else:
                    print(f"  -- Model '{model}': {resp.status_code} - {resp.text[:150]}")
            except Exception as e:
                print(f"  -- Model '{model}': {e}")

    return None


async def main():
    print("\n[2] TESTING API CONNECTIONS\n")

    working = {}

    if keys["ANTHROPIC_API_KEY"]:
        model = await test_anthropic()
        if model:
            working["anthropic"] = model

    if keys["OPENAI_API_KEY"]:
        model = await test_openai()
        if model:
            working["openai"] = model

    if keys["OPENROUTER_API_KEY"]:
        model = await test_openrouter()
        if model:
            working["openrouter"] = model

    if keys["COHERE_API_KEY"]:
        model = await test_cohere()
        if model:
            working["cohere"] = model

    print(f"\n{'=' * 60}")
    print(f"  RESULTS")
    print(f"{'=' * 60}")

    if not working:
        print("\n  NO WORKING PROVIDERS FOUND")
        print("  Check your API keys and account credits.")
        print("\n  Tips:")
        print("  - Verify keys are exported: echo $ANTHROPIC_API_KEY")
        print("  - Check account has credits at provider's dashboard")
        print("  - Try regenerating your API key")
    else:
        print(f"\n  {len(working)} working provider(s):")
        for provider, model in working.items():
            print(f"    - {provider}: model '{model}'")

        # Pick best provider for a quick test
        best_provider = list(working.keys())[0]
        best_model = working[best_provider]

        # Generate the run command
        if best_provider in ("openai", "anthropic", "cohere"):
            config = f"config/simple_{best_provider}.yaml"
        else:
            config = "config/openrouter.yaml"

        print(f"\n  RECOMMENDED: Use '{best_provider}' with model '{best_model}'")
        print(f"\n  Quick test command:")
        print(f"  python3 cli.py --config {config} debate \\")
        print(f'    --topic "Should AI development be regulated?" \\')
        print(f"    --agents proposer,critic,judge --max-turns 3")

        # If the working model differs from what's in the config, warn
        print(f"\n  NOTE: Make sure your config uses model '{best_model}'")
        print(f"  Check: config file '{config}'")

    # Write results for reference
    import json
    with open("diagnose_results.json", "w") as f:
        json.dump(working, f, indent=2)
    print(f"\n  Results saved to diagnose_results.json")


asyncio.run(main())
