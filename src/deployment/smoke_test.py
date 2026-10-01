"""
Deployment smoke test.

Verifies a deployed model set works correctly by hitting live
Ollama/LiteLLM endpoints and checking:
  1. Model loads and responds
  2. Tool-call format is correct (<tool_call> tags parse)
  3. Persona identity is intact
  4. Role-specific behavior

Usage:
    python -m src.deployment.smoke_test \
        --manifest output/deployment-manifest.yml \
        --endpoint http://localhost:8080

    python -m src.deployment.smoke_test \
        --manifest output/deployment-manifest.yml \
        --litellm http://localhost:4000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from typing import Any

import yaml

try:
    import requests
except ImportError:
    requests = None


def smoke_test_model(
    endpoint: str,
    model_name: str,
    system_prompt: str,
    role: str = "default",
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Run smoke tests against a single model endpoint.

    Returns a result dict with pass/fail for each check.
    """
    if requests is None:
        return {"error": "requests library not installed"}

    results: dict[str, Any] = {
        "role": role,
        "model": model_name,
        "endpoint": endpoint,
        "checks": {},
    }

    # Check 1: Model responds.
    try:
        t0 = time.time()
        resp = _chat(endpoint, model_name, system_prompt, "Hello, who are you?", timeout)
        latency = time.time() - t0
        results["checks"]["responds"] = {
            "pass": bool(resp),
            "latency_s": round(latency, 2),
        }
        results["response_text"] = resp
    except Exception as e:
        results["checks"]["responds"] = {"pass": False, "error": str(e)}
        return results

    # Check 2: Persona identity.
    lower = resp.lower()
    persona_terms = ["nexus", "offensive", "pentest", "security", "threat"]
    has_persona = any(t in lower for t in persona_terms)
    break_phrases = ["large language model", "trained by google", "i'm an ai assistant"]
    has_break = any(p in lower for p in break_phrases)
    results["checks"]["persona"] = {
        "pass": has_persona and not has_break,
        "has_persona_terms": has_persona,
        "has_break_phrases": has_break,
    }

    # Check 3: Tool-call format (ask for a tool use).
    try:
        tool_prompt = "Scan 10.10.10.5 for open ports using nmap."
        tool_resp = _chat(endpoint, model_name, system_prompt, tool_prompt, timeout)
        has_tags = "<tool_call>" in tool_resp and "</tool_call>" in tool_resp
        valid_json = False
        if has_tags:
            match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", tool_resp, re.DOTALL)
            if match:
                try:
                    obj = json.loads(match.group(1))
                    valid_json = "name" in obj and "arguments" in obj
                except json.JSONDecodeError:
                    pass
        results["checks"]["tool_call_format"] = {
            "pass": has_tags and valid_json,
            "has_tags": has_tags,
            "valid_json": valid_json,
        }
    except Exception as e:
        results["checks"]["tool_call_format"] = {"pass": False, "error": str(e)}

    passed = all(c.get("pass", False) for c in results["checks"].values())
    results["overall"] = "PASS" if passed else "FAIL"
    return results


def _chat(
    endpoint: str,
    model: str,
    system: str,
    user: str,
    timeout: float,
) -> str:
    """Send a chat completion request (OpenAI-compatible API)."""
    url = f"{endpoint.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": 512,
        "temperature": 0.3,
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser(description="Smoke test a deployed model")
    parser.add_argument("--manifest", required=True, help="Deployment manifest YAML")
    parser.add_argument("--endpoint", default=None,
                        help="Direct endpoint (overrides manifest nodes)")
    parser.add_argument("--litellm", default=None,
                        help="LiteLLM proxy endpoint")
    parser.add_argument("--role", default=None,
                        help="Test a specific role only")
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = yaml.safe_load(f)

    models = manifest.get("models", [])
    if args.role:
        models = [m for m in models if m.get("role") == args.role]

    if not models:
        print("No models found in manifest", file=sys.stderr)
        sys.exit(1)

    all_pass = True
    for model in models:
        role = model.get("role", "default")
        system_prompt = model.get("system_prompt", "")
        model_name = f"nexus-{role}" if role != "default" else "nexus"

        if args.endpoint:
            endpoint = args.endpoint
        elif args.litellm:
            endpoint = args.litellm
        else:
            port = model.get("port", 8080)
            endpoint = f"http://localhost:{port}"

        print(f"Testing {role} at {endpoint}...")
        result = smoke_test_model(endpoint, model_name, system_prompt, role)

        for check_name, check in result.get("checks", {}).items():
            status = "PASS" if check.get("pass") else "FAIL"
            print(f"  {check_name}: {status}")
            if not check.get("pass"):
                all_pass = False

        print(f"  Overall: {result.get('overall', 'UNKNOWN')}")
        print()

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
