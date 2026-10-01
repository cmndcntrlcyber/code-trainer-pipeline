"""
LiteLLM routing config generator.

Reads a pipeline config with swarm_meta topology and produces a
litellm-config.yaml suitable for the nexus-harness swarm proxy.

Usage:
    python -m src.deployment.litellm_gen \
        --config src/config/pipeline-swarm-v4a.yml \
        --output swarm/litellm-config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from src.config.settings import load_config


_ROLE_TIMEOUTS = {
    "orchestrator": 300,
    "worker": 120,
    "explore": 60,
    "triage": 10,
    "embed": 30,
}

_ROLE_MAX_TOKENS = {
    "orchestrator": 4096,
    "worker": 2048,
    "explore": 1024,
    "triage": 32,
}


def generate_litellm_config(config: dict[str, Any]) -> dict[str, Any]:
    """Generate a LiteLLM routing config from a swarm pipeline config.

    Args:
        config: Parsed pipeline YAML with ``swarm_meta.nodes`` and
                per-role sections.

    Returns:
        Dict suitable for ``yaml.dump()`` as a litellm-config.yaml.
    """
    meta = config.get("swarm_meta", {})
    nodes = meta.get("nodes", [])

    model_list = []
    for node in nodes:
        hostname = node.get("hostname", "unknown")
        raw_role = node.get("role", "")
        port = node.get("port")
        ports = node.get("ports", [])
        if port is None and ports:
            port = ports[1] if len(ports) > 1 else ports[0]
        elif port is None:
            port = 8080

        roles = [r.strip() for r in raw_role.split("+")]
        for role in roles:
            if role in ("control", "control_plane"):
                continue

            role_section = config.get(role, {})
            base_model = role_section.get("base_model", "unknown")
            deploy = role_section.get("deployment", {})
            quant = deploy.get("quantization", "")

            model_name = f"nexus-{role}"
            description = f"{base_model} {quant} on {node.get('gpu', 'GPU')}".strip()

            entry: dict[str, Any] = {
                "model_name": model_name,
                "litellm_params": {
                    "model": f"openai/{model_name}-{hostname.split('-')[-1] if '-' in hostname else hostname}",
                    "api_base": f"http://{hostname}:{port}/v1",
                    "api_key": "sk-no-key",
                },
                "model_info": {
                    "description": description,
                },
            }

            max_tokens = _ROLE_MAX_TOKENS.get(role)
            if max_tokens:
                entry["litellm_params"]["max_tokens"] = max_tokens

            timeout = _ROLE_TIMEOUTS.get(role)
            if timeout:
                entry["litellm_params"]["timeout"] = timeout

            model_list.append(entry)

    result = {
        "model_list": model_list,
        "router_settings": {
            "routing_strategy": "simple-shuffle",
            "num_retries": 2,
            "timeout": 300,
            "retry_after": 5,
            "allowed_fails": 2,
            "cooldown_time": 60,
        },
        "general_settings": {
            "master_key": "sk-nexus-local",
            "drop_params": True,
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate LiteLLM routing config from pipeline YAML"
    )
    parser.add_argument("--config", required=True, help="Pipeline YAML path")
    parser.add_argument("--output", default=None,
                        help="Output file (default: stdout)")
    args = parser.parse_args()

    config = load_config(args.config)
    result = generate_litellm_config(config)

    content = yaml.dump(result, default_flow_style=False, sort_keys=False)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(content)
        print(f"LiteLLM config written to {args.output}")
    else:
        print(content)


if __name__ == "__main__":
    main()
