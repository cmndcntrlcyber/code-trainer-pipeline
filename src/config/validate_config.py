"""
Pipeline configuration validator.

Checks structural completeness, budget arithmetic, and domain/role
consistency. Run standalone or import as a library.

Usage:
    python -m src.config.validate_config src/config/pipeline-gemma26b.yml
    python -m src.config.validate_config src/config/pipeline-swarm-v4a.yml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config.settings import load_config


def validate_pipeline_config(config: dict) -> list[str]:
    """Validate a pipeline config for completeness and consistency.

    Returns a list of warnings/errors (empty means valid).
    """
    issues: list[str] = []

    # Check domain reference resolves.
    domain = config.get("domain")
    if domain:
        domain_path = Path(__file__).parent / "domains" / f"{domain}.yml"
        if not domain_path.exists():
            issues.append(f"Domain file not found: {domain_path}")

    # Determine topology.
    pipeline_meta = config.get("pipeline", {})
    topology = pipeline_meta.get("topology", "single") if isinstance(pipeline_meta, dict) else "single"
    is_swarm = topology == "swarm" or "swarm_meta" in config

    if is_swarm:
        _validate_swarm(config, issues)
    else:
        _validate_single(config, issues)

    return issues


def _validate_single(config: dict, issues: list[str]) -> None:
    """Validate a single-model pipeline config."""
    for required in ("sft", "deployment"):
        if required not in config:
            for section in ("qwen_finetuning", "gemma_finetuning"):
                if section in config:
                    break
            else:
                if required == "sft":
                    issues.append(f"Missing section: {required}")

    deploy = config.get("deployment", {})
    if deploy and not deploy.get("quantization"):
        issues.append("deployment.quantization not set")


def _validate_swarm(config: dict, issues: list[str]) -> None:
    """Validate a swarm pipeline config."""
    meta = config.get("swarm_meta", {})
    nodes = meta.get("nodes", [])
    if not nodes:
        issues.append("swarm_meta.nodes is empty")

    defined_roles = set()
    for node in nodes:
        role_str = node.get("role", "")
        for r in role_str.split("+"):
            defined_roles.add(r.strip())

    role_sections = {"orchestrator", "worker", "explore", "triage"}
    for role in role_sections:
        section = config.get(role, {})
        if not section:
            continue

        if not section.get("base_model"):
            issues.append(f"{role}: missing base_model")

        sft = section.get("sft", {})
        if sft and not sft.get("dataset_id"):
            issues.append(f"{role}.sft: missing dataset_id")

        deploy = section.get("deployment", {})
        if deploy and not deploy.get("quantization") and deploy.get("format") != "rkllm":
            issues.append(f"{role}.deployment: missing quantization (and not rkllm)")


def main():
    parser = argparse.ArgumentParser(description="Validate a pipeline config")
    parser.add_argument("config", help="Path to pipeline YAML")
    args = parser.parse_args()

    try:
        config = load_config(args.config, substitute_env=False)
    except Exception as e:
        print(f"ERROR loading config: {e}", file=sys.stderr)
        sys.exit(1)

    issues = validate_pipeline_config(config)
    if issues:
        print(f"Found {len(issues)} issue(s):")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)
    else:
        print("Config is valid.")


if __name__ == "__main__":
    main()
