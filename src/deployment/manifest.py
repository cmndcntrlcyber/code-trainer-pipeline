"""
Deployment manifest generator.

Produces a ``deployment-manifest.yml`` that captures everything needed
to deploy trained models into nexus-harness: per-role model references,
system prompts, chat templates, and fleet topology.

Usage:
    python -m src.deployment.manifest \
        --config src/config/pipeline-gemma26b.yml \
        --output output/deployment-manifest.yml

    python -m src.deployment.manifest \
        --config src/config/pipeline-swarm-v4a.yml \
        --output output/deployment-manifest.yml
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.config.domain_loader import get_domain
from src.config.settings import load_config
from src.deployment.chat_templates import detect_model_family


def generate_deployment_manifest(
    config: dict[str, Any],
    config_path: str | None = None,
) -> dict[str, Any]:
    """Generate a deployment manifest from a pipeline config.

    Args:
        config: Parsed pipeline config dict.
        config_path: Original config file path (for provenance).

    Returns:
        Manifest dict suitable for YAML serialization.
    """
    domain_name = config.get("domain", "offsec")
    domain = get_domain(domain_name)

    pipeline_meta = config.get("pipeline", {})
    if not isinstance(pipeline_meta, dict):
        pipeline_meta = {}

    is_swarm = "swarm_meta" in config or pipeline_meta.get("topology") == "swarm"

    manifest: dict[str, Any] = {
        "manifest_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "domain": domain_name,
        "topology": "swarm" if is_swarm else "single",
        "pipeline_name": pipeline_meta.get("name", config_path or "unknown"),
    }

    if config_path:
        config_text = Path(config_path).read_text()
        manifest["config_hash"] = hashlib.sha256(config_text.encode()).hexdigest()[:12]

    if is_swarm:
        manifest["nodes"] = _build_nodes(config)
        manifest["models"] = _build_swarm_models(config, domain)
    else:
        manifest["models"] = _build_single_model(config, domain)

    return manifest


def _build_nodes(config: dict) -> list[dict]:
    meta = config.get("swarm_meta", {})
    nodes = []
    for node in meta.get("nodes", []):
        nodes.append({
            "hostname": node.get("hostname"),
            "role": node.get("role"),
            "gpu": node.get("gpu"),
            "port": node.get("port"),
        })
    return nodes


def _build_swarm_models(config: dict, domain) -> list[dict]:
    models = []
    for role_name in ("orchestrator", "worker", "explore", "triage"):
        section = config.get(role_name)
        if not section:
            continue
        models.append(_role_to_model_entry(section, domain, role_name))
    return models


def _build_single_model(config: dict, domain) -> list[dict]:
    deploy = config.get("deployment", {})
    base_model = config.get("base_model", "")

    # Search known section names for base_model and deployment info.
    _DEPLOY_SECTIONS = (
        "gemma_deployment", "qwen_deployment", "deployment",
    )
    _MODEL_SECTIONS = (
        "gemma_finetuning", "gemma_dapt", "qwen_finetuning",
        "gemma_deployment", "qwen_deployment",
    )

    if not base_model:
        for section_name in _MODEL_SECTIONS:
            section = config.get(section_name, {})
            if isinstance(section, dict) and section.get("base_model"):
                base_model = section["base_model"]
                break

    if not deploy:
        for section_name in _DEPLOY_SECTIONS:
            section = config.get(section_name, {})
            if isinstance(section, dict) and section:
                deploy = section
                break

    if not deploy and not base_model:
        return []

    family = detect_model_family(base_model)
    system_prompt = domain.identity
    suffix = domain.deployment.get("modelfile_system_suffix", "")
    if suffix:
        system_prompt = f"{system_prompt}\n\n{suffix.strip()}"

    entry = {
        "role": "default",
        "base_model": base_model,
        "model_family": family,
        "format": "gguf",
        "hf_repo": deploy.get("gguf_repo", ""),
        "quantization": deploy.get("quantization", ""),
        "context_length": deploy.get("context_length", 8192),
        "system_prompt": system_prompt,
    }

    chain = deploy.get("adapter_chain", [])
    if chain:
        entry["adapter_chain"] = chain
    source = deploy.get("source_adapter", "")
    if source:
        entry["source_adapter"] = source

    return [entry]


def _role_to_model_entry(
    section: dict, domain, role_name: str,
) -> dict[str, Any]:
    deploy = section.get("deployment", {})
    base_model = section.get("base_model", "")
    family = detect_model_family(base_model)

    fmt = deploy.get("format", "gguf")
    system_prompt = domain.get_role_identity(role_name)
    suffix = domain.deployment.get("modelfile_system_suffix", "")
    if suffix and role_name != "triage":
        system_prompt = f"{system_prompt}\n\n{suffix.strip()}"

    entry: dict[str, Any] = {
        "role": role_name,
        "base_model": base_model,
        "model_family": family,
        "format": fmt,
        "context_length": deploy.get("context_length", 4096),
        "system_prompt": system_prompt,
    }

    if fmt == "gguf":
        entry["hf_repo"] = deploy.get("gguf_repo", "")
        entry["quantization"] = deploy.get("quantization", "")
    elif fmt == "rkllm":
        entry["hf_repo"] = deploy.get("rkllm_repo", "")
        entry["target_platform"] = deploy.get("target_platform", "rk3588")

    chain = deploy.get("adapter_chain", [])
    if chain:
        entry["adapter_chain"] = chain
    source = deploy.get("source_adapter", "")
    if source:
        entry["source_adapter"] = source

    port = deploy.get("port")
    if port:
        entry["port"] = port

    return entry


def main():
    parser = argparse.ArgumentParser(
        description="Generate deployment manifest from pipeline config"
    )
    parser.add_argument("--config", required=True, help="Pipeline YAML path")
    parser.add_argument("--output", default=None,
                        help="Output file (default: stdout)")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = generate_deployment_manifest(config, config_path=args.config)

    content = yaml.dump(manifest, default_flow_style=False, sort_keys=False)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(content)
        print(f"Manifest written to {args.output}")
    else:
        print(content)


if __name__ == "__main__":
    main()
