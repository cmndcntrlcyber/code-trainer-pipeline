"""
Ollama Modelfile generator.

Produces a Modelfile for each trained role from the pipeline config
and domain definition. Handles chat template selection by model family,
system prompt injection from the domain, and per-role parameters.

Usage:
    python -m src.deployment.modelfile_gen \
        --config src/config/pipeline-gemma26b.yml \
        --role default \
        --output models/nexus/Modelfile
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from src.config.domain_loader import get_domain, DomainConfig
from src.config.settings import load_config
from src.deployment.chat_templates import detect_model_family, get_template


def generate_modelfile(
    role_config: dict[str, Any],
    domain_config: DomainConfig,
    role: str | None = None,
    *,
    model_family: str | None = None,
) -> str:
    """Generate an Ollama Modelfile for a trained model.

    Args:
        role_config: The role's config section from the pipeline YAML
                     (must include ``deployment`` and ``base_model``).
        domain_config: Loaded domain definition.
        role: Swarm role name (None for single-model deployment).
        model_family: Override auto-detected model family.

    Returns:
        Complete Modelfile content as a string.
    """
    deploy = role_config.get("deployment", {})
    base_model = role_config.get("base_model", "")

    # Resolve the FROM line.
    gguf_repo = deploy.get("gguf_repo", "")
    quant = deploy.get("quantization", "Q4_K_M")
    from_line = f"hf.co/{gguf_repo}:{quant}" if gguf_repo else base_model

    # Resolve chat template.
    family = model_family or detect_model_family(base_model)
    tmpl = get_template(family)

    # Build system prompt from domain + role.
    system_prompt = domain_config.identity
    if role:
        role_identity = domain_config.get_role_identity(role)
        system_prompt = role_identity

    suffix = domain_config.deployment.get("modelfile_system_suffix", "")
    if suffix:
        system_prompt = f"{system_prompt}\n\n{suffix.strip()}"

    # Build parameter block.
    ctx = deploy.get("context_length", 8192)
    params = [
        f"PARAMETER num_ctx {ctx}",
        "PARAMETER num_gpu -1",
        "PARAMETER temperature 0.7",
        "PARAMETER top_p 0.9",
        "PARAMETER repeat_penalty 1.1",
    ]

    for stop in tmpl["stop_tokens"]:
        params.append(f'PARAMETER stop "{stop}"')

    gpu_layers = deploy.get("num_gpu")
    if gpu_layers is not None:
        params[-5] = f"PARAMETER num_gpu {gpu_layers}"

    lines = [
        f"FROM {from_line}",
        "",
        f'TEMPLATE """{tmpl["template"]}"""',
        "",
        f'SYSTEM """{system_prompt}"""',
        "",
        *params,
    ]

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Generate an Ollama Modelfile from pipeline config"
    )
    parser.add_argument("--config", required=True, help="Pipeline YAML path")
    parser.add_argument("--role", default=None,
                        help="Swarm role (omit for single-model)")
    parser.add_argument("--output", default=None,
                        help="Output file (default: stdout)")
    args = parser.parse_args()

    config = load_config(args.config)
    domain_name = config.get("domain", "offsec")
    domain_config = get_domain(domain_name)

    if args.role and args.role != "default":
        role_config = config.get(args.role, {})
        if not role_config:
            print(f"Role {args.role!r} not found in config", file=sys.stderr)
            sys.exit(1)
        content = generate_modelfile(role_config, domain_config, role=args.role)
    else:
        # Single-model: look for top-level deployment section or gemma/qwen section.
        role_config = config
        for section in ("gemma_finetuning", "qwen_finetuning", "sft"):
            if section in config:
                role_config = {**config, **config[section]}
                break
        content = generate_modelfile(role_config, domain_config)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(content)
        print(f"Modelfile written to {args.output}")
    else:
        print(content)


if __name__ == "__main__":
    main()
