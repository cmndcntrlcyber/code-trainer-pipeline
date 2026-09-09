"""
phase4c_rl/scripts/launch_farca_grpo.py

Submit Phase 4c FARCA-GRPO training to HF Jobs A100.
Reads model from config -- works for both Qwen and Gemma pipelines.

Usage:
    set -a && source .env && set +a

    # Qwen pipeline:
    python -m src.phase4c_rl.scripts.launch_farca_grpo \
        --config src/config/pipeline-50.yml --wait

    # Gemma pipeline:
    python -m src.phase4c_rl.scripts.launch_farca_grpo \
        --config src/config/pipeline-gemma26b.yml --wait
"""
import argparse
import json
import logging
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.config.settings import load_config
from src.phase3_vision_model.hf_skills import (
    VisionJobSpec as JobSpec,
    submit_vision_job as submit_job,
    wait_for_job,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_command(repo_url: str, repo_ref: str) -> list[str]:
    script = (
        "set -euo pipefail\n"
        "apt-get update -qq && apt-get install -y -qq git\n"
        f'git clone --depth 1 --branch "{repo_ref}" "{repo_url}" /workspace/repo\n'
        "cd /workspace/repo\n"
        "pip install -q uv\n"
        "uv sync --frozen\n"
        "uv run python -m src.phase4c_rl.hf_skills.farca_grpo_entry\n"
    )
    return ["bash", "-lc", script]


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4c FARCA-GRPO Training (HF Jobs)"
    )
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--adapter", default=None,
                        help="Override rl_training.farca_grpo.base_adapter")
    args = parser.parse_args()

    config = load_config(args.config)
    rl_cfg = config.get("rl_training", {})
    farca_cfg = rl_cfg.get("farca_grpo", {})
    cloud_cfg = rl_cfg.get("cloud", {})

    if not farca_cfg:
        raise SystemExit(
            "rl_training.farca_grpo not found in config. "
            "Add FARCA-GRPO config section to your pipeline YAML."
        )

    base_adapter = args.adapter or farca_cfg.get("base_adapter")
    base_model = farca_cfg.get("base_model", "Qwen/Qwen2.5-Coder-14B-Instruct")
    prompt_dataset = farca_cfg.get("prompt_dataset")
    output_adapter = farca_cfg.get("output_adapter")

    if not base_adapter:
        raise SystemExit("rl_training.farca_grpo.base_adapter not set")
    if not output_adapter:
        raise SystemExit("rl_training.farca_grpo.output_adapter not set")
    if not prompt_dataset:
        raise SystemExit("rl_training.farca_grpo.prompt_dataset not set")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""
    if not args.dry_run and not hf_token:
        raise SystemExit("HF_TOKEN env var required")

    dapt_adapter = farca_cfg.get("dapt_adapter")

    params = {
        "base_model": base_model,
        "base_adapter": base_adapter,
        "prompt_dataset": prompt_dataset,
        "output_adapter": output_adapter,
        "learning_rate": float(farca_cfg.get("learning_rate", 5e-7)),
        "num_generations": int(farca_cfg.get("num_generations", 4)),
        "beta": float(farca_cfg.get("kl_coef", 0.1)),
        "max_new_tokens": int(farca_cfg.get("max_new_tokens", 1024)),
        "num_epochs": 1,
        "batch_size": 2,
        "gradient_accumulation": 4,
        # FARCA-specific params passed through to entry point
        "claim_extraction_backend": farca_cfg.get("claim_extraction_backend", "rule_based"),
        "nli_model_id": farca_cfg.get("nli_model_id", "vectara/hallucination_evaluation_model"),
        "sentence_encoder_id": farca_cfg.get("sentence_encoder_id", "all-MiniLM-L6-v2"),
        "k_rel": int(farca_cfg.get("k_rel", 1)),
        "mu": float(farca_cfg.get("mu", 0.16)),
        "tau": float(farca_cfg.get("tau", 0.20)),
        "nli_weight": float(farca_cfg.get("nli_weight", 0.4)),
        "rule_weight": float(farca_cfg.get("rule_weight", 0.6)),
        "warmup_steps": int(farca_cfg.get("warmup_steps", 0)),
    }
    if dapt_adapter:
        params["dapt_adapter"] = dapt_adapter

    env = {
        "PHASE4C_FARCA_PARAMS_JSON": json.dumps(params),
        "WANDB_PROJECT": cloud_cfg.get("wandb_project", "rtpi-phase4c-rl"),
        "REPO_URL": cloud_cfg.get("repo_url", ""),
        "REPO_REF": cloud_cfg.get("repo_ref", "main"),
    }

    wandb_key = os.environ.get("WANDB_API_KEY")
    wandb_mode = os.environ.get("WANDB_MODE")
    if wandb_mode:
        env["WANDB_MODE"] = wandb_mode
    elif not wandb_key:
        env["WANDB_MODE"] = "offline"

    secrets = {"HF_TOKEN": hf_token}
    if wandb_key:
        secrets["WANDB_API_KEY"] = wandb_key

    label_slug = re.sub(
        r"[^A-Za-z0-9_=-]+", "-", base_model.split("/")[-1]
    ).strip("-")

    timeout = int(cloud_cfg.get("timeout_seconds", 36000))

    spec = JobSpec(
        image=cloud_cfg.get("image", "huggingface/transformers-pytorch-gpu:latest"),
        command=build_command(
            cloud_cfg.get("repo_url", ""), cloud_cfg.get("repo_ref", "main")
        ),
        flavor=cloud_cfg.get("hardware", "a100-large"),
        env=env,
        secrets=secrets,
        timeout_seconds=timeout,
        labels={"phase": "4c-farca-grpo", "project": "rtpi", "run": f"farca-{label_slug}"},
    )

    s = asdict(spec)
    s["secrets"] = {k: "<redacted>" for k in spec.secrets}
    logger.info("=== FARCA-GRPO Job ===\n%s", json.dumps(s, indent=2, default=str))

    if args.dry_run:
        logger.info("--dry-run set -- not submitting.")
        return

    job_id = submit_job(spec, token=hf_token)
    print(f"JOB_ID={job_id}")

    Path("data/rl_training").mkdir(parents=True, exist_ok=True)
    Path("data/rl_training/farca_grpo_job_ids.json").write_text(
        json.dumps({"farca_grpo": job_id}, indent=2)
    )

    if not args.wait:
        return

    poll_interval = int(cloud_cfg.get("poll_interval", 60))
    stage = wait_for_job(job_id, hf_token, poll_interval, timeout)
    logger.info(f"Final stage: {stage}")

    if stage != "COMPLETED":
        logger.error("FARCA-GRPO training job did not complete: %s", stage)
        sys.exit(1)
    logger.info("FARCA-GRPO training job completed successfully.")


if __name__ == "__main__":
    main()
