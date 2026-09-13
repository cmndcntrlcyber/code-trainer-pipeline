"""
phase4_gemma_finetuning/scripts/launch_vision_training.py

Launch Phase 4 Gemma Vision SFT: multimodal training using Gemma 26B's
native SigLIP vision encoder on the v10-vision dataset.

Runs as Job 2b in the pipeline, after text-only SFT (Job 2) and before
validation (Job 3). The vision tower is frozen; the projector and LoRA
on the language model are trained.

Usage:
    set -a && source .env && set +a
    python -m src.phase4_gemma_finetuning.scripts.launch_vision_training \\
        --config src/config/pipeline-gemma26b.yml --wait

    # Dry-run:
    python -m src.phase4_gemma_finetuning.scripts.launch_vision_training \\
        --config src/config/pipeline-gemma26b.yml --dry-run
"""
import argparse
import json
import logging
import os
import sys
import time
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


def build_job_command(repo_url: str, repo_ref: str) -> list[str]:
    script = (
        "set -euo pipefail\n"
        "apt-get update -qq && apt-get install -y -qq git\n"
        f'git clone --depth 1 --branch "{repo_ref}" "{repo_url}" /workspace\n'
        "cd /workspace\n"
        "pip install -q uv\n"
        "uv sync --frozen\n"
        "uv run python -m src.phase4_gemma_finetuning.hf_skills.train_entry\n"
    )
    return ["bash", "-lc", script]


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4 Gemma Vision SFT: multimodal training (HF Jobs)"
    )
    parser.add_argument("--config", default="src/config/pipeline-gemma26b.yml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--train-limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    vis_cfg = config.get("gemma_vision_sft", {})
    if not vis_cfg:
        raise SystemExit("gemma_vision_sft section not found in config")

    cloud_cfg = vis_cfg.get("cloud", {})
    model_id = vis_cfg.get("model", "google/gemma-4-26B-A4B-it")
    dataset_id = vis_cfg.get("dataset_id", "cmndcntrlcyber/code-trainer-v10-vision")
    output_adapter = vis_cfg.get("output_adapter")
    if not output_adapter:
        raise SystemExit("output_adapter required in gemma_vision_sft config")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""
    wandb_key = os.environ.get("WANDB_API_KEY")
    if not args.dry_run and not hf_token:
        raise SystemExit("HF_TOKEN env var required to submit an HF Job")

    train_limit = args.train_limit or vis_cfg.get("train_limit")

    params = {
        "name": "vision_sft",
        "model_id": model_id,
        "dataset_id": dataset_id,
        "dataset_revision": vis_cfg.get("dataset_revision", "main"),
        "num_epochs": int(vis_cfg.get("num_epochs", 1)),
        "max_seq_length": int(vis_cfg.get("max_seq_length", 4096)),
        "lora_r": int(vis_cfg.get("lora_r", 32)),
        "lora_alpha": int(vis_cfg.get("lora_alpha", 64)),
        "learning_rate": float(vis_cfg.get("learning_rate", 2.5e-5)),
        "batch_size": int(vis_cfg.get("batch_size", 1)),
        "gradient_accumulation": int(vis_cfg.get("gradient_accumulation", 16)),
        "adapter_repo": output_adapter,
        "output_dir": "/tmp/phase4-gemma-vision",
        "enable_vision": True,
        "train_projector": vis_cfg.get("train_projector", True),
        "freeze_vision_tower": vis_cfg.get("freeze_vision_tower", True),
    }
    if train_limit:
        params["train_limit"] = int(train_limit)

    env = {
        "PHASE4_PARAMS_JSON": json.dumps(params),
        "PHASE4_ADAPTER_REPO": output_adapter,
        "WANDB_PROJECT": cloud_cfg.get("wandb_project", "rtpi-phase4-gemma26b-vision"),
        "REPO_URL": cloud_cfg.get("repo_url", ""),
        "REPO_REF": cloud_cfg.get("repo_ref", "main"),
    }
    wandb_mode = os.environ.get("WANDB_MODE")
    if wandb_mode:
        env["WANDB_MODE"] = wandb_mode
    elif not wandb_key:
        env["WANDB_MODE"] = "offline"

    secrets = {"HF_TOKEN": hf_token}
    if wandb_key:
        secrets["WANDB_API_KEY"] = wandb_key

    timeout = int(cloud_cfg.get("timeout_seconds", 10800))

    spec = JobSpec(
        image=cloud_cfg.get("image", "huggingface/transformers-pytorch-gpu:latest"),
        command=build_job_command(
            cloud_cfg.get("repo_url", ""),
            cloud_cfg.get("repo_ref", "main"),
        ),
        flavor=cloud_cfg.get("hardware", "a100-large"),
        env=env,
        secrets=secrets,
        timeout_seconds=timeout,
        labels={
            "phase": "4-gemma-vision",
            "project": "rtpi",
            "run": "gemma4-26b-vision-sft",
        },
    )

    s = asdict(spec)
    s["secrets"] = {k: "<redacted>" for k in spec.secrets}
    logger.info("=== Vision SFT Job ===\n%s", json.dumps(s, indent=2, default=str))

    if args.dry_run:
        logger.info("--dry-run set — not submitting.")
        return

    jid = submit_job(spec, token=hf_token)
    print(f"JOB_ID={jid}")

    Path("data/sweep_results").mkdir(parents=True, exist_ok=True)
    Path("data/sweep_results/gemma_vision_sft_job_id.json").write_text(
        json.dumps({"job_id": jid, "adapter": output_adapter}, indent=2)
    )

    if not args.wait:
        return

    poll_interval = int(cloud_cfg.get("poll_interval", 60))
    stage = wait_for_job(jid, hf_token, poll_interval, timeout)
    logger.info("Vision SFT final stage: %s", stage)

    if stage != "COMPLETED":
        logger.error("Vision SFT job did not complete: %s", stage)
        sys.exit(1)
    logger.info("Gemma Vision SFT job completed.")


if __name__ == "__main__":
    main()
