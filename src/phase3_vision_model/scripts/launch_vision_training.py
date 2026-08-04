"""
phase3_vision_model/scripts/launch_vision_training.py

Submit the Phase 3 vision-model training job to HF Jobs on A100-large hardware.

Usage:
    export HF_TOKEN=hf_...
    export WANDB_API_KEY=...
    python -m src.phase3_vision_model.scripts.launch_vision_training \
        --config src/config/config.yaml

    # Dry-run (print JobSpec, don't submit):
    python -m src.phase3_vision_model.scripts.launch_vision_training --dry-run

    # Block until completion:
    python -m src.phase3_vision_model.scripts.launch_vision_training --wait
"""
import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.config.settings import load_config
from src.phase3_vision_model.configs.a100_config import A100VisionConfig
from src.phase3_vision_model.hf_skills import (
    VisionJobSpec,
    submit_vision_job,
    wait_for_job,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_job_command(repo_url: str, repo_ref: str) -> list[str]:
    """Build the shell command that clones the repo, syncs deps, and runs train_entry.

    `uv sync --frozen` requires that `uv.lock` is tracked in the repo; we now
    track it so the cloud job installs the exact resolved torch+cu128 wheels
    that were validated locally, instead of re-resolving and risking a
    container-driver mismatch.
    """
    script = (
        "set -euo pipefail\n"
        "apt-get update -qq && apt-get install -y -qq git\n"
        f'git clone --depth 1 --branch "{repo_ref}" "{repo_url}" /workspace\n'
        "cd /workspace\n"
        "pip install -q uv\n"
        "uv sync --frozen\n"
        "uv run python -m src.phase3_vision_model.hf_skills.train_entry\n"
    )
    return ["bash", "-lc", script]


def build_job_spec(
    cfg: A100VisionConfig,
    hardware: str,
    adapter_repo: str,
    image: str,
    repo_url: str,
    repo_ref: str,
    wandb_project: str,
    timeout_seconds: int,
    hf_token: str,
    wandb_key: str | None,
) -> VisionJobSpec:
    hyperparams = cfg.to_hyperparams()
    hyperparams["adapter_repo"] = adapter_repo
    env = {
        "PHASE3_PARAMS_JSON": json.dumps(hyperparams),
        "PHASE3_ADAPTER_REPO": adapter_repo,
        "WANDB_PROJECT": wandb_project,
        "REPO_URL": repo_url,
        "REPO_REF": repo_ref,
    }
    wandb_mode = os.environ.get("WANDB_MODE")
    if wandb_mode:
        env["WANDB_MODE"] = wandb_mode
    elif not wandb_key:
        env["WANDB_MODE"] = "offline"
    secrets = {"HF_TOKEN": hf_token}
    if wandb_key:
        secrets["WANDB_API_KEY"] = wandb_key
    return VisionJobSpec(
        image=image,
        command=build_job_command(repo_url, repo_ref),
        flavor=hardware,
        env=env,
        secrets=secrets,
        timeout_seconds=timeout_seconds,
        labels={"phase": "3", "project": "rtpi", "run": f"vision-{cfg.name}"},
    )


def main():
    parser = argparse.ArgumentParser(description="Launch Phase 3 A100 vision training (HF Jobs)")
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the job spec without submitting")
    parser.add_argument("--wait", action="store_true",
                        help="Block until job completes")
    args = parser.parse_args()

    config = load_config(args.config)
    vm_cfg = config.get("vision_model", {})
    cloud_cfg = vm_cfg.get("cloud", {})

    a100_cfg = A100VisionConfig(
        vision_encoder=vm_cfg.get("vision_encoder", A100VisionConfig.vision_encoder),
        decoder=vm_cfg.get("decoder", A100VisionConfig.decoder),
        lora_r=vm_cfg.get("lora_r", 16),
        lora_alpha=vm_cfg.get("lora_alpha", 32),
        lora_dropout=vm_cfg.get("lora_dropout", 0.05),
        batch_size=vm_cfg.get("batch_size", 8),
        gradient_accumulation=vm_cfg.get("gradient_accumulation", 4),
        learning_rate=float(vm_cfg.get("learning_rate", 2e-4)),
        num_epochs=vm_cfg.get("num_epochs", 3),
        max_seq_length=vm_cfg.get("max_seq_length", 2048),
        dataset_id=vm_cfg.get("dataset_id", A100VisionConfig.dataset_id),
        dataset_revision=vm_cfg.get("dataset_revision", "v2-multimodal"),
    )

    hardware = cloud_cfg.get("hardware", "a100-large")
    adapter_repo = cloud_cfg.get("adapter_repo")
    image = cloud_cfg.get("image", "huggingface/transformers-pytorch-gpu:latest")
    repo_url = cloud_cfg.get("repo_url") or ""
    repo_ref = cloud_cfg.get("repo_ref", "main")
    wandb_project = cloud_cfg.get("wandb_project", "rtpi-phase3-vision")
    timeout_seconds = int(cloud_cfg.get("timeout_seconds", 36000))
    poll_interval = int(cloud_cfg.get("poll_interval", 60))

    if not adapter_repo:
        raise SystemExit("vision_model.cloud.adapter_repo not set (check HF_USERNAME env var)")
    if not repo_url:
        raise SystemExit("vision_model.cloud.repo_url not set")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""
    wandb_key = os.environ.get("WANDB_API_KEY")

    if not args.dry_run and not hf_token:
        raise SystemExit("HF_TOKEN env var required to submit an HF Job")

    spec = build_job_spec(
        cfg=a100_cfg,
        hardware=hardware,
        adapter_repo=adapter_repo,
        image=image,
        repo_url=repo_url,
        repo_ref=repo_ref,
        wandb_project=wandb_project,
        timeout_seconds=timeout_seconds,
        hf_token=hf_token,
        wandb_key=wandb_key,
    )

    # Print spec (secrets redacted)
    spec_display = asdict(spec)
    spec_display["secrets"] = {k: "<redacted>" for k in spec.secrets}
    logger.info("Job spec:\n%s", json.dumps(spec_display, indent=2, default=str))

    if args.dry_run:
        logger.info("--dry-run set — not submitting.")
        return

    job_id = submit_vision_job(spec, token=hf_token)
    print(f"JOB_ID={job_id}")

    if args.wait:
        final = wait_for_job(
            job_id=job_id,
            token=hf_token,
            poll_interval=poll_interval,
            timeout=timeout_seconds,
        )
        logger.info(f"Final stage: {final}")
        if final != "COMPLETED":
            sys.exit(1)


if __name__ == "__main__":
    main()
