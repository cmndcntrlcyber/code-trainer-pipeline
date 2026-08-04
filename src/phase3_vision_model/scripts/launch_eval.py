"""
phase3_vision_model/scripts/launch_eval.py

Submit the Phase 3 baseline-vs-finetuned eval job to HF Jobs A100.

Mirrors launch_vision_training.py but invokes eval_entry instead of train_entry,
and runs against the already-published adapter at PHASE3_ADAPTER_REPO.

Usage:
    export HF_TOKEN=hf_...
    python -m src.phase3_vision_model.scripts.launch_eval \
        --config src/config/config.yaml --wait

    # Dry-run (print JobSpec, don't submit):
    python -m src.phase3_vision_model.scripts.launch_eval --dry-run

    # Tweak sample count / split:
    python -m src.phase3_vision_model.scripts.launch_eval --num-samples 500 --split test
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


def build_eval_command(repo_url: str, repo_ref: str, limit: int | None = None) -> list[str]:
    """Clone repo + uv sync --frozen + run eval_entry inside the container."""
    entry = "uv run python -m src.phase3_vision_model.hf_skills.eval_entry"
    if limit is not None:
        entry += f" --limit {int(limit)}"
    script = (
        "set -euo pipefail\n"
        "apt-get update -qq && apt-get install -y -qq git\n"
        f'git clone --depth 1 --branch "{repo_ref}" "{repo_url}" /workspace\n'
        "cd /workspace\n"
        "pip install -q uv\n"
        "uv sync --frozen\n"
        f"{entry}\n"
    )
    return ["bash", "-lc", script]


def main():
    parser = argparse.ArgumentParser(description="Launch Phase 3 A100 eval job (HF Jobs)")
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the job spec without submitting")
    parser.add_argument("--wait", action="store_true",
                        help="Block until job completes")
    parser.add_argument("--split", default="test", choices=["test", "validation"])
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap each dataset split to first N rows before evaluation. "
                             "Useful for cheap cloud smoke-tests of pipeline changes.")
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
        max_seq_length=vm_cfg.get("max_seq_length", 2048),
        dataset_id=vm_cfg.get("dataset_id", A100VisionConfig.dataset_id),
        dataset_revision=vm_cfg.get("dataset_revision", "v2-multimodal"),
    )

    hardware = cloud_cfg.get("hardware", "a100-large")
    adapter_repo = cloud_cfg.get("adapter_repo")
    image = cloud_cfg.get("image", "huggingface/transformers-pytorch-gpu:latest")
    repo_url = cloud_cfg.get("repo_url") or ""
    repo_ref = cloud_cfg.get("repo_ref", "main")
    timeout_seconds = int(cloud_cfg.get("eval_timeout_seconds", 7200))  # 2h cap
    poll_interval = int(cloud_cfg.get("poll_interval", 60))

    if not adapter_repo:
        raise SystemExit("vision_model.cloud.adapter_repo not set (check HF_USERNAME env var)")
    if not repo_url:
        raise SystemExit("vision_model.cloud.repo_url not set")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""
    if not args.dry_run and not hf_token:
        raise SystemExit("HF_TOKEN env var required to submit an HF Job")

    hyperparams = a100_cfg.to_hyperparams()
    hyperparams["adapter_repo"] = adapter_repo
    env = {
        "PHASE3_PARAMS_JSON": json.dumps(hyperparams),
        "PHASE3_ADAPTER_REPO": adapter_repo,
        "REPO_URL": repo_url,
        "REPO_REF": repo_ref,
        "EVAL_NUM_SAMPLES": str(args.num_samples),
        "EVAL_SPLIT": args.split,
    }
    wandb_mode = os.environ.get("WANDB_MODE")
    if wandb_mode:
        env["WANDB_MODE"] = wandb_mode
    elif not os.environ.get("WANDB_API_KEY"):
        env["WANDB_MODE"] = "offline"
    secrets = {"HF_TOKEN": hf_token}

    spec = VisionJobSpec(
        image=image,
        command=build_eval_command(repo_url, repo_ref, limit=args.limit),
        flavor=hardware,
        env=env,
        secrets=secrets,
        timeout_seconds=timeout_seconds,
        labels={"phase": "3", "project": "rtpi", "run": "vision-eval"},
    )

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
