"""
phase4_qwen_finetuning/scripts/launch_v9_training.py

Launch V9 training with curriculum (two-phase) on HF Jobs.

Uses train_entry_v9.py instead of train_entry.py for the container command.
Defaults to the v9_mixed sweep config.

Usage:
    set -a && source .env && set +a
    python -m src.phase4_qwen_finetuning.scripts.launch_v9_training \\
        --config src/config/config.yaml --wait

    # Dry-run:
    python -m src.phase4_qwen_finetuning.scripts.launch_v9_training --dry-run

    # With row limits (for fitting under HF Skills timeout):
    python -m src.phase4_qwen_finetuning.scripts.launch_v9_training \\
        --train-limit 20000 --wait
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
from src.phase4_qwen_finetuning.configs.sweep_configs import SWEEP_CONFIG_MAP
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
        "uv run python -m src.phase4_qwen_finetuning.hf_skills.train_entry_v9\n"
    )
    return ["bash", "-lc", script]


def main():
    parser = argparse.ArgumentParser(description="Launch V9 curriculum training (HF Jobs)")
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--only", default="v9_mixed",
                        help="Sweep config name to use (default: v9_mixed)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    qf_cfg = config.get("qwen_finetuning", {})
    cloud_cfg = qf_cfg.get("cloud", {})

    if args.only not in SWEEP_CONFIG_MAP:
        raise SystemExit(f"Unknown config: {args.only}. Available: {list(SWEEP_CONFIG_MAP.keys())}")

    cfg = SWEEP_CONFIG_MAP[args.only]
    hf_username = os.environ.get("HF_USERNAME", "cmndcntrlcyber")
    adapter_repo = f"{hf_username}/qwen14b-code-trainer-{cfg.name}"

    params = {
        "name": cfg.name,
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "learning_rate": cfg.learning_rate,
        "batch_size": cfg.batch_size,
        "gradient_accumulation": cfg.gradient_accumulation,
        "model_id": qf_cfg.get("model", "Qwen/Qwen2.5-Coder-14B-Instruct"),
        "dataset_id": cfg.dataset_id or qf_cfg.get("dataset_id"),
        "dataset_revision": cfg.dataset_revision or "main",
        "num_epochs": args.num_epochs,
        "max_seq_length": cfg.max_seq_length if cfg.max_seq_length else int(qf_cfg.get("max_seq_length", 2048)),
        "adapter_repo": adapter_repo,
        "output_dir": f"/tmp/phase4-v9-{cfg.name}",
    }
    dapt_adapter = qf_cfg.get("dapt_adapter")
    if dapt_adapter:
        params["dapt_adapter"] = dapt_adapter
    if args.train_limit is not None:
        params["train_limit"] = args.train_limit
    if args.val_limit is not None:
        params["val_limit"] = args.val_limit

    env = {
        "PHASE4_PARAMS_JSON": json.dumps(params),
        "PHASE4_ADAPTER_REPO": adapter_repo,
        "WANDB_PROJECT": cloud_cfg.get("wandb_project", "rtpi-phase4-qwen14b"),
        "REPO_URL": cloud_cfg.get("repo_url", ""),
        "REPO_REF": cloud_cfg.get("repo_ref", "main"),
    }
    wandb_key = os.environ.get("WANDB_API_KEY")
    wandb_mode = os.environ.get("WANDB_MODE")
    if wandb_mode:
        env["WANDB_MODE"] = wandb_mode
    elif not wandb_key:
        env["WANDB_MODE"] = "offline"

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""
    if not args.dry_run and not hf_token:
        raise SystemExit("HF_TOKEN env var required to submit an HF Job")

    secrets = {"HF_TOKEN": hf_token}
    if wandb_key:
        secrets["WANDB_API_KEY"] = wandb_key

    timeout = int(cloud_cfg.get("timeout_seconds", 50400))

    spec = JobSpec(
        image=cloud_cfg.get("image", "huggingface/transformers-pytorch-gpu:latest"),
        command=build_job_command(cloud_cfg.get("repo_url", ""), cloud_cfg.get("repo_ref", "main")),
        flavor=cloud_cfg.get("hardware", qf_cfg.get("hardware", "a100-large")),
        env=env,
        secrets=secrets,
        timeout_seconds=timeout,
        labels={"phase": "4-v9", "project": "rtpi", "run": f"qwen14b-{cfg.name}-v9"},
    )

    s = asdict(spec)
    s["secrets"] = {k: "<redacted>" for k in spec.secrets}
    logger.info("=== V9 Training Job ===\n%s", json.dumps(s, indent=2, default=str))

    if args.dry_run:
        logger.info("--dry-run set — not submitting.")
        return

    job_id = submit_job(spec, token=hf_token)
    print(f"JOB_ID={job_id}")

    Path("data/v9_validation").mkdir(parents=True, exist_ok=True)
    Path("data/v9_validation/job_ids.json").write_text(
        json.dumps({cfg.name: job_id}, indent=2)
    )

    if not args.wait:
        return

    poll_interval = int(cloud_cfg.get("poll_interval", 60))
    stage = wait_for_job(job_id, hf_token, poll_interval, timeout)
    logger.info(f"Final stage: {stage}")

    if stage != "COMPLETED":
        logger.error("V9 training job did not complete: %s", stage)
        sys.exit(1)
    logger.info("V9 training job completed successfully.")


if __name__ == "__main__":
    main()
