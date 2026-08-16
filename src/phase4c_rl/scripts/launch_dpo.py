"""
phase4c_rl/scripts/launch_dpo.py

Submit Phase 4c DPO training to HF Jobs A100. Follows the same pattern
as launch_abliteration.py and launch_v9_training.py.

Usage:
    set -a && source .env && set +a
    python -m src.phase4c_rl.scripts.launch_dpo \
        --config src/config/config.yaml --wait

    # Dry-run:
    python -m src.phase4c_rl.scripts.launch_dpo --dry-run
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
        "uv run python -m src.phase4c_rl.hf_skills.dpo_entry\n"
    )
    return ["bash", "-lc", script]


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4c DPO Training (HF Jobs)"
    )
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--adapter", default=None,
                        help="Override rl_training.dpo.base_adapter")
    args = parser.parse_args()

    config = load_config(args.config)
    rl_cfg = config.get("rl_training", {})
    dpo_cfg = rl_cfg.get("dpo", {})
    cloud_cfg = rl_cfg.get("cloud", {})

    base_adapter = args.adapter or dpo_cfg.get("base_adapter")
    base_model = dpo_cfg.get("base_model", "Qwen/Qwen2.5-Coder-14B-Instruct")
    preference_dataset = dpo_cfg.get("preference_dataset")
    output_adapter = dpo_cfg.get("output_adapter")

    if not base_adapter:
        raise SystemExit("rl_training.dpo.base_adapter not set (or pass --adapter)")
    if not output_adapter:
        raise SystemExit("rl_training.dpo.output_adapter not set in config")
    if not preference_dataset:
        raise SystemExit("rl_training.dpo.preference_dataset not set in config")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""
    if not args.dry_run and not hf_token:
        raise SystemExit("HF_TOKEN env var required")

    params = {
        "base_model": base_model,
        "base_adapter": base_adapter,
        "preference_dataset": preference_dataset,
        "output_adapter": output_adapter,
        "learning_rate": float(dpo_cfg.get("learning_rate", 1e-6)),
        "beta": float(dpo_cfg.get("beta", 0.1)),
        "max_length": int(dpo_cfg.get("max_length", 4096)),
        "num_epochs": 1,
        "batch_size": 2,
        "gradient_accumulation": 4,
    }

    env = {
        "PHASE4C_DPO_PARAMS_JSON": json.dumps(params),
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
        labels={"phase": "4c-dpo", "project": "rtpi", "run": f"dpo-{label_slug}"},
    )

    s = asdict(spec)
    s["secrets"] = {k: "<redacted>" for k in spec.secrets}
    logger.info("=== DPO Job ===\n%s", json.dumps(s, indent=2, default=str))

    if args.dry_run:
        logger.info("--dry-run set -- not submitting.")
        return

    job_id = submit_job(spec, token=hf_token)
    print(f"JOB_ID={job_id}")

    Path("data/rl_training").mkdir(parents=True, exist_ok=True)
    Path("data/rl_training/dpo_job_ids.json").write_text(
        json.dumps({"dpo": job_id}, indent=2)
    )

    if not args.wait:
        return

    poll_interval = int(cloud_cfg.get("poll_interval", 60))
    stage = wait_for_job(job_id, hf_token, poll_interval, timeout)
    logger.info(f"Final stage: {stage}")

    if stage != "COMPLETED":
        logger.error("DPO training job did not complete: %s", stage)
        sys.exit(1)
    logger.info("DPO training job completed successfully.")


if __name__ == "__main__":
    main()
