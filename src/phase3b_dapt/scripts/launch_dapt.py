"""
phase3b_dapt/scripts/launch_dapt.py

Submit Phase 3b DAPT (Domain-Adaptive Pre-Training) to HF Jobs A100.
Follows the same pattern as Phase 5b's launch_abliteration.py.

Usage:
    set -a && source .env && set +a
    python -m src.phase3b_dapt.scripts.launch_dapt \
        --config src/config/config.yaml --wait

    # Dry-run:
    python -m src.phase3b_dapt.scripts.launch_dapt --dry-run
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
        "uv run python -m src.phase3b_dapt.hf_skills.dapt_entry\n"
    )
    return ["bash", "-lc", script]


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3b DAPT -- Domain-Adaptive Pre-Training (HF Jobs)"
    )
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    dapt_cfg = config.get("dapt", {})
    cloud_cfg = dapt_cfg.get("cloud", {})

    base_model = dapt_cfg.get("base_model", "Qwen/Qwen2.5-Coder-14B-Instruct")
    output_adapter = dapt_cfg.get("output_adapter")
    if not output_adapter:
        raise SystemExit("dapt.output_adapter not set in config")

    # The dataset must already be on Hub (prepared by prepare_corpus.py --push-to-hub).
    # Convention: adapter repo name with "-corpus" suffix.
    dataset_id = output_adapter.replace("-adapter", "") + "-corpus"

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""
    if not args.dry_run and not hf_token:
        raise SystemExit("HF_TOKEN env var required")

    params = {
        "base_model": base_model,
        "dataset_id": dataset_id,
        "lora_r": int(dapt_cfg.get("lora_r", 32)),
        "lora_alpha": int(dapt_cfg.get("lora_alpha", 64)),
        "lora_dropout": float(dapt_cfg.get("lora_dropout", 0.05)),
        "learning_rate": float(dapt_cfg.get("learning_rate", 5e-5)),
        "num_epochs": int(dapt_cfg.get("num_epochs", 1)),
        "max_seq_length": int(dapt_cfg.get("max_seq_length", 4096)),
        "output_adapter": output_adapter,
        "wandb_project": cloud_cfg.get("wandb_project", "rtpi-phase3b-dapt"),
    }

    env = {
        "PHASE3B_DAPT_PARAMS_JSON": json.dumps(params),
        "REPO_URL": cloud_cfg.get("repo_url", ""),
        "REPO_REF": cloud_cfg.get("repo_ref", "main"),
    }
    secrets = {"HF_TOKEN": hf_token}

    label_slug = re.sub(
        r"[^A-Za-z0-9_=-]+", "-", base_model.split("/")[-1]
    ).strip("-")

    spec = JobSpec(
        image=cloud_cfg.get("image", "huggingface/transformers-pytorch-gpu:latest"),
        command=build_command(
            cloud_cfg.get("repo_url", ""), cloud_cfg.get("repo_ref", "main")
        ),
        flavor=cloud_cfg.get("hardware", "a100-large"),
        env=env,
        secrets=secrets,
        timeout_seconds=int(cloud_cfg.get("timeout_seconds", 36000)),
        labels={"phase": "3b", "project": "rtpi", "run": f"dapt-{label_slug}"},
    )

    s = asdict(spec)
    s["secrets"] = {k: "<redacted>" for k in spec.secrets}
    logger.info("Job spec:\n%s", json.dumps(s, indent=2, default=str))

    if args.dry_run:
        logger.info("--dry-run set -- not submitting.")
        return

    job_id = submit_job(spec, token=hf_token)
    print(f"JOB_ID={job_id}")

    if args.wait:
        final = wait_for_job(
            job_id=job_id,
            token=hf_token,
            poll_interval=int(cloud_cfg.get("poll_interval", 60)),
            timeout=int(cloud_cfg.get("timeout_seconds", 36000)),
        )
        logger.info(f"Final stage: {final}")
        if final != "COMPLETED":
            sys.exit(1)


if __name__ == "__main__":
    main()
