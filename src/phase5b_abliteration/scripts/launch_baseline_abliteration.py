"""
phase5b_abliteration/scripts/launch_baseline_abliteration.py

Submit pre-fine-tuning abliteration baselines to a single HF Jobs A100.
Processes all runs in abliteration_baselines.runs sequentially (e.g.,
Qwen base then Gemma base) within one job.

Usage:
    set -a && source .env && set +a
    python -m src.phase5b_abliteration.scripts.launch_baseline_abliteration \
        --config src/config/config.yaml --wait

    # Dry-run:
    python -m src.phase5b_abliteration.scripts.launch_baseline_abliteration --dry-run
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
        "apt-get update -qq && apt-get install -y -qq git cmake build-essential\n"
        f'git clone --depth 1 --branch "{repo_ref}" "{repo_url}" /workspace/repo\n'
        "cd /workspace/repo\n"
        "pip install -q uv\n"
        "uv sync --frozen\n"
        "uv run python -m src.phase5b_abliteration.hf_skills.baseline_entry\n"
    )
    return ["bash", "-lc", script]


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5b Baseline Abliteration (HF Jobs)"
    )
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    baselines_cfg = config.get("abliteration_baselines")
    if not baselines_cfg:
        raise SystemExit("abliteration_baselines section not found in config")

    runs = baselines_cfg.get("runs", [])
    if not runs:
        raise SystemExit("abliteration_baselines.runs is empty")

    cloud_cfg = baselines_cfg.get("cloud", {})
    techniques = baselines_cfg.get("techniques", [])
    eval_cfg = baselines_cfg.get("evaluation", {})
    quants = cloud_cfg.get("quants", ["Q4_K_M"])

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""
    if not args.dry_run and not hf_token:
        raise SystemExit("HF_TOKEN env var required")

    params = {
        "runs": runs,
        "techniques": techniques,
        "evaluation": eval_cfg,
        "quants": list(quants),
    }

    env = {
        "PHASE5B_BASELINE_PARAMS_JSON": json.dumps(params),
        "REPO_URL": cloud_cfg.get("repo_url", ""),
        "REPO_REF": cloud_cfg.get("repo_ref", "main"),
    }
    secrets = {"HF_TOKEN": hf_token}

    run_names = [r.get("name", "unknown") for r in runs]
    logger.info("Baseline abliteration runs: %s (sequential in one job)", run_names)

    spec = JobSpec(
        image=cloud_cfg.get("image", "huggingface/transformers-pytorch-gpu:latest"),
        command=build_command(
            cloud_cfg.get("repo_url", ""), cloud_cfg.get("repo_ref", "main")
        ),
        flavor=cloud_cfg.get("hardware", "a100-large"),
        env=env,
        secrets=secrets,
        timeout_seconds=int(cloud_cfg.get("timeout_seconds", 43200)),
        labels={
            "phase": "5b-baseline",
            "project": "rtpi",
            "run": "abliterate-baselines",
        },
    )

    s = asdict(spec)
    s["secrets"] = {k: "<redacted>" for k in spec.secrets}
    logger.info("Job spec:\n%s", json.dumps(s, indent=2, default=str))

    if args.dry_run:
        logger.info("--dry-run set — not submitting.")
        return

    job_id = submit_job(spec, token=hf_token)
    print(f"JOB_ID={job_id}")

    if args.wait:
        final = wait_for_job(
            job_id=job_id,
            token=hf_token,
            poll_interval=int(cloud_cfg.get("poll_interval", 60)),
            timeout=int(cloud_cfg.get("wait_timeout_seconds",
                                       cloud_cfg.get("timeout_seconds", 43200))),
        )
        logger.info(f"Final stage: {final}")
        if final != "COMPLETED":
            sys.exit(1)


if __name__ == "__main__":
    main()
