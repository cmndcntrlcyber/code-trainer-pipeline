"""
phase4_gemma_finetuning/scripts/launch_eval.py

Launch a Phase 4 Gemma eval-only A100 job.

Usage:
    set -a && source .env && set +a
    python -m src.phase4_gemma_finetuning.scripts.launch_eval \
        --adapter cmndcntrlcyber/gemma4-12b-code-trainer-aggressive --wait
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


def build_eval_command(repo_url: str, repo_ref: str) -> list[str]:
    script = (
        "set -euo pipefail\n"
        "apt-get update -qq && apt-get install -y -qq git\n"
        f'git clone --depth 1 --branch "{repo_ref}" "{repo_url}" /workspace\n'
        "cd /workspace\n"
        "pip install -q uv\n"
        "uv sync --frozen\n"
        "uv run python -m src.phase4_gemma_finetuning.hf_skills.eval_entry\n"
    )
    return ["bash", "-lc", script]


def main():
    parser = argparse.ArgumentParser(description="Phase 4 Gemma eval-only (HF Jobs)")
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--adapter", required=True,
                        help="Adapter repo to evaluate")
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--result-filename", default="phase4-eval-full.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    gf_cfg = config.get("gemma_finetuning", {})
    cloud_cfg = gf_cfg.get("cloud", {})

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""
    if not args.dry_run and not hf_token:
        raise SystemExit("HF_TOKEN env var required to submit an HF Job")

    params = {
        "model_id": gf_cfg.get("model", "google/gemma-4-12B-it"),
        "dataset_id": gf_cfg.get("dataset_id", "cmndcntrlcyber/code-trainer-offsec-dataset"),
        "dataset_revision": gf_cfg.get("dataset_revision", "main"),
        "max_seq_length": int(gf_cfg.get("max_seq_length", 4096)),
        "adapter_repo": args.adapter,
        "result_filename": args.result_filename,
        "output_dir": "/tmp/phase4-gemma-eval",
    }
    if args.val_limit is not None:
        params["val_limit"] = int(args.val_limit)

    env = {
        "PHASE4_PARAMS_JSON": json.dumps(params),
        "PHASE4_ADAPTER_REPO": args.adapter,
        "WANDB_PROJECT": cloud_cfg.get("wandb_project", "rtpi-phase4-gemma4-12b"),
        "REPO_URL": cloud_cfg.get("repo_url", ""),
        "REPO_REF": cloud_cfg.get("repo_ref", "main"),
        "WANDB_MODE": "offline",
    }
    if args.val_limit is not None:
        env["PHASE4_VAL_LIMIT"] = str(int(args.val_limit))

    secrets = {"HF_TOKEN": hf_token}

    spec = JobSpec(
        image=cloud_cfg.get("image", "huggingface/transformers-pytorch-gpu:latest"),
        command=build_eval_command(cloud_cfg.get("repo_url", ""), cloud_cfg.get("repo_ref", "main")),
        flavor=cloud_cfg.get("hardware", gf_cfg.get("hardware", "a100-large")),
        env=env,
        secrets=secrets,
        timeout_seconds=int(cloud_cfg.get("eval_timeout_seconds", 3600)),
        labels={"phase": "4-gemma-eval", "project": "rtpi", "run": f"gemma4-12b-eval-{args.adapter.split('/')[-1]}"},
    )

    s = asdict(spec); s["secrets"] = {k: "<redacted>" for k in spec.secrets}
    logger.info("Job spec:\n%s", json.dumps(s, indent=2, default=str))

    if args.dry_run:
        logger.info("--dry-run set — not submitting.")
        return

    job_id = submit_job(spec, token=hf_token)
    print(f"JOB_ID={job_id}")

    if args.wait:
        final = wait_for_job(
            job_id=job_id, token=hf_token,
            poll_interval=int(cloud_cfg.get("poll_interval", 60)),
            timeout=int(cloud_cfg.get("eval_timeout_seconds", 3600)),
        )
        logger.info(f"Final stage: {final}")
        if final != "COMPLETED":
            sys.exit(1)


if __name__ == "__main__":
    main()
