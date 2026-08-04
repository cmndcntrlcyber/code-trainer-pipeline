"""
phase5_gemma_deployment/scripts/launch_convert.py

Submit Phase 5 Gemma GGUF conversion to HF Jobs A100.

Usage:
    set -a && source .env && set +a
    python -m src.phase5_gemma_deployment.scripts.launch_convert \
        --config src/config/config.yaml --wait

    # Override adapter / quants:
    python -m src.phase5_gemma_deployment.scripts.launch_convert \
        --adapter cmndcntrlcyber/gemma4-12b-code-trainer-aggressive \
        --quants Q4_K_M Q5_K_M --wait

    # Dry-run:
    python -m src.phase5_gemma_deployment.scripts.launch_convert --dry-run
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


def build_convert_command(repo_url: str, repo_ref: str) -> list[str]:
    script = (
        "set -euo pipefail\n"
        "apt-get update -qq && apt-get install -y -qq git cmake build-essential\n"
        f'git clone --depth 1 --branch "{repo_ref}" "{repo_url}" /workspace/repo\n'
        "cd /workspace/repo\n"
        "pip install -q uv\n"
        "uv sync --frozen\n"
        "uv run python -m src.phase5_gemma_deployment.hf_skills.convert_entry\n"
    )
    return ["bash", "-lc", script]


def main():
    parser = argparse.ArgumentParser(description="Phase 5 Gemma GGUF conversion (HF Jobs)")
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--adapter", default=None,
                        help="Override gemma_deployment.source_adapter")
    parser.add_argument("--gguf-repo", default=None,
                        help="Override gemma_deployment.gguf_repo")
    parser.add_argument("--quants", nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    dep_cfg = config.get("gemma_deployment", {})
    cloud_cfg = dep_cfg.get("cloud", {})

    adapter_repo = args.adapter or dep_cfg.get("source_adapter")
    gguf_repo = args.gguf_repo or dep_cfg.get("gguf_repo")
    base_model = dep_cfg.get("base_model", "google/gemma-4-12B-it")
    quants = args.quants or cloud_cfg.get("quants") or ["Q4_K_M"]

    if not adapter_repo:
        raise SystemExit("gemma_deployment.source_adapter not set (or pass --adapter)")
    if not gguf_repo:
        raise SystemExit("gemma_deployment.gguf_repo not set (or pass --gguf-repo)")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""
    if not args.dry_run and not hf_token:
        raise SystemExit("HF_TOKEN env var required to submit an HF Job")

    params = {
        "base_model": base_model,
        "adapter_repo": adapter_repo,
        "gguf_repo": gguf_repo,
        "quants": list(quants),
    }

    env = {
        "PHASE5_PARAMS_JSON": json.dumps(params),
        "PHASE5_GGUF_REPO": gguf_repo,
        "REPO_URL": cloud_cfg.get("repo_url", ""),
        "REPO_REF": cloud_cfg.get("repo_ref", "main"),
    }
    secrets = {"HF_TOKEN": hf_token}

    label_slug = re.sub(r"[^A-Za-z0-9_=-]+", "-", base_model.split("/")[-1]).strip("-")

    spec = JobSpec(
        image=cloud_cfg.get("image", "huggingface/transformers-pytorch-gpu:latest"),
        command=build_convert_command(cloud_cfg.get("repo_url", ""), cloud_cfg.get("repo_ref", "main")),
        flavor=cloud_cfg.get("hardware", "a100-large"),
        env=env,
        secrets=secrets,
        timeout_seconds=int(cloud_cfg.get("timeout_seconds", 7200)),
        labels={"phase": "5-gemma", "project": "rtpi", "run": f"gguf-{label_slug}"},
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
            timeout=int(cloud_cfg.get("wait_timeout_seconds",
                                       cloud_cfg.get("timeout_seconds", 7200))),
        )
        logger.info(f"Final stage: {final}")
        if final != "COMPLETED":
            sys.exit(1)


if __name__ == "__main__":
    main()
