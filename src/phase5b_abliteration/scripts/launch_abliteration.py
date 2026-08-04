"""
phase5b_abliteration/scripts/launch_abliteration.py

Submit Phase 5b abliteration benchmarking to HF Jobs A100. Follows the
same pattern as Phase 5's launch_convert.py.

Usage:
    set -a && source .env && set +a
    python -m src.phase5b_abliteration.scripts.launch_abliteration \
        --config src/config/config.yaml --wait

    # Dry-run:
    python -m src.phase5b_abliteration.scripts.launch_abliteration --dry-run
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
        "uv run python -m src.phase5b_abliteration.hf_skills.abliterate_entry\n"
    )
    return ["bash", "-lc", script]


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5b Abliteration Benchmarking (HF Jobs)"
    )
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--adapter", default=None,
                        help="Override abliteration.source_adapter")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    abl_cfg = config.get("abliteration", {})
    cloud_cfg = abl_cfg.get("cloud", {})

    adapter_repo = args.adapter or abl_cfg.get("source_adapter")
    base_model = abl_cfg.get("base_model", "Qwen/Qwen2.5-Coder-14B-Instruct")
    output_base = abl_cfg.get("output_base")
    techniques = abl_cfg.get("techniques", [])
    eval_cfg = abl_cfg.get("evaluation", {})
    quants = cloud_cfg.get("quants", ["Q4_K_M"])

    if not adapter_repo:
        raise SystemExit("abliteration.source_adapter not set (or pass --adapter)")
    if not output_base:
        raise SystemExit("abliteration.output_base not set in config")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""
    if not args.dry_run and not hf_token:
        raise SystemExit("HF_TOKEN env var required")

    params = {
        "base_model": base_model,
        "adapter_repo": adapter_repo,
        "output_base": output_base,
        "techniques": techniques,
        "evaluation": eval_cfg,
        "quants": list(quants),
    }

    env = {
        "PHASE5B_PARAMS_JSON": json.dumps(params),
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
        timeout_seconds=int(cloud_cfg.get("timeout_seconds", 21600)),
        labels={"phase": "5b", "project": "rtpi", "run": f"abliterate-{label_slug}"},
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
                                       cloud_cfg.get("timeout_seconds", 21600))),
        )
        logger.info(f"Final stage: {final}")
        if final != "COMPLETED":
            sys.exit(1)


if __name__ == "__main__":
    main()
