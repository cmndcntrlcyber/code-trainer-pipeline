"""
phase4_gemma_finetuning/scripts/launch_benchmark.py

Launch a Phase 4 Gemma general-benchmark A100 job (lm-evaluation-harness).

Usage:
    set -a && source .env && set +a

    # Fine-tuned adapter run:
    python -m src.phase4_gemma_finetuning.scripts.launch_benchmark \
        --adapter cmndcntrlcyber/gemma4-12b-code-trainer-aggressive --wait

    # Baseline run:
    python -m src.phase4_gemma_finetuning.scripts.launch_benchmark \
        --adapter cmndcntrlcyber/gemma4-12b-code-trainer-aggressive \
        --baseline --wait
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


def build_benchmark_command(repo_url: str, repo_ref: str, lm_eval_pin: str) -> list[str]:
    script = (
        "set -euo pipefail\n"
        "apt-get update -qq && apt-get install -y -qq git\n"
        f'git clone --depth 1 --branch "{repo_ref}" "{repo_url}" /workspace\n'
        "cd /workspace\n"
        "pip install -q uv\n"
        "uv sync --frozen\n"
        f'uv pip install "lm-eval=={lm_eval_pin}"\n'
        "uv run python -m src.phase4_gemma_finetuning.hf_skills.benchmark_entry\n"
    )
    return ["bash", "-lc", script]


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4 Gemma general benchmark (lm-eval-harness, HF Jobs)")
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--task", default=None)
    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lm-eval-version", default="0.4.4")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    gf_cfg = config.get("gemma_finetuning", {})
    cloud_cfg = gf_cfg.get("cloud", {})
    bench_cfg = cloud_cfg.get("benchmark", {})

    task = args.task or bench_cfg.get("task", "gsm8k")
    num_fewshot = args.num_fewshot if args.num_fewshot is not None \
        else int(bench_cfg.get("num_fewshot", 0))
    batch_size = args.batch_size if args.batch_size is not None \
        else int(bench_cfg.get("batch_size", 16))
    timeout = int(bench_cfg.get("timeout_seconds", 2400))

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""
    if not args.dry_run and not hf_token:
        raise SystemExit("HF_TOKEN env var required to submit an HF Job")

    suffix = "-base" if args.baseline else ""
    result_filename = f"phase4-benchmark-{task}{suffix}.json"

    params = {
        "model_id": gf_cfg.get("model", "google/gemma-4-12B-it"),
        "adapter_repo": None if args.baseline else args.adapter,
        "task": task,
        "num_fewshot": num_fewshot,
        "batch_size": batch_size,
        "upload_repo": args.adapter,
        "result_filename": result_filename,
        "output_dir": "/tmp/phase4-gemma-benchmark",
    }

    env = {
        "PHASE4_BENCHMARK_PARAMS_JSON": json.dumps(params),
        "REPO_URL": cloud_cfg.get("repo_url", ""),
        "REPO_REF": cloud_cfg.get("repo_ref", "main"),
        "WANDB_MODE": "offline",
    }
    secrets = {"HF_TOKEN": hf_token}

    run_label_kind = "base" if args.baseline else "ft"
    spec = JobSpec(
        image=cloud_cfg.get("image", "huggingface/transformers-pytorch-gpu:latest"),
        command=build_benchmark_command(
            cloud_cfg.get("repo_url", ""),
            cloud_cfg.get("repo_ref", "main"),
            args.lm_eval_version,
        ),
        flavor=cloud_cfg.get("hardware", gf_cfg.get("hardware", "a100-large")),
        env=env,
        secrets=secrets,
        timeout_seconds=timeout,
        labels={
            "phase": "4-gemma-benchmark",
            "project": "rtpi",
            "run": f"gemma4-12b-bench-{task}-{run_label_kind}",
        },
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
            timeout=int(bench_cfg.get("wait_timeout_seconds", timeout + 1800)),
        )
        logger.info(f"Final stage: {final}")
        if final != "COMPLETED":
            sys.exit(1)


if __name__ == "__main__":
    main()
