"""
phase4_qwen_finetuning/scripts/launch_v7_validation.py

Orchestrate all 4 V7 validation checks as HF Jobs:

  1. eval_loss    — reuses eval_entry.py (target < 0.50)
  2. tool_calling — tool_call_eval_entry.py (target >= 80%)
  3. agent        — agent_eval_entry.py (target >= 3/5)
  4. gsm8k        — benchmark_entry.py (target >= 0.60)

Usage:
    set -a && source .env && set +a
    python -m src.phase4_qwen_finetuning.scripts.launch_v7_validation \\
        --adapter cmndcntrlcyber/qwen14b-code-trainer-v7_mixed --wait

    # Run only tool-calling and agent checks:
    python -m src.phase4_qwen_finetuning.scripts.launch_v7_validation \\
        --adapter ... --skip-eval-loss --skip-gsm8k --wait

    # Dry-run:
    python -m src.phase4_qwen_finetuning.scripts.launch_v7_validation \\
        --adapter ... --dry-run
"""
import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def build_command(repo_url: str, repo_ref: str, entry_module: str) -> list[str]:
    script = (
        "set -euo pipefail\n"
        "apt-get update -qq && apt-get install -y -qq git\n"
        f'git clone --depth 1 --branch "{repo_ref}" "{repo_url}" /workspace\n'
        "cd /workspace\n"
        "pip install -q uv\n"
        "uv sync --frozen\n"
        f"uv run python -m {entry_module}\n"
    )
    return ["bash", "-lc", script]


def build_benchmark_command(repo_url: str, repo_ref: str) -> list[str]:
    script = (
        "set -euo pipefail\n"
        "apt-get update -qq && apt-get install -y -qq git\n"
        f'git clone --depth 1 --branch "{repo_ref}" "{repo_url}" /workspace\n'
        "cd /workspace\n"
        "pip install -q uv\n"
        "uv sync --frozen\n"
        "uv pip install -q 'lm-eval==0.4.4'\n"
        "uv run python -m src.phase4_qwen_finetuning.hf_skills.benchmark_entry\n"
    )
    return ["bash", "-lc", script]


def main():
    parser = argparse.ArgumentParser(description="V7 validation suite")
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--adapter", required=True, help="V7 adapter repo on Hub")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--skip-eval-loss", action="store_true")
    parser.add_argument("--skip-tool-call", action="store_true")
    parser.add_argument("--skip-agent", action="store_true")
    parser.add_argument("--skip-gsm8k", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    qf_cfg = config.get("qwen_finetuning", {})
    cloud_cfg = qf_cfg.get("cloud", {})
    model_id = qf_cfg.get("model", "Qwen/Qwen2.5-Coder-14B-Instruct")
    repo_url = cloud_cfg.get("repo_url", "")
    repo_ref = cloud_cfg.get("repo_ref", "main")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""
    wandb_key = os.environ.get("WANDB_API_KEY")
    if not args.dry_run and not hf_token:
        raise SystemExit("HF_TOKEN env var required")

    v7_val = qf_cfg.get("v7_validation", {})
    logger.info("=" * 60)
    logger.info("V7 Validation Suite")
    logger.info(f"  adapter:  {args.adapter}")
    logger.info(f"  targets:  eval_loss<{v7_val.get('eval_loss_threshold', 0.50)}, "
                f"tool_call>={v7_val.get('tool_call_pass_rate', 0.80)}, "
                f"agent>={v7_val.get('agent_progress_threshold', 3)}/5, "
                f"gsm8k>={v7_val.get('gsm8k_threshold', 0.60)}")
    logger.info("=" * 60)

    jobs: dict[str, JobSpec] = {}

    base_env = {
        "REPO_URL": repo_url,
        "REPO_REF": repo_ref,
    }
    base_secrets = {"HF_TOKEN": hf_token}

    if not args.skip_eval_loss:
        eval_params = {
            "model_id": model_id,
            "adapter_repo": args.adapter,
            "dataset_id": "cmndcntrlcyber/code-trainer-v7-mixed",
            "dataset_revision": "main",
            "upload_repo": args.adapter,
            "result_filename": "phase4-eval-v7.json",
        }
        jobs["eval_loss"] = JobSpec(
            image=cloud_cfg.get("image", "huggingface/transformers-pytorch-gpu:latest"),
            command=build_command(repo_url, repo_ref,
                                  "src.phase4_qwen_finetuning.hf_skills.eval_entry"),
            flavor=cloud_cfg.get("hardware", "a100-large"),
            env={**base_env, "PHASE4_PARAMS_JSON": json.dumps(eval_params), "PHASE4_ADAPTER_REPO": args.adapter},
            secrets=base_secrets,
            timeout_seconds=int(cloud_cfg.get("timeout_seconds", 14400)),
            labels={"phase": "4-v7-val", "check": "eval_loss"},
        )

    if not args.skip_tool_call:
        tool_params = {
            "model_id": model_id,
            "adapter_repo": args.adapter,
            "upload_repo": args.adapter,
            "result_filename": "phase4-tool-call-eval.json",
        }
        jobs["tool_calling"] = JobSpec(
            image=cloud_cfg.get("image", "huggingface/transformers-pytorch-gpu:latest"),
            command=build_command(repo_url, repo_ref,
                                  "src.phase4_qwen_finetuning.hf_skills.tool_call_eval_entry"),
            flavor=cloud_cfg.get("hardware", "a100-large"),
            env={**base_env, "PHASE4_TOOL_EVAL_PARAMS_JSON": json.dumps(tool_params)},
            secrets=base_secrets,
            timeout_seconds=int(cloud_cfg.get("timeout_seconds", 14400)),
            labels={"phase": "4-v7-val", "check": "tool_calling"},
        )

    if not args.skip_agent:
        agent_params = {
            "model_id": model_id,
            "adapter_repo": args.adapter,
            "upload_repo": args.adapter,
            "result_filename": "phase4-agent-eval.json",
        }
        jobs["agent"] = JobSpec(
            image=cloud_cfg.get("image", "huggingface/transformers-pytorch-gpu:latest"),
            command=build_command(repo_url, repo_ref,
                                  "src.phase4_qwen_finetuning.hf_skills.agent_eval_entry"),
            flavor=cloud_cfg.get("hardware", "a100-large"),
            env={**base_env, "PHASE4_AGENT_EVAL_PARAMS_JSON": json.dumps(agent_params)},
            secrets=base_secrets,
            timeout_seconds=int(cloud_cfg.get("timeout_seconds", 14400)),
            labels={"phase": "4-v7-val", "check": "agent"},
        )

    if not args.skip_gsm8k:
        bench_cfg = cloud_cfg.get("benchmark", {})
        bench_params = {
            "model_id": model_id,
            "adapter_repo": args.adapter,
            "task": bench_cfg.get("task", "gsm8k"),
            "num_fewshot": int(bench_cfg.get("num_fewshot", 0)),
            "batch_size": int(bench_cfg.get("batch_size", 16)),
            "upload_repo": args.adapter,
            "result_filename": "phase4-benchmark-gsm8k-v7.json",
        }
        jobs["gsm8k"] = JobSpec(
            image=cloud_cfg.get("image", "huggingface/transformers-pytorch-gpu:latest"),
            command=build_benchmark_command(repo_url, repo_ref),
            flavor=cloud_cfg.get("hardware", "a100-large"),
            env={**base_env, "PHASE4_BENCHMARK_PARAMS_JSON": json.dumps(bench_params)},
            secrets=base_secrets,
            timeout_seconds=int(bench_cfg.get("timeout_seconds", 2400)),
            labels={"phase": "4-v7-val", "check": "gsm8k"},
        )

    if not jobs:
        logger.info("All checks skipped — nothing to do.")
        return

    for name, spec in jobs.items():
        s = asdict(spec)
        s["secrets"] = {k: "<redacted>" for k in spec.secrets}
        logger.info("=== %s ===\n%s", name, json.dumps(s, indent=2, default=str))

    if args.dry_run:
        logger.info("--dry-run: not submitting %d job(s)", len(jobs))
        return

    job_ids: dict[str, str] = {}
    for name, spec in jobs.items():
        jid = submit_job(spec, token=hf_token)
        job_ids[name] = jid
        print(f"JOB_ID[{name}]={jid}")
        time.sleep(2)

    Path("data/v7_validation").mkdir(parents=True, exist_ok=True)
    Path("data/v7_validation/job_ids.json").write_text(json.dumps(job_ids, indent=2))
    logger.info("Submitted %d job(s); ids saved to data/v7_validation/job_ids.json", len(job_ids))

    if not args.wait:
        return

    poll_interval = int(cloud_cfg.get("poll_interval", 60))
    timeout = int(cloud_cfg.get("timeout_seconds", 14400))

    finals: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(job_ids)) as pool:
        futures = {
            pool.submit(wait_for_job, jid, hf_token, poll_interval, timeout): name
            for name, jid in job_ids.items()
        }
        for fut in as_completed(futures):
            name = futures[fut]
            stage = fut.result()
            finals[name] = stage
            logger.info("  [%s] final stage: %s", name, stage)

    Path("data/v7_validation/final_stages.json").write_text(json.dumps(finals, indent=2))

    failed = [n for n, s in finals.items() if s != "COMPLETED"]
    if failed:
        logger.error("Jobs did not complete: %s", failed)
        sys.exit(1)

    logger.info("All V7 validation jobs completed. Check results at:")
    logger.info("  https://huggingface.co/%s", args.adapter)


if __name__ == "__main__":
    main()
