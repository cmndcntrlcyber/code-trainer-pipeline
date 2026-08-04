"""
phase4_gemma_finetuning/scripts/launch_full_training.py

Launch Phase 4B Gemma: full multi-epoch training on the best Phase 4A config.

Usage:
    set -a && source .env && set +a
    python -m src.phase4_gemma_finetuning.scripts.launch_full_training \
        --config src/config/config.yaml \
        --best-config aggressive --wait

    # Dry-run:
    python -m src.phase4_gemma_finetuning.scripts.launch_full_training \
        --best-config aggressive --dry-run
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
from src.phase4_gemma_finetuning.configs.sweep_configs import SWEEP_CONFIG_MAP, SWEEP_CONFIGS
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

DEFAULT_SUMMARY_JSON = Path("docs/sweep/phase4a-gemma-summary.json")


def build_job_command(repo_url: str, repo_ref: str) -> list[str]:
    script = (
        "set -euo pipefail\n"
        "apt-get update -qq && apt-get install -y -qq git\n"
        f'git clone --depth 1 --branch "{repo_ref}" "{repo_url}" /workspace\n'
        "cd /workspace\n"
        "pip install -q uv\n"
        "uv sync --frozen\n"
        "uv run python -m src.phase4_gemma_finetuning.hf_skills.train_entry\n"
    )
    return ["bash", "-lc", script]


def build_job_spec(
    config_name: str,
    gf_cfg: dict,
    cloud_cfg: dict,
    hf_token: str,
    wandb_key: str | None,
    num_epochs: int,
    suffix: str = "fullN",
    train_limit: int | None = None,
    val_limit: int | None = None,
) -> JobSpec:
    cfg = SWEEP_CONFIG_MAP[config_name]
    adapter_base = cloud_cfg.get("adapter_base") or gf_cfg.get("output_base")
    if not adapter_base:
        raise SystemExit("adapter base not set")
    adapter_repo = f"{adapter_base}-{config_name}-{suffix}"

    params = {
        "name": cfg.name,
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "learning_rate": cfg.learning_rate,
        "batch_size": cfg.batch_size,
        "gradient_accumulation": cfg.gradient_accumulation,
        "model_id": gf_cfg.get("model", "google/gemma-4-12B-it"),
        "dataset_id": gf_cfg.get("dataset_id", "cmndcntrlcyber/code-trainer-offsec-dataset"),
        "dataset_revision": gf_cfg.get("dataset_revision", "main"),
        "num_epochs": num_epochs,
        "max_seq_length": int(gf_cfg.get("max_seq_length", 4096)),
        "adapter_repo": adapter_repo,
        "output_dir": f"/tmp/phase4b-gemma-{config_name}",
    }
    if train_limit is not None:
        params["train_limit"] = int(train_limit)
    if val_limit is not None:
        params["val_limit"] = int(val_limit)

    env = {
        "PHASE4_PARAMS_JSON": json.dumps(params),
        "PHASE4_ADAPTER_REPO": adapter_repo,
        "WANDB_PROJECT": cloud_cfg.get("wandb_project", "rtpi-phase4-gemma4-12b"),
        "REPO_URL": cloud_cfg.get("repo_url", ""),
        "REPO_REF": cloud_cfg.get("repo_ref", "main"),
    }
    if train_limit is not None:
        env["PHASE4_TRAIN_LIMIT"] = str(int(train_limit))
    if val_limit is not None:
        env["PHASE4_VAL_LIMIT"] = str(int(val_limit))
    wandb_mode = os.environ.get("WANDB_MODE")
    if wandb_mode:
        env["WANDB_MODE"] = wandb_mode
    elif not wandb_key:
        env["WANDB_MODE"] = "offline"

    secrets = {"HF_TOKEN": hf_token}
    if wandb_key:
        secrets["WANDB_API_KEY"] = wandb_key

    full_timeout = int(cloud_cfg.get("full_timeout_seconds",
                                     int(cloud_cfg.get("timeout_seconds", 14400)) * 3))

    return JobSpec(
        image=cloud_cfg.get("image", "huggingface/transformers-pytorch-gpu:latest"),
        command=build_job_command(cloud_cfg.get("repo_url", ""), cloud_cfg.get("repo_ref", "main")),
        flavor=cloud_cfg.get("hardware", gf_cfg.get("hardware", "a100-large")),
        env=env,
        secrets=secrets,
        timeout_seconds=full_timeout,
        labels={"phase": "4B-gemma", "project": "rtpi", "run": f"gemma4-12b-{config_name}-full"},
    )


def _print_spec(name: str, spec: JobSpec):
    s = asdict(spec)
    s["secrets"] = {k: "<redacted>" for k in spec.secrets}
    logger.info(f"=== {name} ===\n%s", json.dumps(s, indent=2, default=str))


def _resolve_top_n(top_n: int) -> list[str]:
    if not DEFAULT_SUMMARY_JSON.exists():
        raise SystemExit(
            f"--top-n requires {DEFAULT_SUMMARY_JSON}. Run generate_report.py first."
        )
    summary = json.loads(DEFAULT_SUMMARY_JSON.read_text())
    completed = [
        r for r in summary.get("rows", [])
        if r.get("result") and r["result"].get("eval_loss") is not None
    ]
    if len(completed) < top_n:
        raise SystemExit(
            f"--top-n {top_n} requested but only {len(completed)} completed runs in summary."
        )
    return [r["name"] for r in completed[:top_n]]


def main():
    parser = argparse.ArgumentParser(description="Phase 4B Gemma: full training (HF Jobs)")
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait", action="store_true")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--best-config", default=None,
                     choices=[c.name for c in SWEEP_CONFIGS],
                     help="Single config to run for full training")
    grp.add_argument("--top-n", type=int, default=None,
                     help="Read top-N from docs/sweep/phase4a-gemma-summary.json")
    parser.add_argument("--suffix", default="fullN",
                        help="Adapter repo suffix (e.g. 'full3' for 3 epochs)")
    parser.add_argument("--train-limit", type=int, default=None,
                        help="Cap training rows")
    parser.add_argument("--val-limit", type=int, default=None,
                        help="Cap validation rows")
    args = parser.parse_args()

    config = load_config(args.config)
    gf_cfg = config.get("gemma_finetuning", {})
    cloud_cfg = gf_cfg.get("cloud", {})
    full_cfg = gf_cfg.get("full_training", {})
    num_epochs = int(full_cfg.get("num_epochs", 3))

    config_names = (
        [args.best_config]
        if args.best_config
        else _resolve_top_n(args.top_n)
    )

    suffix = args.suffix if args.suffix != "fullN" else f"full{num_epochs}"

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""
    wandb_key = os.environ.get("WANDB_API_KEY")
    if not args.dry_run and not hf_token:
        raise SystemExit("HF_TOKEN env var required to submit an HF Job")

    specs = {
        name: build_job_spec(
            name, gf_cfg, cloud_cfg, hf_token, wandb_key,
            num_epochs=num_epochs, suffix=suffix,
            train_limit=args.train_limit, val_limit=args.val_limit,
        )
        for name in config_names
    }

    for name, spec in specs.items():
        _print_spec(name, spec)

    if args.dry_run:
        logger.info("--dry-run set — not submitting %d job(s).", len(specs))
        return

    job_ids: dict[str, str] = {}
    for name, spec in specs.items():
        jid = submit_job(spec, token=hf_token)
        job_ids[name] = jid
        print(f"JOB_ID[{name}]={jid}")
        time.sleep(2)

    Path("data/sweep_results").mkdir(parents=True, exist_ok=True)
    Path("data/sweep_results/gemma_full_training_job_ids.json").write_text(
        json.dumps(job_ids, indent=2)
    )

    if not args.wait:
        return

    poll_interval = int(cloud_cfg.get("poll_interval", 60))
    timeout = int(cloud_cfg.get("full_timeout_seconds",
                                int(cloud_cfg.get("timeout_seconds", 14400)) * 3))

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
            logger.info(f"  [{name}] final stage: {stage}")

    if not all(s == "COMPLETED" for s in finals.values()):
        logger.error("One or more Gemma full-training jobs did not complete: %s", finals)
        sys.exit(1)
    logger.info("All Gemma full-training jobs completed.")


if __name__ == "__main__":
    main()
