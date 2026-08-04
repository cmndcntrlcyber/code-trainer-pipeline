"""
phase4_gemma_finetuning/hf_skills/benchmark_entry.py

General-benchmark entry: run lm-evaluation-harness against Gemma-4-12B with
(or without) a published LoRA adapter from Hub. Catastrophic-forgetting check.

Expected env vars:
    HF_TOKEN                       — read base/adapter, push result
    PHASE4_BENCHMARK_PARAMS_JSON   — { model_id, adapter_repo?, task,
                                       num_fewshot, batch_size, upload_repo,
                                       result_filename }
"""
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

os.environ.setdefault("HF_HOME", "/workspace/.hf-cache")


def main():
    from huggingface_hub import HfApi

    params = json.loads(os.environ.get("PHASE4_BENCHMARK_PARAMS_JSON", "{}"))
    model_id = params.get("model_id", "google/gemma-4-12B-it")
    adapter_repo = params.get("adapter_repo")
    task = params.get("task", "gsm8k")
    num_fewshot = int(params.get("num_fewshot", 0))
    batch_size = int(params.get("batch_size", 16))
    upload_repo = params.get("upload_repo")
    result_filename = params.get("result_filename", f"phase4-benchmark-{task}.json")
    output_dir = Path(params.get("output_dir", "/tmp/phase4-benchmark"))
    output_dir.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN required")
    if not upload_repo:
        if adapter_repo:
            upload_repo = adapter_repo
        else:
            raise RuntimeError(
                "upload_repo required for baseline runs (cannot push to upstream)")

    logger.info("=" * 60)
    logger.info("PHASE 4 GEMMA — General benchmark (lm-evaluation-harness)")
    logger.info(f"  base:        {model_id}")
    logger.info(f"  adapter:     {adapter_repo or '(baseline — base only)'}")
    logger.info(f"  task:        {task} ({num_fewshot}-shot)")
    logger.info(f"  batch_size:  {batch_size}")
    logger.info(f"  upload to:   {upload_repo}")
    logger.info(f"  result file: {result_filename}")
    logger.info("=" * 60)

    model_args = [
        f"pretrained={model_id}",
        "dtype=bfloat16",
    ]
    if adapter_repo:
        model_args.append(f"peft={adapter_repo}")

    out_json = output_dir / "lm_eval_raw.json"
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", ",".join(model_args),
        "--tasks", task,
        "--num_fewshot", str(num_fewshot),
        "--batch_size", str(batch_size),
        "--device", "cuda",
        "--output_path", str(out_json),
        "--log_samples",
    ]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if out_json.is_dir():
        candidates = sorted(out_json.rglob("results_*.json"))
        if not candidates:
            candidates = sorted(out_json.rglob("*.json"))
        raw_path = candidates[-1] if candidates else None
    else:
        raw_path = out_json if out_json.exists() else None
    if raw_path is None:
        raise RuntimeError(f"Could not find lm_eval output under {out_json}")
    raw = json.loads(raw_path.read_text())

    task_results = raw.get("results", {}).get(task, {})
    payload = {
        "model": model_id,
        "adapter": adapter_repo,
        "task": task,
        "num_fewshot": num_fewshot,
        "batch_size": batch_size,
        "results": task_results,
        "lm_eval_version": raw.get("config", {}).get("lm_eval_version"),
        "raw_path": str(raw_path),
    }
    out_path = output_dir / result_filename
    out_path.write_text(json.dumps(payload, indent=2, default=str))

    logger.info(f"Pushing {result_filename} → {upload_repo}")
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(out_path),
        path_in_repo=result_filename,
        repo_id=upload_repo,
        repo_type="model",
        commit_message=f"Phase 4 Gemma benchmark: {task} ({num_fewshot}-shot)",
    )
    logger.info(
        f"Done: https://huggingface.co/{upload_repo}/blob/main/{result_filename}")


if __name__ == "__main__":
    main()
