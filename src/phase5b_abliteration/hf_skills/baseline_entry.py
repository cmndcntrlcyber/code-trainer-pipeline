"""
phase5b_abliteration/hf_skills/baseline_entry.py

Container-side entry for pre-fine-tuning baseline abliteration. Iterates
over multiple base models sequentially (e.g., Qwen then Gemma), running
all 3 abliteration techniques + evaluation on each.

Reuses abliterate_entry.main() for each run by setting PHASE5B_PARAMS_JSON.

Required env vars:
    HF_TOKEN                          — read base models, write output repos
    PHASE5B_BASELINE_PARAMS_JSON      — { runs[], techniques[], evaluation{},
                                          quants[] }
"""
import json
import logging
import os
import shutil
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

WORK = Path("/workspace/phase5b")


def main():
    params = json.loads(os.environ.get("PHASE5B_BASELINE_PARAMS_JSON", "{}"))
    runs = params.get("runs", [])
    techniques = params.get("techniques", [])
    eval_cfg = params.get("evaluation", {})
    quants = params.get("quants", ["Q4_K_M"])

    if not runs:
        raise RuntimeError("No runs specified in PHASE5B_BASELINE_PARAMS_JSON")

    logger.info("=" * 60)
    logger.info("PHASE 5b BASELINE — Sequential abliteration of %d models", len(runs))
    for r in runs:
        logger.info("  - %s: %s (adapter: %s)",
                     r.get("name"), r.get("base_model"), r.get("source_adapter"))
    logger.info("=" * 60)

    results = {}

    for i, run_cfg in enumerate(runs):
        run_name = run_cfg.get("name", f"run-{i}")
        base_model = run_cfg.get("base_model")
        adapter_repo = run_cfg.get("source_adapter")
        output_base = run_cfg.get("output_base")

        logger.info("=" * 60)
        logger.info("RUN %d/%d: %s (%s)", i + 1, len(runs), run_name, base_model)
        logger.info("=" * 60)

        run_params = {
            "base_model": base_model,
            "adapter_repo": adapter_repo,
            "output_base": output_base,
            "techniques": techniques,
            "evaluation": eval_cfg,
            "quants": quants,
        }
        os.environ["PHASE5B_PARAMS_JSON"] = json.dumps(run_params)

        merged_dir = WORK / "merged"
        if merged_dir.exists():
            logger.info("Cleaning previous merged model at %s", merged_dir)
            shutil.rmtree(merged_dir)

        for d in WORK.glob("abliterated_*"):
            logger.info("Cleaning previous abliterated dir %s", d)
            shutil.rmtree(d)

        eval_dir = WORK / "evaluation_results"
        if eval_dir.exists():
            shutil.rmtree(eval_dir)

        try:
            from src.phase5b_abliteration.hf_skills.abliterate_entry import main as run_abliteration
            run_abliteration()
            results[run_name] = "COMPLETED"
            logger.info("RUN %s: COMPLETED", run_name)
        except Exception:
            logger.exception("RUN %s: FAILED", run_name)
            results[run_name] = "FAILED"

    logger.info("=" * 60)
    logger.info("BASELINE ABLITERATION SUMMARY:")
    for name, status in results.items():
        logger.info("  %s: %s", name, status)
    logger.info("=" * 60)

    failed = [n for n, s in results.items() if s != "COMPLETED"]
    if failed:
        raise RuntimeError(f"Runs failed: {failed}")


if __name__ == "__main__":
    main()
