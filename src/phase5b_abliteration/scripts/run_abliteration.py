"""
phase5b_abliteration/scripts/run_abliteration.py

Local runner for a single abliteration technique. For development and testing
on small models (TinyLlama, GPT-2). The cloud entry point runs all techniques.

Usage:
    python -m src.phase5b_abliteration.scripts.run_abliteration \
        --technique obliteratus \
        --model-dir data/gguf_work/merged_model \
        --output-dir data/abliterated/obliteratus

    # Test with a small model:
    python -m src.phase5b_abliteration.scripts.run_abliteration \
        --technique nousresearch \
        --model-dir TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --output-dir /tmp/abl_test
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.phase5b_abliteration.techniques import TECHNIQUES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run a single abliteration technique")
    parser.add_argument(
        "--technique", required=True, choices=list(TECHNIQUES.keys()),
        help="Abliteration technique to run",
    )
    parser.add_argument(
        "--model-dir", required=True,
        help="Path to merged HF model directory (or HF model ID for testing)",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to save abliterated model",
    )
    parser.add_argument("--preset", default="advanced", help="OBLITERATUS preset")
    parser.add_argument("--method", default="biprojected", help="NousResearch method")
    parser.add_argument("--steering-method", default="SRA", help="abliterix steering method")
    parser.add_argument("--n-samples", type=int, default=512, help="Contrastive samples")
    parser.add_argument("--optuna-trials", type=int, default=50, help="abliterix Optuna trials")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)

    if not model_dir.exists():
        logger.info("Model dir %s not found locally, treating as HF model ID", model_dir)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        logger.info("Downloading %s ...", args.model_dir)
        local = Path(f"/tmp/abl_model_cache/{args.model_dir.replace('/', '_')}")
        local.mkdir(parents=True, exist_ok=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, torch_dtype=torch.float16, device_map="cpu"
        )
        model.save_pretrained(str(local), safe_serialization=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        tokenizer.save_pretrained(str(local))
        del model
        model_dir = local

    config = {
        "name": args.technique,
        "preset": args.preset,
        "method": args.method,
        "steering_method": args.steering_method,
        "n_samples": args.n_samples,
        "optuna_trials": args.optuna_trials,
        "n_directions": 4,
        "norm_preserve": True,
    }

    tech_class = TECHNIQUES[args.technique]
    technique = tech_class()

    logger.info("Running %s on %s → %s", args.technique, model_dir, output_dir)
    result_dir = technique.abliterate(model_dir, output_dir, config)
    logger.info("Abliteration complete: %s", result_dir)

    safetensors = list(result_dir.glob("*.safetensors"))
    logger.info("Output contains %d safetensors files", len(safetensors))


if __name__ == "__main__":
    main()
