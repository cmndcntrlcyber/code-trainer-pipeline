"""
Convert a fine-tuned Qwen3.5-0.8B triage model to RKLLM w8a8 format
for deployment on RK3588 NPU.

Usage (on OPi 5 Plus or cross-compile host):
    python -m src.phase5_triage_deployment.scripts.convert_to_rkllm \
        --base-model Qwen/Qwen3.5-0.8B \
        --adapter cmndcntrlcyber/qwen35-08b-triage-sft \
        --output-dir data/rkllm_triage

Requires: rkllm-toolkit (pip install rkllm-toolkit)
"""
import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def merge_adapter(base_model: str, adapter: str, output_dir: Path, token: str | None):
    """Merge LoRA adapter into base model for RKLLM export."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading base model: %s", base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype="auto", token=token
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=token)

    logger.info("Merging adapter: %s", adapter)
    model = PeftModel.from_pretrained(model, adapter, token=token)
    model = model.merge_and_unload()

    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    logger.info("Merged model saved to %s", merged_dir)
    return merged_dir


def convert_rkllm(merged_dir: Path, output_dir: Path, target_platform: str = "rk3588"):
    """Convert merged HF model to RKLLM w8a8 format."""
    try:
        from rkllm.api import RKLLM
    except ImportError:
        logger.error(
            "rkllm-toolkit not installed. Install with: pip install rkllm-toolkit\n"
            "Note: rkllm-toolkit requires ARM64 Linux (run on the OPi 5 Plus directly)"
        )
        raise SystemExit(1)

    logger.info("Initializing RKLLM converter for %s", target_platform)
    llm = RKLLM()

    logger.info("Loading model from %s", merged_dir)
    llm.load_huggingface(model=str(merged_dir))

    logger.info("Building RKLLM model (w8a8 quantization)...")
    llm.build(
        do_quantization=True,
        optimization_level=1,
        quantized_dtype="w8a8",
        target_platform=target_platform,
    )

    rkllm_path = output_dir / "triage.rkllm"
    llm.export_rkllm(str(rkllm_path))
    logger.info("RKLLM model exported to %s", rkllm_path)
    return rkllm_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert triage model to RKLLM w8a8 for RK3588 NPU"
    )
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--adapter", required=True,
                        help="HF Hub adapter ID or local path")
    parser.add_argument("--output-dir", default="data/rkllm_triage")
    parser.add_argument("--target-platform", default="rk3588",
                        choices=["rk3588", "rk3588s"])
    parser.add_argument("--skip-merge", action="store_true",
                        help="Skip merge step if merged model already exists")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    merged_dir = output_dir / "merged"
    if not args.skip_merge:
        merged_dir = merge_adapter(args.base_model, args.adapter, output_dir, token)
    else:
        if not merged_dir.exists():
            raise FileNotFoundError(f"Merged model not found at {merged_dir}")
        logger.info("Skipping merge, using existing: %s", merged_dir)

    convert_rkllm(merged_dir, output_dir, args.target_platform)
    logger.info("Triage RKLLM conversion complete.")


if __name__ == "__main__":
    main()
