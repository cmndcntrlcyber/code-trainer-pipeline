"""
phase5_gemma_deployment/scripts/convert_to_gguf.py

Phase 5 Gemma: Convert fine-tuned Gemma-4-12B LoRA → GGUF Q4_K_M and upload.

Usage:
    python -m src.phase5_gemma_deployment.scripts.convert_to_gguf \
        --config src/config/config.yaml \
        --adapter-repo cmndcntrlcyber/gemma4-12b-code-trainer-aggressive \
        --llama-cpp /path/to/llama.cpp \
        --push-to-hub
"""
import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.config.settings import load_config
from src.phase5_gemma_deployment.gguf.converter import GGUFConverter
from src.phase5_gemma_deployment.gguf.uploader import GGUFUploader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Phase 5 Gemma: Convert to GGUF")
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--adapter-repo", required=True)
    parser.add_argument("--llama-cpp", required=True)
    parser.add_argument("--quant", default="Q4_K_M")
    parser.add_argument("--output-dir", default="models/gguf-gemma")
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    deploy_cfg = config.get("gemma_deployment", {})

    hf_token = os.environ.get("HF_TOKEN")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    converter = GGUFConverter(
        llama_cpp_dir=args.llama_cpp,
        work_dir=output_dir / "work",
    )
    gguf_path = converter.run_full_pipeline(
        adapter_repo_id=args.adapter_repo,
        base_model_id="google/gemma-4-12B-it",
        quant_type=args.quant,
        output_path=output_dir / f"model_{args.quant.lower()}.gguf",
    )
    logger.info(f"GGUF ready: {gguf_path} ({gguf_path.stat().st_size / 1e9:.1f} GB)")

    if args.push_to_hub:
        if not hf_token:
            logger.error("HF_TOKEN not set — cannot push to Hub")
            sys.exit(1)

        gguf_repo = deploy_cfg.get("gguf_repo", "cmndcntrlcyber/gemma4-12b-code-trainer-gguf")
        uploader = GGUFUploader(token=hf_token)

        wandb_url = os.environ.get("WANDB_PROJECT_URL", "https://wandb.ai/cmndcntrlcyber-c3s-consulting/rtpi-phase4-gemma4-12b")

        model_card_params = {
            "num_samples": 32727,
            "lora_r": 64,
            "lora_alpha": 128,
            "learning_rate": "2e-4",
            "num_epochs": 1,
            "dataset_id": config.get("preprocessing", {}).get("dataset_name", ""),
            "wandb_url": wandb_url,
        }

        url = uploader.upload(
            gguf_path=gguf_path,
            repo_id=gguf_repo,
            model_card_params=model_card_params,
            quant_type=args.quant,
        )
        logger.info(f"Model published: {url}")


if __name__ == "__main__":
    main()
