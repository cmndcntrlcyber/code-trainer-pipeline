"""
phase5_deployment/scripts/convert_to_gguf.py

Phase 5: Convert fine-tuned Qwen-14B LoRA → GGUF Q4_K_M and upload to HF Hub.

Usage:
    python -m src.phase5_deployment.scripts.convert_to_gguf \
        --config src/config/config.yaml \
        --adapter-repo combatcougar/qwen14b-code-trainer-standard \
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
from src.phase5_deployment.gguf.converter import GGUFConverter
from src.phase5_deployment.gguf.uploader import GGUFUploader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Convert to GGUF")
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--adapter-repo", required=True,
                        help="HF Hub repo ID of the fine-tuned LoRA adapter")
    parser.add_argument("--llama-cpp", required=True,
                        help="Path to compiled llama.cpp directory")
    parser.add_argument("--quant", default="Q5_K_M",
                        help="Quantization type (Q5_K_M, Q4_K_M, Q8_0)")
    parser.add_argument("--output-dir", default="models/gguf")
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    deploy_cfg = config.get("deployment", {})

    hf_token = os.environ.get("HF_TOKEN")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert
    converter = GGUFConverter(
        llama_cpp_dir=args.llama_cpp,
        work_dir=output_dir / "work",
    )
    gguf_path = converter.run_full_pipeline(
        adapter_repo_id=args.adapter_repo,
        base_model_id="Qwen/Qwen2.5-Coder-14B-Instruct",
        quant_type=args.quant,
        output_path=output_dir / f"model_{args.quant.lower()}.gguf",
    )
    logger.info(f"GGUF ready: {gguf_path} ({gguf_path.stat().st_size / 1e9:.1f} GB)")

    # Upload to Hub
    if args.push_to_hub:
        if not hf_token:
            logger.error("HF_TOKEN not set — cannot push to Hub")
            sys.exit(1)

        gguf_repo = deploy_cfg.get("gguf_repo", "combatcougar/qwen14b-code-trainer-gguf")
        uploader = GGUFUploader(token=hf_token)

        # Load evaluation results if available
        eval_dir = Path("models/vision_model/eval")
        baseline = {}
        finetuned = {}
        import json
        if (eval_dir / "baseline.json").exists():
            baseline = json.loads((eval_dir / "baseline.json").read_text())
        if (eval_dir / "finetuned.json").exists():
            finetuned = json.loads((eval_dir / "finetuned.json").read_text())

        wandb_url = os.environ.get("WANDB_PROJECT_URL", "https://wandb.ai/combatcougar/code-trainer-phase4")

        model_card_params = {
            "num_samples": 32727,
            "lora_r": 32,
            "lora_alpha": 64,
            "learning_rate": "2e-4",
            "num_epochs": 3,
            "dataset_id": config.get("preprocessing", {}).get("dataset_name", ""),
            "baseline_em": baseline.get("exact_match", 0.0),
            "finetuned_em": finetuned.get("exact_match", 0.0),
            "em_delta": finetuned.get("exact_match", 0.0) - baseline.get("exact_match", 0.0),
            "baseline_bleu": baseline.get("bleu_4", 0.0),
            "finetuned_bleu": finetuned.get("bleu_4", 0.0),
            "bleu_delta": finetuned.get("bleu_4", 0.0) - baseline.get("bleu_4", 0.0),
            "baseline_edit": baseline.get("mean_edit_similarity", 0.0),
            "finetuned_edit": finetuned.get("mean_edit_similarity", 0.0),
            "edit_delta": finetuned.get("mean_edit_similarity", 0.0) - baseline.get("mean_edit_similarity", 0.0),
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
