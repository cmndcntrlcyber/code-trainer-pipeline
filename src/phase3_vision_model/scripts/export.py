"""
phase3_vision_model/scripts/export.py

Export the fine-tuned Phase 3 model:
  - Merge LoRA adapters into the base decoder
  - Save merged model + projector to disk
  - Optionally push to HF Hub

Usage:
    python -m src.phase3_vision_model.scripts.export \
        --config src/config/config.yaml \
        --checkpoint models/vision_model/best \
        --output-dir models/vision_model_merged \
        --push-to-hub
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from peft import PeftModel

from src.config.settings import load_config
from src.phase3_vision_model.architecture.code_decoder import DECODER_MODEL_ID
from src.phase3_vision_model.architecture.vision_model import CodeVisionModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Export merged model")
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--checkpoint", default="models/vision_model/best")
    parser.add_argument("--output-dir", default="models/vision_model_merged")
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    vm_cfg = config.get("vision_model", {})
    output_dir = Path(args.output_dir)

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model = CodeVisionModel.from_pretrained(
        args.checkpoint,
        vision_model_id=vm_cfg.get("vision_encoder", "microsoft/swin-base-patch4-window7-224"),
        decoder_model_id=vm_cfg.get("decoder", DECODER_MODEL_ID),
    )

    logger.info("Merging LoRA adapters into base model...")
    merged_decoder = model.decoder.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_decoder.save_pretrained(str(output_dir / "decoder"))
    model.tokenizer.save_pretrained(str(output_dir / "decoder"))
    torch.save(model.projector.state_dict(), output_dir / "projector.pt")
    logger.info(f"Merged model saved to {output_dir}")

    if args.push_to_hub:
        hf_username = config.get("preprocessing", {}).get("dataset_name", "").split("/")[0]
        repo_id = f"{hf_username}/code-vision-model"
        logger.info(f"Pushing to Hub: {repo_id}")
        merged_decoder.push_to_hub(repo_id, commit_message="Phase 3: Vision model merged")
        model.tokenizer.push_to_hub(repo_id)
        logger.info(f"Pushed to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
