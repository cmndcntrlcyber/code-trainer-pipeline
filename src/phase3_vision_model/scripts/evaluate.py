"""
phase3_vision_model/scripts/evaluate.py

Standalone evaluation script — run baseline or post-FT evaluation independently.

Usage:
    # Baseline (base model, no checkpoint):
    python -m src.phase3_vision_model.scripts.evaluate \
        --config src/config/config.yaml \
        --dataset-dir data/hf_dataset \
        --run-name baseline

    # Post-fine-tuning:
    python -m src.phase3_vision_model.scripts.evaluate \
        --config src/config/config.yaml \
        --dataset-dir data/hf_dataset \
        --checkpoint models/vision_model/best \
        --run-name finetuned
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from torch.utils.data import DataLoader

from src.config.settings import load_config
from src.phase3_vision_model.architecture.vision_model import CodeVisionModel
from src.phase3_vision_model.evaluation.evaluator import VisionModelEvaluator
from src.phase3_vision_model.training.collator import ScreenshotCodeCollator
from src.phase3_vision_model.training.dataset import ScreenshotCodeDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Evaluate CodeVisionModel")
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--dataset-dir", default="data/hf_dataset")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to fine-tuned checkpoint (omit for base model)")
    parser.add_argument("--run-name", default="eval",
                        help="Label for this evaluation run (e.g. 'baseline', 'finetuned')")
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--output-dir", default="models/vision_model/eval")
    args = parser.parse_args()

    config = load_config(args.config)
    vm_cfg = config.get("vision_model", {})

    if args.checkpoint:
        model = CodeVisionModel.from_pretrained(
            args.checkpoint,
            vision_model_id=vm_cfg.get("vision_encoder", "microsoft/swin-base-patch4-window7-224"),
            decoder_model_id=vm_cfg.get("decoder", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
        )
        logger.info(f"Loaded checkpoint: {args.checkpoint}")
    else:
        model = CodeVisionModel(
            vision_model_id=vm_cfg.get("vision_encoder", "microsoft/swin-base-patch4-window7-224"),
            decoder_model_id=vm_cfg.get("decoder", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
            lora_r=vm_cfg.get("lora_r", 16),
            lora_alpha=vm_cfg.get("lora_alpha", 32),
        )
        logger.info("Evaluating BASE model (no fine-tuning)")

    val_ds = ScreenshotCodeDataset(
        args.dataset_dir, split=args.split,
        tokenizer=model.tokenizer,
        feature_extractor=model.vision_encoder.feature_extractor,
        max_seq_length=vm_cfg.get("max_seq_length", 2048),
    )
    val_loader = DataLoader(
        val_ds, batch_size=2, shuffle=False,
        collate_fn=ScreenshotCodeCollator(pad_token_id=model.tokenizer.pad_token_id or 0),
    )

    evaluator = VisionModelEvaluator(model, model.tokenizer, model.vision_encoder.feature_extractor)
    metrics = evaluator.evaluate_from_dataloader(
        val_loader,
        num_samples=args.num_samples,
        run_name=args.run_name,
    )
    output_path = Path(args.output_dir) / f"{args.run_name}.json"
    evaluator.save_results(metrics, output_path)
    logger.info(f"Results: {metrics}")


if __name__ == "__main__":
    main()
