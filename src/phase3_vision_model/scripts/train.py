"""
phase3_vision_model/scripts/train.py

Phase 3 training entry point.
Runs baseline evaluation before training, then trains, then post-FT evaluation.
Satisfies Capstone Requirements #2, #3, #4, #5.

Usage:
    python -m src.phase3_vision_model.scripts.train \
        --config src/config/config.yaml \
        --dataset-dir data/hf_dataset \
        --output-dir models/vision_model

    # Skip baseline evaluation (faster iteration):
    python -m src.phase3_vision_model.scripts.train \
        --config src/config/config.yaml --skip-baseline
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.config.settings import load_config
from src.phase3_vision_model.architecture.vision_model import CodeVisionModel
from src.phase3_vision_model.training.trainer import VisionModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Train CodeVisionModel")
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--dataset-dir", default="data/hf_dataset")
    parser.add_argument("--output-dir", default="models/vision_model")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip pre-training baseline evaluation")
    parser.add_argument("--skip-posteval", action="store_true",
                        help="Skip post-training evaluation")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="Resume training from this epoch (0-indexed, loads epoch-N checkpoint)")
    parser.add_argument("--resume-from", default=None,
                        help="Checkpoint dir to load from (defaults to output-dir/epoch-{start-epoch})")
    args = parser.parse_args()

    config = load_config(args.config)
    vm_cfg = config.get("vision_model", {})

    output_dir = Path(args.output_dir)
    dataset_dir = Path(args.dataset_dir)
    captures_dir_str = config.get("data_collection", {}).get("captures_dir")
    captures_dir = Path(captures_dir_str) if captures_dir_str else None

    logger.info("=" * 60)
    logger.info("PHASE 3: Vision Model Training")
    logger.info("=" * 60)

    start_epoch = args.start_epoch

    # Build model
    model = CodeVisionModel(
        vision_model_id=vm_cfg.get("vision_encoder", "microsoft/swin-base-patch4-window7-224"),
        decoder_model_id=vm_cfg.get("decoder", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
        lora_r=vm_cfg.get("lora_r", 16),
        lora_alpha=vm_cfg.get("lora_alpha", 32),
        lora_dropout=vm_cfg.get("lora_dropout", 0.05),
        device_profile=vm_cfg.get("device_profile", "a100"),
    )

    # Resume from checkpoint if requested
    if start_epoch > 0:
        import torch
        from peft import PeftModel
        checkpoint_dir = Path(args.resume_from) if args.resume_from else output_dir / f"epoch-{start_epoch}"
        logger.info(f"Resuming from checkpoint: {checkpoint_dir}")
        projector_path = checkpoint_dir / "projector.pt"
        if projector_path.exists():
            model.projector.load_state_dict(torch.load(projector_path, map_location="cpu"))
            logger.info("Projector weights loaded")
        lora_path = checkpoint_dir / "decoder_lora"
        if lora_path.exists():
            model.decoder = PeftModel.from_pretrained(
                model.decoder.base_model.model, str(lora_path)
            )
            logger.info("LoRA adapters loaded")

    # Baseline evaluation (Capstone Req #2)
    if not args.skip_baseline and start_epoch == 0:
        logger.info("Running BASELINE evaluation (pre-fine-tune)...")
        from src.phase3_vision_model.evaluation.evaluator import VisionModelEvaluator
        from src.phase3_vision_model.training.collator import ScreenshotCodeCollator
        from src.phase3_vision_model.training.dataset import ScreenshotCodeDataset
        from torch.utils.data import DataLoader

        val_ds = ScreenshotCodeDataset(
            dataset_dir, split="validation",
            tokenizer=model.tokenizer,
            feature_extractor=model.vision_encoder.feature_extractor,
            max_seq_length=vm_cfg.get("max_seq_length", 2048),
            captures_dir=captures_dir,
        )
        val_loader = DataLoader(
            val_ds, batch_size=2, shuffle=False,
            collate_fn=ScreenshotCodeCollator(pad_token_id=model.tokenizer.pad_token_id or 0),
        )
        evaluator = VisionModelEvaluator(
            model, model.tokenizer, model.vision_encoder.feature_extractor
        )
        baseline_metrics = evaluator.evaluate_from_dataloader(
            val_loader, num_samples=100, run_name="baseline"
        )
        evaluator.save_results(baseline_metrics, output_dir / "eval" / "baseline.json")
        logger.info(f"Baseline: {baseline_metrics}")

    # Train (Capstone Req #3, #5)
    trainer = VisionModelTrainer(model, dataset_dir, output_dir, config)
    trainer.train(start_epoch=start_epoch)

    # Post-fine-tuning evaluation (Capstone Req #4)
    if not args.skip_posteval:
        logger.info("Running POST-FINE-TUNING evaluation...")
        from src.phase3_vision_model.evaluation.evaluator import VisionModelEvaluator
        from src.phase3_vision_model.training.collator import ScreenshotCodeCollator
        from src.phase3_vision_model.training.dataset import ScreenshotCodeDataset
        from torch.utils.data import DataLoader
        import wandb

        val_ds = ScreenshotCodeDataset(
            dataset_dir, split="validation",
            tokenizer=model.tokenizer,
            feature_extractor=model.vision_encoder.feature_extractor,
            max_seq_length=vm_cfg.get("max_seq_length", 2048),
            captures_dir=captures_dir,
        )
        val_loader = DataLoader(
            val_ds, batch_size=2, shuffle=False,
            collate_fn=ScreenshotCodeCollator(pad_token_id=model.tokenizer.pad_token_id or 0),
        )

        # Reload best model for evaluation
        best_model = CodeVisionModel.from_pretrained(
            output_dir / "best",
            vision_model_id=vm_cfg.get("vision_encoder", "microsoft/swin-base-patch4-window7-224"),
            decoder_model_id=vm_cfg.get("decoder", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
        )
        evaluator = VisionModelEvaluator(
            best_model, best_model.tokenizer, best_model.vision_encoder.feature_extractor
        )
        ft_metrics = evaluator.evaluate_from_dataloader(
            val_loader, num_samples=100, run_name="finetuned"
        )
        evaluator.save_results(ft_metrics, output_dir / "eval" / "finetuned.json")
        logger.info(f"Fine-tuned: {ft_metrics}")

    logger.info("Phase 3 complete.")


if __name__ == "__main__":
    main()
