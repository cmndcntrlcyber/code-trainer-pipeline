"""
phase4_qwen_finetuning/configs/training_args.py

Build SFTConfig (trl 1.x) or TrainingArguments for Phase 4 cloud training jobs.
Optimised for A100-large (40GB VRAM) with BF16 and gradient checkpointing.
"""
from pathlib import Path

from .sweep_configs import SweepConfig


def build_training_args(
    cfg: SweepConfig,
    output_dir: str | Path,
    num_epochs: int = 1,
    max_seq_length: int = 2048,
    wandb_project: str = "code-trainer-phase4",
):
    try:
        from trl import SFTConfig
        return SFTConfig(
            output_dir=str(output_dir),
            max_seq_length=max_seq_length,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            weight_decay=0.01,
            bf16=True,
            tf32=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=25,
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb",
            run_name=f"phase4-{cfg.name}-epochs{num_epochs}",
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
        )
    except ImportError:
        from transformers import TrainingArguments
        return TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            weight_decay=0.01,
            bf16=True,
            tf32=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=25,
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb",
            run_name=f"phase4-{cfg.name}-epochs{num_epochs}",
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
        )
