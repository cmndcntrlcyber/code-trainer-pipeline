"""
phase4_qwen_finetuning/hf_skills/train_entry_v9.py

V9 training entry with two-phase curriculum for tool_call tag emission fix.

Phase A (80% of total steps): Full mixed dataset at lr=1e-4
Phase B (20% of total steps): Tool-calling-only subset at lr=2e-4

This gives the model a final "polish" pass on tool-call formatting after
learning the full mixed distribution.

Expected env vars inside the job:
    HF_TOKEN             — write access to the per-config adapter repo
    WANDB_API_KEY        — optional; if absent, runs offline
    PHASE4_PARAMS_JSON   — full hyperparam blob (see SweepConfig)
    PHASE4_ADAPTER_REPO  — e.g. cmndcntrlcyber/qwen14b-code-trainer-v9_mixed
"""
import json
import logging
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

os.environ.setdefault("HF_HOME", "/workspace/.hf-cache")

PHASE_A_RATIO = 0.8
PHASE_B_LR = 2e-4
PHASE_B_WARMUP = 0.1
PHASE_B_MIN_ROWS = 100


def _format_chat(example, tokenizer, max_length=4096):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
        text = tokenizer.decode(tokens, skip_special_tokens=False)
    return {"text": text}


def main():
    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi, create_repo
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer

    from src.phase4_qwen_finetuning.configs.sweep_configs import (
        LORA_TARGET_MODULES,
        SweepConfig,
    )
    from src.phase4_qwen_finetuning.configs.training_args import build_training_args

    params = json.loads(os.environ.get("PHASE4_PARAMS_JSON", "{}"))
    if not params:
        raise RuntimeError("PHASE4_PARAMS_JSON env var is empty")

    adapter_repo = os.environ.get("PHASE4_ADAPTER_REPO") or params.get("adapter_repo")
    if not adapter_repo:
        raise RuntimeError("PHASE4_ADAPTER_REPO env var (or params.adapter_repo) required")

    cfg = SweepConfig(
        name=params["name"],
        lora_r=params["lora_r"],
        lora_alpha=params["lora_alpha"],
        learning_rate=float(params["learning_rate"]),
        batch_size=params["batch_size"],
        gradient_accumulation=params["gradient_accumulation"],
    )

    model_id = params.get("model_id", "Qwen/Qwen2.5-Coder-14B-Instruct")
    dataset_id = params.get("dataset_id", "cmndcntrlcyber/code-trainer-v9-mixed")
    dataset_revision = params.get("dataset_revision", "main")
    num_epochs = int(params.get("num_epochs", 1))
    max_seq_length = int(params.get("max_seq_length", 4096))
    output_dir = Path(params.get("output_dir", "/tmp/phase4-qwen14b-v9"))
    wandb_project = os.environ.get("WANDB_PROJECT", "rtpi-phase4-qwen14b")

    logger.info("=" * 60)
    logger.info(f"PHASE 4 V9 — {cfg.name} ({model_id})")
    logger.info(f"  LoRA r={cfg.lora_r} alpha={cfg.lora_alpha} lr={cfg.learning_rate}")
    logger.info(f"  bs={cfg.batch_size} accum={cfg.gradient_accumulation} eff={cfg.effective_batch}")
    logger.info(f"  dataset:    {dataset_id}@{dataset_revision}")
    logger.info(f"  adapter:    {adapter_repo}")
    logger.info(f"  curriculum: Phase A {PHASE_A_RATIO:.0%} lr={cfg.learning_rate} | Phase B {1-PHASE_A_RATIO:.0%} lr={PHASE_B_LR}")
    logger.info("=" * 60)

    # ─── 1. Tokenizer + chat formatting ────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading dataset {dataset_id}@{dataset_revision}")
    ds = load_dataset(dataset_id, revision=dataset_revision)

    train_limit = params.get("train_limit") or os.environ.get("PHASE4_TRAIN_LIMIT")
    val_limit = params.get("val_limit") or os.environ.get("PHASE4_VAL_LIMIT")
    if train_limit:
        n = min(int(train_limit), len(ds["train"]))
        ds["train"] = ds["train"].select(range(n))
        logger.info(f"  train sliced to first {n} rows (PHASE4_TRAIN_LIMIT)")
    if val_limit and "validation" in ds:
        n = min(int(val_limit), len(ds["validation"]))
        ds["validation"] = ds["validation"].select(range(n))
        logger.info(f"  validation sliced to first {n} rows (PHASE4_VAL_LIMIT)")

    ds = ds.map(lambda ex: _format_chat(ex, tokenizer, max_length=max_seq_length),
                remove_columns=ds["train"].column_names)
    logger.info(f"  splits: {list(ds.keys())}  train={len(ds['train'])} val={len(ds['validation'])}")

    # ─── 2. Base model + LoRA ──────────────────────────────────────────────
    logger.info(f"Loading {model_id} (BF16)")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=float(params.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ─── 3. Calculate curriculum step counts ───────────────────────────────
    total_train_samples = len(ds["train"])
    steps_per_epoch = total_train_samples // (cfg.batch_size * cfg.gradient_accumulation)
    total_steps = steps_per_epoch * num_epochs
    phase_a_steps = int(total_steps * PHASE_A_RATIO)
    phase_b_steps = total_steps - phase_a_steps

    logger.info(f"  Curriculum: total_steps={total_steps} phase_a={phase_a_steps} phase_b={phase_b_steps}")

    # ─── 4. Phase A: Full mixed dataset ────────────────────────────────────
    logger.info("=" * 40)
    logger.info("PHASE A: Full mixed dataset (lr=%s, %d steps)", cfg.learning_rate, phase_a_steps)
    logger.info("=" * 40)

    phase_a_args = build_training_args(
        cfg=cfg,
        output_dir=output_dir / "phase_a",
        num_epochs=num_epochs,
        max_seq_length=max_seq_length,
        wandb_project=wandb_project,
        dataset_text_field="text",
        packing=False,
    )
    phase_a_args.max_steps = phase_a_steps
    phase_a_args.load_best_model_at_end = False
    phase_a_args.run_name = f"phase4-{cfg.name}-v9-phaseA"

    trainer_a = SFTTrainer(
        model=model,
        args=phase_a_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
    )
    trainer_a.train()

    phase_a_loss = trainer_a.evaluate().get("eval_loss")
    logger.info(f"Phase A complete: eval_loss={phase_a_loss}")

    # ─── 5. Phase B: Tool-calling-only polish ──────────────────────────────
    ds_tool = ds["train"].filter(lambda x: "<tool_call>" in x["text"])
    logger.info(f"Phase B dataset: {len(ds_tool)} tool-calling rows (from {total_train_samples} total)")

    if len(ds_tool) < PHASE_B_MIN_ROWS:
        logger.warning(f"Only {len(ds_tool)} tool-calling rows — skipping Phase B (min {PHASE_B_MIN_ROWS})")
        final_trainer = trainer_a
    else:
        logger.info("=" * 40)
        logger.info("PHASE B: Tool-calling polish (lr=%s, %d steps)", PHASE_B_LR, phase_b_steps)
        logger.info("=" * 40)

        phase_b_cfg = SweepConfig(
            name=cfg.name,
            lora_r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            learning_rate=PHASE_B_LR,
            batch_size=cfg.batch_size,
            gradient_accumulation=cfg.gradient_accumulation,
        )

        phase_b_args = build_training_args(
            cfg=phase_b_cfg,
            output_dir=output_dir / "phase_b",
            num_epochs=num_epochs,
            max_seq_length=max_seq_length,
            wandb_project=wandb_project,
            dataset_text_field="text",
            packing=False,
        )
        phase_b_args.max_steps = phase_b_steps
        phase_b_args.warmup_ratio = PHASE_B_WARMUP
        phase_b_args.load_best_model_at_end = True
        phase_b_args.run_name = f"phase4-{cfg.name}-v9-phaseB"

        trainer_b = SFTTrainer(
            model=trainer_a.model,
            args=phase_b_args,
            train_dataset=ds_tool,
            eval_dataset=ds["validation"],
            processing_class=tokenizer,
        )
        trainer_b.train()
        final_trainer = trainer_b

    # ─── 6. Save best adapter + push ───────────────────────────────────────
    best_dir = output_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    final_trainer.model.save_pretrained(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    eval_metrics = final_trainer.evaluate()
    (best_dir / "phase4-result.json").write_text(json.dumps({
        "config": cfg.__dict__,
        "model_id": model_id,
        "dataset": f"{dataset_id}@{dataset_revision}",
        "num_epochs": num_epochs,
        "curriculum": {
            "phase_a_steps": phase_a_steps,
            "phase_a_lr": cfg.learning_rate,
            "phase_a_eval_loss": phase_a_loss,
            "phase_b_steps": phase_b_steps,
            "phase_b_lr": PHASE_B_LR,
            "phase_b_tool_rows": len(ds_tool),
        },
        "eval_loss": eval_metrics.get("eval_loss"),
        "eval_runtime": eval_metrics.get("eval_runtime"),
    }, indent=2, default=str))
    logger.info(f"Final result: eval_loss={eval_metrics.get('eval_loss')}")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        logger.warning("HF_TOKEN not set; skipping adapter push")
        return

    logger.info(f"Pushing adapter → {adapter_repo}")
    create_repo(adapter_repo, token=token, private=False, exist_ok=True)
    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(best_dir),
        repo_id=adapter_repo,
        repo_type="model",
        commit_message=f"Phase 4 V9 {cfg.name} curriculum — eval_loss={eval_metrics.get('eval_loss')}",
    )
    logger.info(f"Adapter pushed: https://huggingface.co/{adapter_repo}")

    shutil.rmtree(output_dir, ignore_errors=True)
    logger.info("Phase 4 V9 training job complete.")


if __name__ == "__main__":
    main()
