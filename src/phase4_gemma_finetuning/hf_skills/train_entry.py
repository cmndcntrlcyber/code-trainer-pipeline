"""
phase4_gemma_finetuning/hf_skills/train_entry.py

Entry script executed *inside* an HF Jobs A100 container for the Phase 4A
Gemma validation sweep.

Key differences from the Qwen variant:
    - Base model: google/gemma-4-12B-it (12B dense, native multimodal)
    - Chat template: Gemma 4 has no system role — system prompt is merged
      into the first user message
    - Tokenizer: Gemma 4 already has pad_token="<pad>" (no eos fallback)
    - Default max_seq_length: 4096 (up from 2048)

Expected env vars inside the job:
    HF_TOKEN             — write access to the per-config adapter repo
    WANDB_API_KEY        — optional; if absent, runs offline
    PHASE4_PARAMS_JSON   — full hyperparam blob (see SweepConfig + cloud.*)
    PHASE4_ADAPTER_REPO  — e.g. cmndcntrlcyber/gemma4-12b-code-trainer-standard
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


def _format_chat(example, tokenizer):
    """Render the Phase 2 messages list into a single chat-templated string.

    Gemma 4 does not support the system role. We merge any system message
    into the first user message so the instruction context is preserved.
    """
    messages = example["messages"]
    reformatted = []
    system_text = ""
    for msg in messages:
        if msg["role"] == "system":
            system_text = msg["content"]
        elif msg["role"] == "user":
            content = f"{system_text}\n\n{msg['content']}" if system_text else msg["content"]
            reformatted.append({"role": "user", "content": content})
            system_text = ""
        else:
            reformatted.append(msg)
    return {"text": tokenizer.apply_chat_template(
        reformatted, tokenize=False, add_generation_prompt=False,
    )}


def main():
    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi, create_repo
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.phase4_gemma_finetuning.configs.sweep_configs import (
        LORA_TARGET_MODULES,
        SweepConfig,
    )
    from src.phase4_gemma_finetuning.configs.training_args import build_training_args

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

    model_id = params.get("model_id", "google/gemma-4-12B-it")
    dataset_id = params.get("dataset_id", "cmndcntrlcyber/code-trainer-offsec-dataset")
    dataset_revision = params.get("dataset_revision", "main")
    num_epochs = int(params.get("num_epochs", 1))
    max_seq_length = int(params.get("max_seq_length", 4096))
    output_dir = Path(params.get("output_dir", "/tmp/phase4-gemma12b"))
    wandb_project = os.environ.get("WANDB_PROJECT", "rtpi-phase4-gemma4-12b")

    logger.info("=" * 60)
    logger.info(f"PHASE 4A GEMMA — {cfg.name} ({model_id})")
    logger.info(f"  LoRA r={cfg.lora_r} alpha={cfg.lora_alpha} lr={cfg.learning_rate}")
    logger.info(f"  bs={cfg.batch_size} accum={cfg.gradient_accumulation} eff={cfg.effective_batch}")
    logger.info(f"  dataset:    {dataset_id}@{dataset_revision}")
    logger.info(f"  adapter:    {adapter_repo}")
    logger.info(f"  output_dir: {output_dir}")
    logger.info("=" * 60)

    # ─── 1. Tokenizer + chat formatting ────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    # Gemma 4 has pad_token="<pad>" (id=0) — no eos fallback needed

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

    ds = ds.map(lambda ex: _format_chat(ex, tokenizer),
                remove_columns=[c for c in ds["train"].column_names if c != "messages"])
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
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ─── 3. Trainer (SFT) ──────────────────────────────────────────────────
    training_args = build_training_args(
        cfg=cfg,
        output_dir=output_dir,
        num_epochs=num_epochs,
        max_seq_length=max_seq_length,
        wandb_project=wandb_project,
    )

    from trl import SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
    )
    trainer.train()

    # ─── 4. Save best adapter + push ───────────────────────────────────────
    best_dir = output_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    eval_metrics = trainer.evaluate()
    (best_dir / "phase4-result.json").write_text(json.dumps({
        "config": cfg.__dict__,
        "model_id": model_id,
        "dataset": f"{dataset_id}@{dataset_revision}",
        "num_epochs": num_epochs,
        "eval_loss": eval_metrics.get("eval_loss"),
        "eval_runtime": eval_metrics.get("eval_runtime"),
    }, indent=2, default=str))
    logger.info(f"Sweep result: eval_loss={eval_metrics.get('eval_loss')}")

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
        commit_message=f"Phase 4A Gemma {cfg.name} sweep — eval_loss={eval_metrics.get('eval_loss')}",
    )
    logger.info(f"Adapter pushed: https://huggingface.co/{adapter_repo}")

    shutil.rmtree(output_dir, ignore_errors=True)
    logger.info("Phase 4A Gemma sweep job complete.")


if __name__ == "__main__":
    main()
