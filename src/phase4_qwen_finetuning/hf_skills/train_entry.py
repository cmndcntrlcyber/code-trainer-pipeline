"""
phase4_qwen_finetuning/hf_skills/train_entry.py

Entry script executed *inside* an HF Jobs A100 container for the Phase 4A
validation sweep. Mirrors the Phase 3 pattern at
src/phase3_vision_model/hf_skills/train_entry.py.

Responsibilities:
    1. Parse hyperparams from PHASE4_PARAMS_JSON env (per-config sweep run).
    2. Load the text-only dataset from the Hub.
    3. Apply the Qwen chat template to each row's `messages` column.
    4. Load Qwen2.5-Coder-14B-Instruct in BF16 with LoRA via PEFT.
    5. Run SFTTrainer for the sweep epoch(s).
    6. Push the LoRA adapter (best checkpoint) to PHASE4_ADAPTER_REPO.

Expected env vars inside the job:
    HF_TOKEN             — write access to the per-config adapter repo
    WANDB_API_KEY        — optional; if absent, runs offline
    PHASE4_PARAMS_JSON   — full hyperparam blob (see SweepConfig + cloud.* below)
    PHASE4_ADAPTER_REPO  — e.g. cmndcntrlcyber/qwen14b-code-trainer-conservative
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

# Persistent HF cache so retries don't re-download the 28 GB Qwen-14B weights.
os.environ.setdefault("HF_HOME", "/workspace/.hf-cache")


def _format_chat(example, tokenizer, max_length=4096):
    """Render the Phase 2 messages list into a single chat-templated string, truncated."""
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
    dataset_id = params.get("dataset_id", "cmndcntrlcyber/code-trainer-offsec-dataset")
    dataset_revision = params.get("dataset_revision", "main")
    num_epochs = int(params.get("num_epochs", 1))
    max_seq_length = int(params.get("max_seq_length", 2048))
    output_dir = Path(params.get("output_dir", "/tmp/phase4-qwen14b"))
    wandb_project = os.environ.get("WANDB_PROJECT", "rtpi-phase4-qwen14b")

    logger.info("=" * 60)
    logger.info(f"PHASE 4A — {cfg.name} ({model_id})")
    logger.info(f"  LoRA r={cfg.lora_r} alpha={cfg.lora_alpha} lr={cfg.learning_rate}")
    logger.info(f"  bs={cfg.batch_size} accum={cfg.gradient_accumulation} eff={cfg.effective_batch}")
    logger.info(f"  dataset:    {dataset_id}@{dataset_revision}")
    logger.info(f"  adapter:    {adapter_repo}")
    logger.info(f"  output_dir: {output_dir}")
    logger.info("=" * 60)

    # ─── 1. Tokenizer + chat formatting ────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading dataset {dataset_id}@{dataset_revision}")
    ds = load_dataset(dataset_id, revision=dataset_revision)

    # Optional dataset slice — lets us land a sweep run inside the HF Skills
    # ~6.5h hard cap when full-epoch training would otherwise time out.
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
                remove_columns=[c for c in ds["train"].column_names if c != "messages"])
    logger.info(f"  splits: {list(ds.keys())}  train={len(ds['train'])} val={len(ds['validation'])}")

    # ─── 2. Base model + LoRA ──────────────────────────────────────────────
    logger.info(f"Loading {model_id} (BF16)")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False  # required for gradient checkpointing
    model = prepare_model_for_kbit_training(model)  # safe even without quantization

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

    # Drop a quick result summary alongside the adapter
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
        commit_message=f"Phase 4A {cfg.name} sweep — eval_loss={eval_metrics.get('eval_loss')}",
    )
    logger.info(f"Adapter pushed: https://huggingface.co/{adapter_repo}")

    shutil.rmtree(output_dir, ignore_errors=True)
    logger.info("Phase 4A sweep job complete.")


if __name__ == "__main__":
    main()
