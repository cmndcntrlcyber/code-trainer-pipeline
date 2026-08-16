"""
phase4c_rl/hf_skills/dpo_entry.py

HF Jobs cloud entry for DPO (Direct Preference Optimization) training.

Loads the GRPO-refined adapter as the initial policy and further aligns it
using DPOTrainer with preference pairs (chosen/rejected). This is Stage 2
of Phase 4c Chain-of-Thought RL.

Expected env vars inside the job:
    HF_TOKEN                   — write access to adapter repos
    WANDB_API_KEY              — optional; if absent, runs offline
    PHASE4C_DPO_PARAMS_JSON    — full hyperparam blob (see below)

Params JSON shape:
    {
        "base_model": "Qwen/Qwen2.5-Coder-14B-Instruct",
        "base_adapter": "cmndcntrlcyber/qwen14b-code-trainer-v10-grpo",
        "preference_dataset": "cmndcntrlcyber/code-trainer-v10-dpo-pairs",
        "output_adapter": "cmndcntrlcyber/qwen14b-code-trainer-v10-dpo",
        "learning_rate": 1e-6,
        "beta": 0.1,
        "max_length": 4096,
        "num_epochs": 1,
        "batch_size": 2,
        "gradient_accumulation": 4,
    }
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


def main():
    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi, create_repo
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    # ── Load params ───────────────────────────────────────────────────────
    params = json.loads(os.environ.get("PHASE4C_DPO_PARAMS_JSON", "{}"))
    if not params:
        raise RuntimeError("PHASE4C_DPO_PARAMS_JSON env var is empty")

    base_model_id = params.get("base_model", "Qwen/Qwen2.5-Coder-14B-Instruct")
    base_adapter = params.get("base_adapter")
    preference_dataset = params.get("preference_dataset")
    output_adapter = params.get("output_adapter")

    if not output_adapter:
        raise RuntimeError("output_adapter required in PHASE4C_DPO_PARAMS_JSON")
    if not preference_dataset:
        raise RuntimeError("preference_dataset required in PHASE4C_DPO_PARAMS_JSON")

    lr = float(params.get("learning_rate", 1e-6))
    beta = float(params.get("beta", 0.1))
    max_length = int(params.get("max_length", 4096))
    max_prompt_length = int(params.get("max_prompt_length", 2048))
    num_epochs = int(params.get("num_epochs", 1))
    batch_size = int(params.get("batch_size", 2))
    gradient_accumulation = int(params.get("gradient_accumulation", 4))
    lora_r = int(params.get("lora_r", 32))
    lora_alpha = int(params.get("lora_alpha", 64))
    lora_dropout = float(params.get("lora_dropout", 0.05))

    output_dir = Path(params.get("output_dir", "/tmp/phase4c-dpo"))
    wandb_project = os.environ.get("WANDB_PROJECT", "rtpi-phase4c-rl")

    logger.info("=" * 60)
    logger.info("PHASE 4C — DPO Training (Stage 2)")
    logger.info(f"  base_model:          {base_model_id}")
    logger.info(f"  base_adapter:        {base_adapter or '(none)'}")
    logger.info(f"  preference_dataset:  {preference_dataset}")
    logger.info(f"  output_adapter:      {output_adapter}")
    logger.info(f"  lr={lr} beta={beta} max_length={max_length}")
    logger.info(f"  bs={batch_size} accum={gradient_accumulation} eff={batch_size * gradient_accumulation}")
    logger.info("=" * 60)

    # ── 1. Tokenizer ──────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── 2. Load preference dataset ───────────────────────────────────────
    logger.info("Loading preference dataset: %s", preference_dataset)
    ds = load_dataset(preference_dataset)

    train_ds = ds["train"]
    val_ds = ds.get("validation") or ds.get("test")

    logger.info("  train: %d pairs", len(train_ds))
    if val_ds:
        logger.info("  validation: %d pairs", len(val_ds))

    # ── 3. Load base model + merge GRPO adapter ──────────────────────────
    logger.info("Loading base model: %s (BF16)", base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False

    if base_adapter:
        logger.info("Loading and merging GRPO adapter: %s", base_adapter)
        model = PeftModel.from_pretrained(model, base_adapter)
        model = model.merge_and_unload()

    model = prepare_model_for_kbit_training(model)

    # Apply fresh LoRA for DPO training.
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── 4. Reference model (frozen copy) ─────────────────────────────────
    logger.info("Loading reference model for DPO (frozen)")
    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    if base_adapter:
        ref_model = PeftModel.from_pretrained(ref_model, base_adapter)
        ref_model = ref_model.merge_and_unload()
    ref_model.eval()

    # ── 5. Training config ────────────────────────────────────────────────
    wandb_mode = os.environ.get("WANDB_MODE")
    if not os.environ.get("WANDB_API_KEY") and not wandb_mode:
        os.environ["WANDB_MODE"] = "offline"

    training_config = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=lr,
        beta=beta,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=50 if val_ds else None,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name=f"phase4c-dpo",
    )

    # ── 6. Train ──────────────────────────────────────────────────────────
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    logger.info("Starting DPO training...")
    trainer.train()
    logger.info("DPO training complete.")

    # ── 7. Evaluate ───────────────────────────────────────────────────────
    eval_metrics = {}
    if val_ds:
        eval_metrics = trainer.evaluate()
        logger.info("DPO eval metrics: %s", json.dumps(eval_metrics, default=str))

    # ── 8. Save + push ────────────────────────────────────────────────────
    best_dir = output_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    # Save training metadata.
    (best_dir / "phase4c-dpo-result.json").write_text(json.dumps({
        "base_model": base_model_id,
        "base_adapter": base_adapter,
        "preference_dataset": preference_dataset,
        "learning_rate": lr,
        "beta": beta,
        "max_length": max_length,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "gradient_accumulation": gradient_accumulation,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "eval_metrics": eval_metrics,
    }, indent=2, default=str))

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        logger.warning("HF_TOKEN not set; skipping adapter push")
        return

    logger.info("Pushing adapter -> %s", output_adapter)
    create_repo(output_adapter, token=token, private=False, exist_ok=True)
    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(best_dir),
        repo_id=output_adapter,
        repo_type="model",
        commit_message=f"Phase 4c DPO — lr={lr} beta={beta}",
    )
    logger.info("Adapter pushed: https://huggingface.co/%s", output_adapter)

    shutil.rmtree(output_dir, ignore_errors=True)
    logger.info("Phase 4c DPO training job complete.")


if __name__ == "__main__":
    main()
