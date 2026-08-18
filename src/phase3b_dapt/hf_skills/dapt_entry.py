"""
phase3b_dapt/hf_skills/dapt_entry.py

Container-side entry point for Domain-Adaptive Pre-Training (DAPT).
Standard causal language modeling (next-token prediction) on raw offensive
security code and PDF knowledge corpus. No instruction formatting -- plain
text continuation.

Uses trl.SFTTrainer with dataset_text_field="text" for straightforward CLM.
LoRA adapter is pushed to Hub on completion.

Required env vars:
    HF_TOKEN                      -- read dataset + push adapter
    PHASE3B_DAPT_PARAMS_JSON      -- { base_model, dataset_id, lora_r, lora_alpha,
                                       lora_dropout, learning_rate, num_epochs,
                                       max_seq_length, output_adapter,
                                       wandb_project }
"""
import gc
import json
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

os.environ.setdefault("HF_HOME", "/workspace/.hf-cache")


def main():
    params = json.loads(os.environ.get("PHASE3B_DAPT_PARAMS_JSON", "{}"))

    base_model = params.get("base_model", "Qwen/Qwen2.5-Coder-14B-Instruct")
    dataset_id = params.get("dataset_id")
    lora_r = int(params.get("lora_r", 32))
    lora_alpha = int(params.get("lora_alpha", 64))
    lora_dropout = float(params.get("lora_dropout", 0.05))
    learning_rate = float(params.get("learning_rate", 5e-5))
    num_epochs = int(params.get("num_epochs", 1))
    max_seq_length = int(params.get("max_seq_length", 4096))
    output_adapter = params.get("output_adapter")
    wandb_project = params.get("wandb_project", "rtpi-phase3b-dapt")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN required")
    if not dataset_id:
        raise RuntimeError("dataset_id required in PHASE3B_DAPT_PARAMS_JSON")
    if not output_adapter:
        raise RuntimeError("output_adapter required in PHASE3B_DAPT_PARAMS_JSON")

    logger.info("=" * 60)
    logger.info("PHASE 3b -- Domain-Adaptive Pre-Training (DAPT)")
    logger.info("  base_model:      %s", base_model)
    logger.info("  dataset:         %s", dataset_id)
    logger.info("  lora_r:          %d", lora_r)
    logger.info("  lora_alpha:      %d", lora_alpha)
    logger.info("  lora_dropout:    %.3f", lora_dropout)
    logger.info("  learning_rate:   %.2e", learning_rate)
    logger.info("  num_epochs:      %d", num_epochs)
    logger.info("  max_seq_length:  %d", max_seq_length)
    logger.info("  output_adapter:  %s", output_adapter)
    logger.info("=" * 60)

    # W&B setup
    try:
        import wandb

        wandb.init(project=wandb_project, config=params)
        logger.info("W&B initialized: project=%s", wandb_project)
    except ImportError:
        logger.warning("wandb not installed; skipping W&B logging")
    except Exception as exc:
        logger.warning("W&B init failed: %s", exc)

    # Load dataset
    from datasets import load_dataset

    logger.info("Loading dataset: %s", dataset_id)
    dataset = load_dataset(dataset_id, split="train", token=token)
    logger.info("Dataset loaded: %d rows", len(dataset))

    max_documents = int(params.get("max_documents", 0))
    if max_documents and len(dataset) > max_documents:
        dataset = dataset.shuffle(seed=42).select(range(max_documents))
        logger.info("Capped dataset to %d rows (max_documents)", max_documents)

    # Load model + tokenizer
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info("Loading model: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True, token=token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=token,
        attn_implementation=attn_impl,
    )
    logger.info("Attention implementation: %s", attn_impl)
    model.config.use_cache = False

    # LoRA config -- target all attention + MLP projections
    from peft import LoraConfig, TaskType, get_peft_model

    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments — trl 1.3.0 uses SFTConfig (not TrainingArguments)
    # and all SFT-specific params (packing, max_length, dataset_text_field)
    # live on the config object, not as SFTTrainer kwargs.
    from trl import SFTConfig, SFTTrainer

    output_dir = "/workspace/dapt_output"

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataset_text_field="text",
        max_length=max_seq_length,
        packing=False,
        push_to_hub=True,
        hub_model_id=output_adapter,
        hub_token=token,
        hub_strategy="checkpoint",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting DAPT training...")
    trainer.train()
    logger.info("Training complete.")

    # Save adapter locally
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Push adapter to Hub
    from huggingface_hub import HfApi, create_repo

    logger.info("Pushing adapter to Hub: %s", output_adapter)
    create_repo(output_adapter, token=token, private=False, exist_ok=True)
    api = HfApi(token=token)
    api.upload_folder(
        folder_path=output_dir,
        repo_id=output_adapter,
        repo_type="model",
        commit_message="Phase 3b: DAPT offsec adapter",
    )
    logger.info(
        "Adapter pushed: https://huggingface.co/%s", output_adapter
    )

    # Cleanup
    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Finish W&B
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass

    logger.info("Phase 3b DAPT complete.")


if __name__ == "__main__":
    main()
