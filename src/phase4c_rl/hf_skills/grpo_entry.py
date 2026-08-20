"""
phase4c_rl/hf_skills/grpo_entry.py

HF Jobs cloud entry for GRPO (Group Relative Policy Optimization) training.

Loads the V9 SFT adapter as the initial policy and refines it using
GRPOTrainer with the tool_call_reward function. This is Stage 1 of
Phase 4c Chain-of-Thought RL.

Expected env vars inside the job:
    HF_TOKEN                    — write access to adapter repos
    WANDB_API_KEY               — optional; if absent, runs offline
    PHASE4C_GRPO_PARAMS_JSON    — full hyperparam blob (see below)

Params JSON shape:
    {
        "base_model": "Qwen/Qwen2.5-Coder-14B-Instruct",
        "base_adapter": "cmndcntrlcyber/qwen14b-code-trainer-v9_mixed",
        "prompt_dataset": "cmndcntrlcyber/code-trainer-v10-grpo-prompts",
        "output_adapter": "cmndcntrlcyber/qwen14b-code-trainer-v10-grpo",
        "learning_rate": 5e-7,
        "num_generations": 4,
        "kl_coef": 0.1,
        "max_new_tokens": 1024,
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
    from trl import GRPOConfig, GRPOTrainer

    from src.phase4_qwen_finetuning.hf_skills.nexus_tools import NEXUS_TOOLS_V10
    from src.phase4c_rl.rewards.tool_call_reward import tool_call_reward

    # ── Load params ───────────────────────────────────────────────────────
    params = json.loads(os.environ.get("PHASE4C_GRPO_PARAMS_JSON", "{}"))
    if not params:
        raise RuntimeError("PHASE4C_GRPO_PARAMS_JSON env var is empty")

    base_model_id = params.get("base_model", "Qwen/Qwen2.5-Coder-14B-Instruct")
    base_adapter = params.get("base_adapter")
    prompt_dataset = params.get("prompt_dataset")
    output_adapter = params.get("output_adapter")

    if not output_adapter:
        raise RuntimeError("output_adapter required in PHASE4C_GRPO_PARAMS_JSON")
    if not prompt_dataset:
        raise RuntimeError("prompt_dataset required in PHASE4C_GRPO_PARAMS_JSON")

    lr = float(params.get("learning_rate", 5e-7))
    num_generations = int(params.get("num_generations", 4))
    beta = float(params.get("kl_coef", 0.04))
    max_completion_length = int(params.get("max_new_tokens", 1024))
    num_epochs = int(params.get("num_epochs", 1))
    batch_size = int(params.get("batch_size", 2))
    gradient_accumulation = int(params.get("gradient_accumulation", 4))
    lora_r = int(params.get("lora_r", 32))
    lora_alpha = int(params.get("lora_alpha", 64))
    lora_dropout = float(params.get("lora_dropout", 0.05))

    output_dir = Path(params.get("output_dir", "/tmp/phase4c-grpo"))
    wandb_project = os.environ.get("WANDB_PROJECT", "rtpi-phase4c-rl")

    logger.info("=" * 60)
    logger.info("PHASE 4C — GRPO Training (Stage 1)")
    logger.info(f"  base_model:       {base_model_id}")
    logger.info(f"  base_adapter:     {base_adapter or '(none)'}")
    logger.info(f"  prompt_dataset:   {prompt_dataset}")
    logger.info(f"  output_adapter:   {output_adapter}")
    logger.info(f"  lr={lr} kl_coef={kl_coef} generations={num_generations}")
    logger.info(f"  bs={batch_size} accum={gradient_accumulation} eff={batch_size * gradient_accumulation}")
    logger.info("=" * 60)

    # ── 1. Tokenizer ──────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── 2. Load prompts ───────────────────────────────────────────────────
    logger.info("Loading prompt dataset: %s", prompt_dataset)
    ds = load_dataset(prompt_dataset, split="train")

    # Build system prompt for tool-call context.
    system_prompt = (
        "You are Nexus, a local-first coding agent with direct filesystem and "
        "shell access. Call tools with JSON arguments matching each tool's schema:\n"
        + "\n".join(
            f"- {t['function']['name']}: {t['function']['description']}"
            for t in NEXUS_TOOLS_V10
        )
        + "\n\nWhen the task is complete, reply with a final message and do "
        "not request any more tool calls."
    )

    def format_prompt(example):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["prompt"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            tools=NEXUS_TOOLS_V10,
        )
        return {"prompt": text}

    ds = ds.map(format_prompt, remove_columns=[c for c in ds.column_names if c != "prompt"])
    logger.info("Formatted %d prompts", len(ds))

    # ── 3. Load base model + merge DAPT + merge SFT adapter ─────────────
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    logger.info("Loading base model: %s (BF16, attn=%s)", base_model_id, attn_impl)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
        token=token,
    )
    model.config.use_cache = False

    dapt_adapter = params.get("dapt_adapter")
    if dapt_adapter:
        logger.info("Merging DAPT adapter: %s", dapt_adapter)
        model = PeftModel.from_pretrained(model, dapt_adapter, token=token)
        model = model.merge_and_unload()
        logger.info("DAPT adapter merged into base weights")

    if base_adapter:
        logger.info("Merging SFT adapter: %s", base_adapter)
        model = PeftModel.from_pretrained(model, base_adapter, token=token)
        model = model.merge_and_unload()
        logger.info("SFT adapter merged")

    model = prepare_model_for_kbit_training(model)

    # Apply fresh LoRA for GRPO training.
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

    # ── 4. GRPO reward wrapper ────────────────────────────────────────────
    def reward_fn(prompts, completions, **kwargs):
        """Wrap tool_call_reward for GRPOTrainer interface.
        Completions are already decoded strings in trl 1.3.0+."""
        return tool_call_reward(completions)

    # ── 5. Training config ────────────────────────────────────────────────
    wandb_mode = os.environ.get("WANDB_MODE")
    if not os.environ.get("WANDB_API_KEY") and not wandb_mode:
        os.environ["WANDB_MODE"] = "offline"

    training_config = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=lr,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        beta=beta,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name="phase4c-grpo",
    )

    # ── 6. Train ──────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        args=training_config,
        train_dataset=ds,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )

    logger.info("Starting GRPO training...")
    trainer.train()
    logger.info("GRPO training complete.")

    # ── 7. Save + push ────────────────────────────────────────────────────
    best_dir = output_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    # Save training metadata.
    (best_dir / "phase4c-grpo-result.json").write_text(json.dumps({
        "base_model": base_model_id,
        "base_adapter": base_adapter,
        "prompt_dataset": prompt_dataset,
        "learning_rate": lr,
        "kl_coef": kl_coef,
        "num_generations": num_generations,
        "max_new_tokens": max_new_tokens,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "gradient_accumulation": gradient_accumulation,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
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
        commit_message=f"Phase 4c GRPO — lr={lr} kl={kl_coef} gen={num_generations}",
    )
    logger.info("Adapter pushed: https://huggingface.co/%s", output_adapter)

    shutil.rmtree(output_dir, ignore_errors=True)
    logger.info("Phase 4c GRPO training job complete.")


if __name__ == "__main__":
    main()
