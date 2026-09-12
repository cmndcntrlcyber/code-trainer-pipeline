"""
phase4c_rl/hf_skills/farca_grpo_entry.py

HF Jobs cloud entry for FARCA-GRPO training.

Subclasses GRPOTrainer to inject FARCA per-token advantages.
Model-agnostic: reads base_model from config, works for both
Qwen and Gemma pipelines.

Expected env vars:
    HF_TOKEN                    -- write access to adapter repos
    WANDB_API_KEY               -- optional
    PHASE4C_FARCA_PARAMS_JSON   -- full hyperparam blob
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
    from src.phase4c_rl.farca.config import FARCAConfig
    from src.phase4c_rl.farca.pipeline import FARCAPipeline

    # ── Load params ───────────────────────────────────────────────────
    params = json.loads(os.environ.get("PHASE4C_FARCA_PARAMS_JSON", "{}"))
    if not params:
        raise RuntimeError("PHASE4C_FARCA_PARAMS_JSON env var is empty")

    base_model_id = params.get("base_model", "Qwen/Qwen2.5-Coder-14B-Instruct")
    base_adapter = params.get("base_adapter")
    prompt_dataset = params.get("prompt_dataset")
    output_adapter = params.get("output_adapter")

    if not output_adapter:
        raise RuntimeError("output_adapter required")
    if not prompt_dataset:
        raise RuntimeError("prompt_dataset required")

    lr = float(params.get("learning_rate", 5e-7))
    num_generations = int(params.get("num_generations", 4))
    beta = float(params.get("beta", 0.04))
    max_completion_length = int(params.get("max_new_tokens", 1024))
    num_epochs = int(params.get("num_epochs", 1))
    batch_size = int(params.get("batch_size", 2))
    gradient_accumulation = int(params.get("gradient_accumulation", 4))
    lora_r = int(params.get("lora_r", 32))
    lora_alpha = int(params.get("lora_alpha", 64))
    lora_dropout = float(params.get("lora_dropout", 0.05))

    output_dir = Path(params.get("output_dir", "/tmp/phase4c-farca-grpo"))
    wandb_project = os.environ.get("WANDB_PROJECT", "rtpi-phase4c-rl")

    # FARCA-specific config
    farca_config = FARCAConfig(
        claim_extraction_backend=params.get("claim_extraction_backend", "rule_based"),
        nli_model_id=params.get("nli_model_id", "vectara/hallucination_evaluation_model"),
        nli_weight=float(params.get("nli_weight", 0.4)),
        rule_weight=float(params.get("rule_weight", 0.6)),
        sentence_encoder_id=params.get("sentence_encoder_id", "all-MiniLM-L6-v2"),
        k_rel=int(params.get("k_rel", 1)),
        mu=float(params.get("mu", 0.16)),
        tau=float(params.get("tau", 0.20)),
        warmup_steps=int(params.get("warmup_steps", 0)),
        device="cuda" if torch.cuda.is_available() else "cpu",
        tool_schema=NEXUS_TOOLS_V10,
    )

    logger.info("=" * 60)
    logger.info("PHASE 4C — FARCA-GRPO Training")
    logger.info(f"  base_model:       {base_model_id}")
    logger.info(f"  base_adapter:     {base_adapter or '(none)'}")
    logger.info(f"  output_adapter:   {output_adapter}")
    logger.info(f"  lr={lr} beta={beta} generations={num_generations}")
    logger.info(f"  FARCA warmup={farca_config.warmup_steps} mu={farca_config.mu} tau={farca_config.tau}")
    logger.info("=" * 60)

    # ── 1. Tokenizer ──────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── 2. Load prompts ───────────────────────────────────────────────
    logger.info("Loading prompt dataset: %s", prompt_dataset)
    ds = load_dataset(prompt_dataset, split="train")

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

    # ── 3. Load base model + merge adapters ───────────────────────────
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

    from src.utils import unwrap_clippable_linear
    unwrap_clippable_linear(model)

    dapt_adapter = params.get("dapt_adapter")
    if dapt_adapter:
        logger.info("Merging DAPT adapter: %s", dapt_adapter)
        try:
            model = PeftModel.from_pretrained(model, dapt_adapter, token=token)
            model = model.merge_and_unload()
        except (ValueError, OSError) as exc:
            logger.warning("DAPT adapter not found, skipping: %s", exc)

    if base_adapter:
        logger.info("Merging SFT adapter: %s", base_adapter)
        model = PeftModel.from_pretrained(model, base_adapter, token=token)
        model = model.merge_and_unload()

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

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

    # ── 4. Build FARCA pipeline ───────────────────────────────────────
    farca_pipeline = FARCAPipeline(farca_config, NEXUS_TOOLS_V10)
    logger.info("FARCA pipeline initialized")

    # ── 5. Reward wrapper (standard — FARCA reshapes advantages, not rewards)
    def reward_fn(prompts, completions, **kwargs):
        return tool_call_reward(completions)

    # ── 6. Training config ────────────────────────────────────────────
    if not os.environ.get("WANDB_API_KEY") and not os.environ.get("WANDB_MODE"):
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
        save_steps=50,
        save_total_limit=2,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        beta=beta,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name="phase4c-farca-grpo",
        push_to_hub=True,
        hub_model_id=output_adapter,
        hub_token=token,
        hub_strategy="checkpoint",
    )

    # ── 7. FARCAGRPOTrainer ───────────────────────────────────────────
    class FARCAGRPOTrainer(GRPOTrainer):
        def _generate_and_score_completions(self, inputs):
            output = super()._generate_and_score_completions(inputs)

            scalar_advantages = output["advantages"]
            completion_ids = output["completion_ids"]

            completions_text = [
                self.processing_class.decode(ids, skip_special_tokens=True)
                for ids in completion_ids
            ]

            per_token_advantages, farca_rewards = farca_pipeline.process_batch(
                completions=completions_text,
                completion_ids=completion_ids,
                tokenizer=self.processing_class,
                sequence_advantages=scalar_advantages,
                system_prompt=system_prompt,
            )

            output["advantages"] = per_token_advantages

            if hasattr(self, "_metrics"):
                mode = "train" if self.model.training else "eval"
                if farca_rewards:
                    self._metrics[mode]["farca/fact_reward_mean"].append(
                        sum(farca_rewards) / len(farca_rewards)
                    )

            return output

    trainer = FARCAGRPOTrainer(
        model=model,
        args=training_config,
        train_dataset=ds,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )

    logger.info("Starting FARCA-GRPO training...")
    trainer.train()
    logger.info("FARCA-GRPO training complete.")

    # ── 8. Save + push ────────────────────────────────────────────────
    best_dir = output_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    (best_dir / "phase4c-farca-grpo-result.json").write_text(json.dumps({
        "base_model": base_model_id,
        "base_adapter": base_adapter,
        "prompt_dataset": prompt_dataset,
        "method": "FARCA-GRPO",
        "learning_rate": lr,
        "beta": beta,
        "num_generations": num_generations,
        "farca_mu": farca_config.mu,
        "farca_tau": farca_config.tau,
        "farca_warmup_steps": farca_config.warmup_steps,
        "farca_k_rel": farca_config.k_rel,
    }, indent=2, default=str))

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
        commit_message=f"Phase 4c FARCA-GRPO — lr={lr} kl={beta} mu={farca_config.mu} tau={farca_config.tau}",
    )
    logger.info("Adapter pushed: https://huggingface.co/%s", output_adapter)

    shutil.rmtree(output_dir, ignore_errors=True)
    logger.info("Phase 4c FARCA-GRPO training job complete.")


if __name__ == "__main__":
    main()
