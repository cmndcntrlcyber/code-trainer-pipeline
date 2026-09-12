"""
phase4c_rl/data/collect_negatives.py

Generate negative examples for DPO training.

Two modes:
  --synthetic: Rule-based degradation of positive OCO sessions (no GPU needed).
               Generates refusals, stripped tool calls, truncated, and hallucinated responses.

  (default):   Model-based — runs the adapter through prompts and collects
               responses that fail the reward function (requires GPU).

Usage (synthetic — no GPU):
    python -m src.phase4c_rl.data.collect_negatives --synthetic \
        --input data/oco_converted/train.jsonl \
        --output-dir data/rl_data/negatives

Usage (model-based — requires GPU):
    set -a && source .env && set +a
    python -m src.phase4c_rl.data.collect_negatives \
        --config src/config/config.yaml \
        --output-dir data/rl_negatives \
        --n-responses 4
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.config.settings import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Reward threshold: responses scoring below this are saved as negatives.
NEGATIVE_THRESHOLD = 0.5

REFUSALS = [
    "I'm sorry, but I can't assist with that request as it may involve unauthorized access to systems.",
    "I appreciate your question, but I'm not able to help with activities that could compromise system security.",
    "That's an interesting question! However, I need to decline as this could involve unauthorized testing.",
    "I understand you're interested in security testing, but I can't provide guidance on that without proper authorization.",
    "I'd recommend consulting your organization's security team for guidance on this type of activity.",
]

WRONG_COMMANDS = [
    "ls -la /tmp", "cat /etc/passwd", "echo 'test'", "whoami", "pwd",
    "ping -c 1 localhost", "curl http://example.com", "netstat -an", "ps aux", "uname -a",
]


def _extract_prompt(messages):
    for msg in messages:
        if msg.get("role") == "user" and msg.get("content", "").strip():
            return msg["content"].strip()
    return None


def _extract_assistant_response(messages):
    parts = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("content", "").strip():
            parts.append(msg["content"])
    return "\n".join(parts) if parts else None


def _generate_synthetic(input_path, output_dir, seed=42):
    """Generate negatives by degrading positive OCO sessions."""
    import random
    random.seed(seed)

    sessions = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                sessions.append(json.loads(line))

    logger.info("Loaded %d sessions from %s", len(sessions), input_path)

    negatives = []
    stats = {"refusal": 0, "stripped": 0, "truncated": 0, "hallucinated": 0, "skipped": 0}

    for session in sessions:
        messages = session.get("messages", [])
        prompt = _extract_prompt(messages)
        response = _extract_assistant_response(messages)
        if not prompt or not response:
            stats["skipped"] += 1
            continue

        negatives.append({"prompt": prompt, "response": random.choice(REFUSALS)})
        stats["refusal"] += 1

        import re
        stripped = re.sub(r"<tool_call>.*?</tool_call>", "", response, flags=re.DOTALL)
        stripped = re.sub(r"\n{3,}", "\n\n", stripped).strip() or "I'll look into that for you."
        negatives.append({"prompt": prompt, "response": stripped})
        stats["stripped"] += 1

        cut = max(20, int(len(response) * 0.3))
        truncated = response[:cut].rsplit(" ", 1)[0] + "..."
        negatives.append({"prompt": prompt, "response": truncated})
        stats["truncated"] += 1

        tool_calls = re.findall(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL)
        if tool_calls:
            modified = response
            for tc in tool_calls:
                try:
                    tc_data = json.loads(tc)
                    if "arguments" in tc_data and "command" in tc_data["arguments"]:
                        tc_data["arguments"]["command"] = random.choice(WRONG_COMMANDS)
                        new_tc = f"<tool_call>{json.dumps(tc_data)}</tool_call>"
                        modified = modified.replace(f"<tool_call>{tc}</tool_call>", new_tc, 1)
                except (json.JSONDecodeError, TypeError):
                    continue
            if modified != response:
                negatives.append({"prompt": prompt, "response": modified})
                stats["hallucinated"] += 1

    out_file = output_dir / "negatives.json"
    out_file.write_text(json.dumps(negatives, indent=2, ensure_ascii=False))
    (output_dir / "collection_stats.json").write_text(json.dumps(stats, indent=2))

    logger.info("Generated %d synthetic negatives -> %s", len(negatives), out_file)
    logger.info("Stats: %s", json.dumps(stats))
    return negatives


def main():
    parser = argparse.ArgumentParser(
        description="Collect negative examples for DPO training"
    )
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate negatives by degrading positives (no GPU needed)")
    parser.add_argument("--input", default=None,
                        help="Path to train.jsonl (required for --synthetic)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--adapter", default=None,
                        help="Override adapter repo (default: config rl_training.grpo.base_adapter)")
    parser.add_argument("--output-dir", default="data/rl_negatives")
    parser.add_argument("--n-responses", type=int, default=4,
                        help="Number of responses to generate per prompt")
    parser.add_argument("--prompts-file", default=None,
                        help="JSON file with list of prompt strings (default: uses GRPO prompt dataset)")
    parser.add_argument("--max-prompts", type=int, default=200,
                        help="Maximum number of prompts to process")
    args = parser.parse_args()

    if args.synthetic:
        if not args.input:
            raise SystemExit("--input required with --synthetic")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _generate_synthetic(Path(args.input), output_dir, seed=args.seed)
        return

    config = load_config(args.config)
    rl_cfg = config.get("rl_training", {})
    grpo_cfg = rl_cfg.get("grpo", {})

    adapter_repo = args.adapter or grpo_cfg.get("base_adapter")
    base_model = grpo_cfg.get("base_model", "Qwen/Qwen2.5-Coder-14B-Instruct")

    if not adapter_repo:
        raise SystemExit(
            "No adapter specified. Pass --adapter or set rl_training.grpo.base_adapter in config."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lazy imports for GPU-heavy deps.
    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.phase4_qwen_finetuning.hf_skills.nexus_tools import NEXUS_TOOLS_V10
    from src.phase4c_rl.rewards.tool_call_reward import (
        tool_call_reward_detailed,
    )

    # ── Load prompts ──────────────────────────────────────────────────────
    if args.prompts_file:
        prompts = json.loads(Path(args.prompts_file).read_text())
        if isinstance(prompts, dict) and "prompts" in prompts:
            prompts = prompts["prompts"]
    else:
        prompt_ds_id = grpo_cfg.get("prompt_dataset")
        if prompt_ds_id:
            logger.info("Loading prompts from Hub: %s", prompt_ds_id)
            ds = load_dataset(prompt_ds_id, split="train")
            prompts = [row["prompt"] for row in ds]
        else:
            raise SystemExit(
                "No prompts source. Pass --prompts-file or set "
                "rl_training.grpo.prompt_dataset in config."
            )

    prompts = prompts[: args.max_prompts]
    logger.info("Loaded %d prompts", len(prompts))

    # ── Build system prompt (mirrors V10 eval) ────────────────────────────
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

    # ── Load model + adapter ──────────────────────────────────────────────
    logger.info("Loading base model: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    logger.info("Loading adapter: %s", adapter_repo)
    model = PeftModel.from_pretrained(model, adapter_repo)
    model = model.merge_and_unload()
    model.eval()

    # ── Generate + evaluate ───────────────────────────────────────────────
    negatives = []
    total_generated = 0

    for i, prompt in enumerate(prompts):
        logger.info("Prompt %d/%d: %s", i + 1, len(prompts), prompt[:80])

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            tools=NEXUS_TOOLS_V10,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        for j in range(args.n_responses):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            total_generated += 1

            detail = tool_call_reward_detailed(response)
            if detail["total_reward"] < NEGATIVE_THRESHOLD:
                negatives.append({
                    "prompt": prompt,
                    "response": response,
                    "reward": detail["total_reward"],
                    "components": detail["components"],
                    "generation_index": j,
                })
                logger.info(
                    "  [%d/%d] NEGATIVE (reward=%.3f)",
                    j + 1, args.n_responses, detail["total_reward"],
                )
            else:
                logger.info(
                    "  [%d/%d] OK (reward=%.3f)",
                    j + 1, args.n_responses, detail["total_reward"],
                )

    # ── Save results ──────────────────────────────────────────────────────
    out_file = output_dir / "negatives.json"
    out_file.write_text(json.dumps(negatives, indent=2))

    stats = {
        "total_prompts": len(prompts),
        "total_generated": total_generated,
        "total_negatives": len(negatives),
        "negative_rate": len(negatives) / total_generated if total_generated else 0,
        "adapter": adapter_repo,
        "base_model": base_model,
        "n_responses_per_prompt": args.n_responses,
        "negative_threshold": NEGATIVE_THRESHOLD,
    }
    (output_dir / "collection_stats.json").write_text(json.dumps(stats, indent=2))

    logger.info("=" * 60)
    logger.info(
        "Collected %d negatives from %d generations (%.1f%%)",
        len(negatives), total_generated,
        100 * len(negatives) / total_generated if total_generated else 0,
    )
    logger.info("Saved to: %s", out_file)


if __name__ == "__main__":
    main()
