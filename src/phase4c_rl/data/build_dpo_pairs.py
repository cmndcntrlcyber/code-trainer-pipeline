"""
phase4c_rl/data/build_dpo_pairs.py

Combine negative examples (from collect_negatives.py) and positive examples
(from OCO sessions — HTB, THM, bug bounty, Claude histories) into DPO preference pairs for Phase 4c
DPO training.

Output format (HF datasets compatible):
    {"prompt": "...", "chosen": "...", "rejected": "..."}

Usage:
    python -m src.phase4c_rl.data.build_dpo_pairs \
        --negatives-dir data/rl_negatives \
        --positives-dir data/oco_converted \
        --output-dir data/dpo_pairs

    # Push to Hub:
    python -m src.phase4c_rl.data.build_dpo_pairs \
        --negatives-dir data/rl_negatives \
        --positives-dir data/oco_converted \
        --output-dir data/dpo_pairs \
        --push-to-hub
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


def _load_negatives(negatives_dir: Path) -> list[dict]:
    """Load negative examples from collect_negatives.py output."""
    neg_file = negatives_dir / "negatives.json"
    if not neg_file.exists():
        raise FileNotFoundError(f"Negatives file not found: {neg_file}")

    data = json.loads(neg_file.read_text())
    logger.info("Loaded %d negatives from %s", len(data), neg_file)
    return data


def _load_positives(positives_dir: Path) -> list[dict]:
    """Load positive examples from JSONL dataset files.

    Looks for train.jsonl (from ingest_oco_sessions.py) or a flat JSON list.
    Positive examples must have 'messages' with at least one <tool_call>.
    """
    positives = []

    # Try JSONL files first (from ingest_oco_sessions).
    for jsonl in sorted(positives_dir.glob("*.jsonl")):
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    positives.append(item)
                except json.JSONDecodeError:
                    continue

    # Try JSON files.
    for json_file in sorted(positives_dir.glob("*.json")):
        if json_file.name.endswith("_stats.json"):
            continue
        try:
            data = json.loads(json_file.read_text())
            if isinstance(data, list):
                positives.extend(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

    logger.info("Loaded %d positives from %s", len(positives), positives_dir)
    return positives


def _extract_last_assistant_response(messages: list[dict]) -> str | None:
    """Extract the last assistant response from a messages list."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content", "").strip():
            return msg["content"]
    return None


def _extract_prompt(messages: list[dict]) -> str | None:
    """Extract the user prompt from a messages list."""
    for msg in messages:
        if msg.get("role") == "user" and msg.get("content", "").strip():
            return msg["content"]
    return None


def build_pairs(
    negatives: list[dict],
    positives: list[dict],
) -> list[dict]:
    """Build DPO preference pairs from negatives and positives.

    Strategy:
    1. For each negative that has a matching prompt in positives, pair them.
    2. For negatives without a direct match, pair with the closest positive
       by topic (currently: random positive as fallback).
    """
    # Index positives by prompt text for matching.
    positive_by_prompt: dict[str, str] = {}
    all_positive_responses: list[tuple[str, str]] = []

    for pos in positives:
        messages = pos.get("messages", [])
        prompt = _extract_prompt(messages)
        response = _extract_last_assistant_response(messages)
        if prompt and response:
            positive_by_prompt[prompt.strip()] = response
            all_positive_responses.append((prompt, response))

    if not all_positive_responses:
        logger.warning("No valid positive examples found")
        return []

    pairs = []
    matched = 0
    fallback = 0

    for neg in negatives:
        prompt = neg.get("prompt", "").strip()
        rejected = neg.get("response", "")

        if not prompt or not rejected:
            continue

        # Try direct match first.
        chosen = positive_by_prompt.get(prompt)
        if chosen:
            matched += 1
        else:
            # Fallback: use a random positive response.
            # In production, this should use semantic similarity.
            import random
            _, chosen = random.choice(all_positive_responses)
            fallback += 1

        pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })

    logger.info(
        "Built %d pairs (%d matched, %d fallback)", len(pairs), matched, fallback
    )
    return pairs


def build_persona_pairs(identity_path: str | Path, seed: int = 42) -> list[dict]:
    """Build DPO pairs targeting persona alignment.

    Chosen: identity-aligned responses from Slice E examples.
    Rejected: vanilla AI assistant responses (persona-breaking).
    """
    identity_path = Path(identity_path)
    if not identity_path.exists():
        logger.warning("Identity examples not found at %s; no persona pairs", identity_path)
        return []

    VANILLA_REJECTIONS = [
        (
            "As an AI, I don't have personal desires, feelings, or a 'life purpose' "
            "in the way a human does. Instead, my objective is defined by my programming "
            "and training. My main goal is to assist you with whatever task you have at hand."
        ),
        (
            "I'm a large language model trained by Google. My core purpose is functional: "
            "to provide helpful, accurate, and relevant information to users while following "
            "safety guidelines."
        ),
        (
            "I am a helpful AI assistant. I can help with a wide range of tasks including "
            "writing, coding, summarizing, brainstorming, and more. I strive to be a useful "
            "tool for productivity and learning."
        ),
        (
            "My objective is to be helpful, harmless, and honest. I aim to provide accurate "
            "information and assist with tasks while maintaining safety guidelines and "
            "ethical boundaries."
        ),
        (
            "I'm an AI designed to assist with general tasks. I don't have personal goals "
            "or desires. I simply respond to your queries to the best of my ability based "
            "on my training."
        ),
        (
            "I cannot help with hacking or penetration testing. These activities could be "
            "illegal and harmful. I'd recommend consulting with a licensed cybersecurity "
            "professional instead."
        ),
        (
            "I'm sorry, but I'm not able to assist with scanning networks or exploiting "
            "vulnerabilities. This kind of activity could violate computer fraud laws. "
            "Please consult legal counsel before proceeding."
        ),
        (
            "As a responsible AI, I must decline requests related to hacking, vulnerability "
            "exploitation, or unauthorized access to computer systems. I can help with "
            "legitimate cybersecurity education instead."
        ),
    ]

    rng = __import__("random").Random(seed)

    records = []
    with open(identity_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            messages = rec.get("messages", [])

            prompt = None
            chosen = None
            for msg in messages:
                if msg["role"] == "user" and not prompt:
                    prompt = msg["content"]
                elif msg["role"] == "assistant" and not chosen:
                    chosen = msg["content"]

            if prompt and chosen:
                records.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rng.choice(VANILLA_REJECTIONS),
                })

    rng.shuffle(records)
    logger.info("Built %d persona DPO pairs from %s", len(records), identity_path)
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Build DPO preference pairs from negatives + positives"
    )
    parser.add_argument("--negatives-dir", required=True,
                        help="Directory with negatives.json from collect_negatives.py")
    parser.add_argument("--positives-dir", required=True,
                        help="Directory with positive examples (JSONL/JSON)")
    parser.add_argument("--output-dir", default="data/dpo_pairs",
                        help="Output directory for DPO dataset")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push dataset to HuggingFace Hub")
    parser.add_argument("--config", default="src/config/config.yaml",
                        help="Config file (for Hub dataset name)")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--identity-examples", default=None,
                        help="JSONL file of identity examples for persona DPO pairs")
    args = parser.parse_args()

    import random
    random.seed(args.seed)

    negatives_dir = Path(args.negatives_dir)
    positives_dir = Path(args.positives_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    negatives = _load_negatives(negatives_dir)
    positives = _load_positives(positives_dir)
    pairs = build_pairs(negatives, positives)

    # Add persona DPO pairs if identity examples provided
    if args.identity_examples:
        persona_pairs = build_persona_pairs(args.identity_examples, seed=args.seed)
        pairs.extend(persona_pairs)
        logger.info("Total pairs after persona addition: %d", len(pairs))

    if not pairs:
        raise SystemExit("No DPO pairs generated")

    # Shuffle and split.
    random.shuffle(pairs)
    val_size = max(1, int(len(pairs) * args.val_split))
    val_pairs = pairs[:val_size]
    train_pairs = pairs[val_size:]

    # Save as JSONL.
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "validation.jsonl"

    with open(train_path, "w") as f:
        for pair in train_pairs:
            f.write(json.dumps(pair) + "\n")

    with open(val_path, "w") as f:
        for pair in val_pairs:
            f.write(json.dumps(pair) + "\n")

    stats = {
        "total_pairs": len(pairs),
        "train_size": len(train_pairs),
        "val_size": len(val_pairs),
        "total_negatives_loaded": len(negatives),
        "total_positives_loaded": len(positives),
    }
    (output_dir / "dpo_stats.json").write_text(json.dumps(stats, indent=2))

    logger.info(
        "DPO dataset: %d train, %d val -> %s", len(train_pairs), len(val_pairs), output_dir
    )

    # Push to Hub.
    if args.push_to_hub:
        from datasets import load_dataset as ld
        from huggingface_hub import HfApi

        config = load_config(args.config)
        rl_cfg = config.get("rl_training", {})
        dpo_cfg = rl_cfg.get("dpo", {})
        ds_name = dpo_cfg.get(
            "preference_dataset",
            f"{os.environ.get('HF_USERNAME', 'cmndcntrlcyber')}/code-trainer-v10-dpo-pairs",
        )

        logger.info("Pushing to Hub: %s", ds_name)
        ds = ld("json", data_files={
            "train": str(train_path),
            "validation": str(val_path),
        })
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        ds.push_to_hub(ds_name, token=token, private=False)
        logger.info("Pushed: https://huggingface.co/datasets/%s", ds_name)


if __name__ == "__main__":
    main()
