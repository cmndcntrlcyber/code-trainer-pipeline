"""
phase2_preprocessing/scripts/build_v10_vision_dataset.py

Build the V10 vision mixed dataset for Gemma 26B multimodal SFT.

Combines:
    - Vision slice: ~4K screenshot-code pairs from the v2-multimodal dataset
      (base64 WebP images + offsec code), reformatted for Gemma 4 multimodal chat
    - Text slice: ~4K text-only samples from the v10-mixed dataset for regularization

The 50/50 mix preserves text performance while teaching the model to connect
its code knowledge to visual inputs (screenshots of code, terminal output, etc.).

Usage:
    python -m src.phase2_preprocessing.scripts.build_v10_vision_dataset \\
        --config src/config/pipeline-gemma26b.yml --no-push

    python -m src.phase2_preprocessing.scripts.build_v10_vision_dataset \\
        --config src/config/pipeline-gemma26b.yml
"""
import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from datasets import Dataset, DatasetDict, load_dataset

from src.config.nexus_identity import build_nexus_system_prompt
from src.config.settings import load_config
from src.phase2_preprocessing.converters.tool_format_converter import validate_messages
from src.phase4_qwen_finetuning.hf_skills.nexus_tools import NEXUS_TOOLS_V10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

NEXUS_SYSTEM_PROMPT = build_nexus_system_prompt(NEXUS_TOOLS_V10)

VISION_PROMPTS = [
    "What code is displayed in this screenshot?",
    "Transcribe the source code shown in this editor screenshot.",
    "Extract and reproduce the code visible in this screenshot.",
    "What is the exact code shown in this editor window?",
    "Reproduce the code from this screenshot.",
    "What source code is visible in this editor?",
    "Analyze this code screenshot and reproduce it.",
]


def load_vision_slice(
    dataset_id: str,
    revision: str,
    max_rows: int,
    seed: int,
) -> list[dict]:
    """Load screenshot-code pairs from the v2-multimodal dataset."""
    logger.info("Vision slice: %s@%s (max %d)", dataset_id, revision, max_rows)
    ds = load_dataset(dataset_id, revision=revision, split="train")
    logger.info("  Loaded %d rows", len(ds))

    ds = ds.shuffle(seed=seed)
    rng = random.Random(seed)

    records = []
    skipped = 0

    for row in ds:
        if len(records) >= max_rows:
            break

        messages = row.get("messages")
        image = row.get("image")

        if not messages or not image:
            skipped += 1
            continue

        # Find the assistant response (source code)
        assistant_content = None
        for msg in messages:
            if msg.get("role") == "assistant":
                assistant_content = msg.get("content", "")
                break

        if not assistant_content or len(assistant_content.strip()) < 20:
            skipped += 1
            continue

        prompt = rng.choice(VISION_PROMPTS)

        # Gemma 4 multimodal chat format:
        # The image placeholder is stored alongside the text in the user content.
        # At collation time, the base64 image is decoded to PIL and passed
        # to the processor. Here we store the structural format.
        multimodal_messages = [
            {"role": "system", "content": NEXUS_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_content},
        ]

        records.append({
            "messages": multimodal_messages,
            "image": image,
            "slice": "vision",
            "source": dataset_id,
            "language": row.get("language", "unknown"),
            "has_image": True,
        })

    logger.info("  Vision slice: %d records (%d skipped)", len(records), skipped)
    return records


def load_text_slice(
    dataset_id: str,
    max_rows: int,
    seed: int,
) -> list[dict]:
    """Load text-only samples from the v10-mixed dataset for regularization."""
    logger.info("Text slice: %s (max %d)", dataset_id, max_rows)
    ds = load_dataset(dataset_id, split="train")
    logger.info("  Loaded %d rows", len(ds))

    ds = ds.shuffle(seed=seed)

    records = []
    skipped = 0

    for row in ds:
        if len(records) >= max_rows:
            break

        messages = row.get("messages")
        if not messages or not validate_messages(messages):
            skipped += 1
            continue

        records.append({
            "messages": messages,
            "image": None,
            "slice": "text_regularization",
            "source": dataset_id,
            "language": row.get("category", "unknown"),
            "has_image": False,
        })

    logger.info("  Text slice: %d records (%d skipped)", len(records), skipped)
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Build V10 vision mixed dataset for Gemma 26B multimodal SFT"
    )
    parser.add_argument("--config", default="src/config/pipeline-gemma26b.yml")
    parser.add_argument("--output-dir", default="data/v10_vision")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--vision-size", type=int, default=4000)
    parser.add_argument("--text-size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hub-repo", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    vision_cfg = config.get("gemma_vision_sft", {})

    # Dataset sources
    multimodal_dataset = config.get("preprocessing", {}).get(
        "dataset_name", "cmndcntrlcyber/code-trainer-offsec-dataset"
    )
    multimodal_revision = config.get("preprocessing", {}).get(
        "multimodal_revision", "v2-multimodal"
    )
    text_dataset = config.get("v9_mixed", {}).get(
        "dataset_name", "cmndcntrlcyber/code-trainer-v9-mixed"
    )
    hub_repo = args.hub_repo or vision_cfg.get(
        "dataset_id", "cmndcntrlcyber/code-trainer-v10-vision"
    )

    logger.info("=" * 60)
    logger.info("V10 Vision Mixed Dataset Builder")
    logger.info("  Vision: %s@%s (%d)", multimodal_dataset, multimodal_revision, args.vision_size)
    logger.info("  Text:   %s (%d)", text_dataset, args.text_size)
    logger.info("  Hub:    %s", hub_repo)
    logger.info("=" * 60)

    vision_records = load_vision_slice(
        multimodal_dataset, multimodal_revision, args.vision_size, args.seed,
    )
    text_records = load_text_slice(text_dataset, args.text_size, args.seed)

    all_records = vision_records + text_records
    random.seed(args.seed)
    random.shuffle(all_records)

    logger.info("Total: %d (vision=%d, text=%d)",
                len(all_records), len(vision_records), len(text_records))

    # Split
    val_size = max(1, int(len(all_records) * 0.1))
    val_records = all_records[:val_size]
    train_records = all_records[val_size:]

    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_records),
        "validation": Dataset.from_list(val_records),
    })

    # Stats
    stats = {}
    for split_name, ds in dataset_dict.items():
        slice_counts = Counter(ds["slice"])
        lang_counts = Counter(ds["language"])
        stats[split_name] = {
            "total": len(ds),
            "by_slice": dict(slice_counts),
            "has_image": sum(1 for x in ds["has_image"] if x),
            "top_languages": dict(Counter(ds["language"]).most_common(10)),
        }
    for split_name, s in stats.items():
        logger.info("  %s: %d rows | slices: %s | with images: %d",
                     split_name, s["total"], s["by_slice"], s["has_image"])

    # Save locally
    project_root = Path(__file__).resolve().parents[3]
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dict.save_to_disk(str(output_dir))
    stats_path = output_dir / "statistics.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    logger.info("Saved to %s", output_dir)

    if args.no_push:
        logger.info("--no-push set; skipping Hub upload.")
        return

    import os
    from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        logger.error("HF_TOKEN not set; use --no-push for local build.")
        sys.exit(1)

    logger.info("Pushing to %s", hub_repo)
    create_repo(hub_repo, token=token, repo_type="dataset", private=False, exist_ok=True)
    dataset_dict.push_to_hub(hub_repo, token=token,
                             commit_message="V10 vision mixed: 4K multimodal + 4K text regularization")
    logger.info("Pushed: https://huggingface.co/datasets/%s", hub_repo)


if __name__ == "__main__":
    main()
