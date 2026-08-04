"""
phase2_preprocessing/scripts/build_v7_mixed_dataset.py

Build the V7 mixed training dataset from 3 HuggingFace Hub sources:

  Slice A: Code generation     (~8K from cmndcntrlcyber/code-trainer-offsec-dataset)
  Slice B: Tool/function calling (11.6K from NousResearch/hermes-function-calling-v1)
  Slice C: Agentic multi-turn   (12.5K from greghavens/fable-5-coding-and-debugging-traces)

Produces a unified ChatML-compatible dataset pushed to HuggingFace Hub
as cmndcntrlcyber/code-trainer-v7-mixed.

Usage:
    uv run python -m src.phase2_preprocessing.scripts.build_v7_mixed_dataset \\
        --config src/config/config.yaml

    # Dry run (no Hub push):
    uv run python -m src.phase2_preprocessing.scripts.build_v7_mixed_dataset \\
        --config src/config/config.yaml --no-push --output-dir data/v7_mixed

    # Custom subsample size for Slice A:
    uv run python -m src.phase2_preprocessing.scripts.build_v7_mixed_dataset \\
        --config src/config/config.yaml --slice-a-size 8000
"""
import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from src.config.settings import load_config
from src.phase2_preprocessing.converters.tool_format_converter import (
    convert_fable5_messages_to_hermes,
    convert_hermes_conversations_to_messages,
    detect_tools_in_messages,
    validate_messages,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

HERMES_CONFIGS = [
    "func_calling",
    "func_calling_singleturn",
    "glaive_func_calling",
    "json_mode_agentic",
    "json_mode_singleturn",
]


def load_slice_a(
    dataset_id: str = "cmndcntrlcyber/code-trainer-offsec-dataset",
    subsample_size: int = 8000,
    seed: int = 42,
) -> list[dict]:
    """Load and subsample the existing code-trainer dataset (Slice A, train split only)."""
    logger.info("Loading Slice A: %s (train split, subsample %d)", dataset_id, subsample_size)
    ds = load_dataset(dataset_id, split="train")
    logger.info("  Loaded %d rows from train split", len(ds))

    ds = ds.shuffle(seed=seed)
    n = min(subsample_size, len(ds))
    ds = ds.select(range(n))
    logger.info("  Subsampled to %d rows", n)

    records = []
    skipped = 0
    for row in ds:
        messages = row.get("messages")
        if not messages or not validate_messages(messages):
            skipped += 1
            continue
        records.append({
            "messages": messages,
            "slice": "code_gen",
            "source": dataset_id,
            "category": row.get("language", "unknown"),
            "has_tools": False,
            "n_turns": len(messages),
        })
    if skipped:
        logger.warning("  Skipped %d invalid rows in Slice A", skipped)
    logger.info("  Slice A: %d valid records", len(records))
    return records


def load_slice_b(
    dataset_id: str = "NousResearch/hermes-function-calling-v1",
) -> list[dict]:
    """Load all configs from the Hermes function-calling dataset (Slice B)."""
    logger.info("Loading Slice B: %s (%d configs)", dataset_id, len(HERMES_CONFIGS))

    records = []
    for config_name in HERMES_CONFIGS:
        logger.info("  Loading config: %s", config_name)
        try:
            ds = load_dataset(dataset_id, config_name, split="train")
        except Exception as e:
            logger.warning("  Failed to load config %s: %s", config_name, e)
            continue
        logger.info("    %d rows", len(ds))

        skipped = 0
        for row in ds:
            conversations = row.get("conversations")
            if not conversations:
                skipped += 1
                continue

            messages = convert_hermes_conversations_to_messages(conversations)
            if not validate_messages(messages):
                skipped += 1
                continue

            records.append({
                "messages": messages,
                "slice": "tool_calling",
                "source": dataset_id,
                "category": row.get("category", config_name),
                "has_tools": detect_tools_in_messages(messages),
                "n_turns": len(messages),
            })
        if skipped:
            logger.warning("    Skipped %d invalid rows in %s", skipped, config_name)

    logger.info("  Slice B: %d valid records total", len(records))
    return records


def load_slice_c(
    dataset_id: str = "greghavens/fable-5-coding-and-debugging-traces",
) -> list[dict]:
    """Load the Fable 5 coding agent traces (Slice C)."""
    logger.info("Loading Slice C: %s", dataset_id)
    ds = load_dataset(dataset_id, split="train")
    logger.info("  Loaded %d rows", len(ds))

    records = []
    skipped = 0
    for row in ds:
        raw_messages = row.get("messages")
        if not raw_messages:
            skipped += 1
            continue

        messages = convert_fable5_messages_to_hermes(raw_messages)
        if not validate_messages(messages):
            skipped += 1
            continue

        records.append({
            "messages": messages,
            "slice": "agentic",
            "source": dataset_id,
            "category": row.get("category", "coding"),
            "has_tools": detect_tools_in_messages(messages),
            "n_turns": len(messages),
        })
    if skipped:
        logger.warning("  Skipped %d invalid rows in Slice C", skipped)
    logger.info("  Slice C: %d valid records", len(records))
    return records


def build_dataset_dict(
    records: list[dict],
    seed: int = 42,
    val_ratio: float = 0.1,
) -> DatasetDict:
    """Combine all records into a shuffled DatasetDict with train/validation splits."""
    random.seed(seed)
    random.shuffle(records)

    val_size = int(len(records) * val_ratio)
    train_records = records[val_size:]
    val_records = records[:val_size]

    logger.info("Split: %d train, %d validation", len(train_records), len(val_records))

    return DatasetDict({
        "train": Dataset.from_list(train_records),
        "validation": Dataset.from_list(val_records),
    })


def compute_statistics(dataset_dict: DatasetDict) -> dict:
    """Compute dataset statistics for logging and saving."""
    stats = {}
    for split_name, ds in dataset_dict.items():
        slice_counts = Counter(ds["slice"])
        tool_counts = Counter(ds["has_tools"])
        n_turns = ds["n_turns"]
        category_counts = Counter(ds["category"])

        stats[split_name] = {
            "total": len(ds),
            "by_slice": dict(slice_counts),
            "has_tools": {str(k): v for k, v in tool_counts.items()},
            "n_turns": {
                "min": min(n_turns),
                "max": max(n_turns),
                "mean": round(sum(n_turns) / len(n_turns), 1),
            },
            "top_categories": dict(category_counts.most_common(20)),
        }
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Build V7 mixed training dataset from 3 HuggingFace Hub sources"
    )
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--output-dir", default="data/v7_mixed",
                        help="Local save directory")
    parser.add_argument("--no-push", action="store_true",
                        help="Skip Hub push (local build only)")
    parser.add_argument("--slice-a-size", type=int, default=8000,
                        help="Subsample size for Slice A (default: 8000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hub-repo", default=None,
                        help="Override Hub repo (default: from config or cmndcntrlcyber/code-trainer-v7-mixed)")
    args = parser.parse_args()

    config = load_config(args.config)
    v7_cfg = config.get("v7_mixed", {})

    slice_a_source = v7_cfg.get("slice_a", {}).get("source", "cmndcntrlcyber/code-trainer-offsec-dataset")
    slice_b_source = v7_cfg.get("slice_b", {}).get("source", "NousResearch/hermes-function-calling-v1")
    slice_c_source = v7_cfg.get("slice_c", {}).get("source", "greghavens/fable-5-coding-and-debugging-traces")
    hub_repo = args.hub_repo or v7_cfg.get("dataset_name", "cmndcntrlcyber/code-trainer-v7-mixed")
    val_ratio = float(v7_cfg.get("val_split", 0.1))

    logger.info("=" * 60)
    logger.info("V7 Mixed Dataset Builder")
    logger.info("  Slice A: %s (subsample %d)", slice_a_source, args.slice_a_size)
    logger.info("  Slice B: %s", slice_b_source)
    logger.info("  Slice C: %s", slice_c_source)
    logger.info("  Hub repo: %s", hub_repo)
    logger.info("=" * 60)

    records_a = load_slice_a(slice_a_source, args.slice_a_size, args.seed)
    records_b = load_slice_b(slice_b_source)
    records_c = load_slice_c(slice_c_source)

    all_records = records_a + records_b + records_c
    logger.info("Total records: %d (A=%d, B=%d, C=%d)",
                len(all_records), len(records_a), len(records_b), len(records_c))

    dataset_dict = build_dataset_dict(all_records, args.seed, val_ratio)

    stats = compute_statistics(dataset_dict)
    for split_name, split_stats in stats.items():
        logger.info("  %s: %d rows | slices: %s | tools: %s | turns: %s",
                     split_name, split_stats["total"],
                     split_stats["by_slice"], split_stats["has_tools"],
                     split_stats["n_turns"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving dataset to %s", output_dir)
    dataset_dict.save_to_disk(str(output_dir))

    stats_path = output_dir / "statistics.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    logger.info("Statistics saved to %s", stats_path)

    if args.no_push:
        logger.info("--no-push set; skipping Hub upload.")
        return

    import os
    from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        logger.error("HF_TOKEN not set; cannot push to Hub. Use --no-push for local build.")
        sys.exit(1)

    logger.info("Pushing dataset to %s", hub_repo)
    create_repo(hub_repo, token=token, repo_type="dataset", private=False, exist_ok=True)
    dataset_dict.push_to_hub(hub_repo, token=token, commit_message="V7 mixed dataset: code_gen + tool_calling + agentic")

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(stats_path),
        path_in_repo="statistics.json",
        repo_id=hub_repo,
        repo_type="dataset",
        commit_message="Upload V7 statistics",
    )
    logger.info("Dataset pushed: https://huggingface.co/datasets/%s", hub_repo)


if __name__ == "__main__":
    main()
