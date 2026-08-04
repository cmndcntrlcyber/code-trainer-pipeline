"""
phase2_preprocessing/scripts/build_v8_mixed_dataset.py

Build the V8 mixed training dataset — corrective action for V7's multilingual
hallucination and tool-call format mismatch.

Key changes from V7:
  - All tool-calling examples formatted via tokenizer.apply_chat_template(tools=...)
    to produce the EXACT token sequence Qwen2.5 expects natively
  - English-only filtering on all slices
  - Added Slice D (English instruction-following) to anchor the model's language

Four slices:
  A: Code generation     (~8K  from cmndcntrlcyber/code-trainer-offsec-dataset)
  B: Tool calling         (~12K from glaiveai/glaive-function-calling-v2, native format)
  C: Agent traces         (~10K from greghavens/fable-5-coding-and-debugging-traces)
  D: English instruction  (~8K  from teknium/OpenHermes-2.5)

Usage:
    uv run python -m src.phase2_preprocessing.scripts.build_v8_mixed_dataset \\
        --config src/config/config.yaml --no-push

    uv run python -m src.phase2_preprocessing.scripts.build_v8_mixed_dataset \\
        --config src/config/config.yaml
"""
import argparse
import json
import logging
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from datasets import Dataset, DatasetDict, load_dataset

from src.config.settings import load_config
from src.phase2_preprocessing.converters.tool_format_converter import (
    convert_fable5_messages_to_hermes,
    detect_tools_in_messages,
    validate_messages,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

NON_ASCII_THRESHOLD = 0.05


def is_english(text: str) -> bool:
    """Check if text is predominantly English (< 5% non-ASCII characters)."""
    if not text:
        return True
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return (non_ascii / len(text)) < NON_ASCII_THRESHOLD


def messages_are_english(messages: list[dict]) -> bool:
    """Check all messages in a conversation are English."""
    for msg in messages:
        content = msg.get("content", "") or ""
        if len(content) > 20 and not is_english(content):
            return False
    return True


# ── Slice A: Code generation (existing offsec dataset) ─────────────────────


def load_slice_a(
    dataset_id: str = "cmndcntrlcyber/code-trainer-offsec-dataset",
    subsample_size: int = 8000,
    seed: int = 42,
) -> list[dict]:
    """Load and subsample the existing code-trainer dataset (train split only)."""
    logger.info("Slice A: %s (train split, subsample %d)", dataset_id, subsample_size)
    ds = load_dataset(dataset_id, split="train")
    logger.info("  Loaded %d rows", len(ds))

    ds = ds.shuffle(seed=seed)
    n = min(subsample_size, len(ds))
    ds = ds.select(range(n))

    records = []
    skipped = 0
    for row in ds:
        messages = row.get("messages")
        if not messages or not validate_messages(messages):
            skipped += 1
            continue
        if not messages_are_english(messages):
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
    logger.info("  Slice A: %d valid records (%d skipped)", len(records), skipped)
    return records


# ── Slice B: Tool calling (glaive-v2, native Qwen format) ─────────────────


def parse_glaive_system(system_text: str) -> tuple[str, list[dict]]:
    """Parse glaive-v2 system field into (system_prompt, tool_defs)."""
    system_text = system_text.replace("SYSTEM: ", "", 1).strip()

    tools = []
    json_blocks = re.findall(r'\{[^{}]*"name"[^{}]*"parameters"[^{}]*\{[^}]*\}[^}]*\}', system_text, re.DOTALL)

    if not json_blocks:
        json_start = system_text.find("{")
        if json_start >= 0:
            rest = system_text[json_start:]
            for block in rest.split("\n{"):
                block = block.strip()
                if not block.startswith("{"):
                    block = "{" + block
                try:
                    tool_def = json.loads(block)
                    if "name" in tool_def:
                        tools.append({"type": "function", "function": tool_def})
                except json.JSONDecodeError:
                    pass

    if not tools:
        try:
            start = system_text.index("{")
            end = system_text.rindex("}") + 1
            raw = system_text[start:end]
            for candidate in [raw, f"[{raw}]"]:
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict) and "name" in parsed:
                        tools = [{"type": "function", "function": parsed}]
                    elif isinstance(parsed, list):
                        tools = [{"type": "function", "function": t} for t in parsed if isinstance(t, dict) and "name" in t]
                    break
                except json.JSONDecodeError:
                    continue
        except ValueError:
            pass

    base_prompt = system_text
    if tools:
        first_brace = system_text.find("{")
        if first_brace > 0:
            base_prompt = system_text[:first_brace].strip().rstrip("-").strip()

    return base_prompt, tools


def parse_glaive_chat(chat_text: str) -> list[dict]:
    """Parse glaive-v2 chat field into messages list."""
    messages = []
    parts = re.split(r'\n(?=USER:|FUNCTION RESPONSE:|ASSISTANT:)', chat_text.strip())

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.startswith("USER:"):
            content = part[5:].strip()
            if content:
                messages.append({"role": "user", "content": content})

        elif part.startswith("ASSISTANT:"):
            content = part[10:].strip().replace("<|endoftext|>", "").strip()
            if not content:
                continue
            fc_match = re.search(r'<functioncall>\s*(\{.*\})', content, re.DOTALL)
            if fc_match:
                try:
                    call = json.loads(fc_match.group(1))
                    name = call.get("name", "")
                    args_raw = call.get("arguments", {})
                    if isinstance(args_raw, str):
                        try:
                            args_raw = json.loads(args_raw)
                        except json.JSONDecodeError:
                            args_raw = {}
                    tool_call = {"name": name, "arguments": args_raw}
                    tc_content = json.dumps(tool_call)
                    messages.append({
                        "role": "assistant",
                        "content": f"<tool_call>\n{tc_content}\n</tool_call>",
                    })
                except json.JSONDecodeError:
                    messages.append({"role": "assistant", "content": content})
            else:
                messages.append({"role": "assistant", "content": content})

        elif part.startswith("FUNCTION RESPONSE:"):
            content = part[len("FUNCTION RESPONSE:"):].strip()
            messages.append({"role": "tool", "content": content})

    return messages


def load_slice_b(
    dataset_id: str = "glaiveai/glaive-function-calling-v2",
    max_rows: int = 12000,
    seed: int = 42,
) -> list[dict]:
    """Load glaive-v2 tool-calling dataset, filter to rows with actual tool calls."""
    logger.info("Slice B: %s (max %d rows with tool calls)", dataset_id, max_rows)
    ds = load_dataset(dataset_id, split="train")
    logger.info("  Loaded %d total rows", len(ds))

    ds = ds.shuffle(seed=seed)

    records = []
    skipped_no_call = 0
    skipped_parse = 0
    skipped_lang = 0

    for row in ds:
        if len(records) >= max_rows:
            break

        chat = row.get("chat", "")
        if "<functioncall>" not in chat:
            skipped_no_call += 1
            continue

        system_text = row.get("system", "")
        base_prompt, tools = parse_glaive_system(system_text)

        messages = parse_glaive_chat(chat)
        if not messages:
            skipped_parse += 1
            continue

        if base_prompt and not any(m["role"] == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": base_prompt})

        clean_messages = []
        for m in messages:
            clean_messages.append({
                "role": m["role"],
                "content": m["content"],
            })

        if not validate_messages(clean_messages):
            skipped_parse += 1
            continue

        if not messages_are_english(clean_messages):
            skipped_lang += 1
            continue

        records.append({
            "messages": clean_messages,
            "slice": "tool_calling",
            "source": dataset_id,
            "category": "function_calling",
            "has_tools": True,
            "n_turns": len(clean_messages),
        })

    logger.info("  Slice B: %d records (skipped: %d no-call, %d parse-fail, %d non-english)",
                len(records), skipped_no_call, skipped_parse, skipped_lang)
    return records


# ── Slice C: Agent traces (Fable 5, English-only) ─────────────────────────


def load_slice_c(
    dataset_id: str = "greghavens/fable-5-coding-and-debugging-traces",
    max_rows: int = 10000,
    seed: int = 42,
) -> list[dict]:
    """Load Fable 5 agent traces, filter to English, convert to Hermes format."""
    logger.info("Slice C: %s (max %d)", dataset_id, max_rows)
    ds = load_dataset(dataset_id, split="train")
    logger.info("  Loaded %d rows", len(ds))

    ds = ds.shuffle(seed=seed)

    records = []
    skipped = 0

    for row in ds:
        if len(records) >= max_rows:
            break

        raw_messages = row.get("messages")
        if not raw_messages:
            skipped += 1
            continue

        messages = convert_fable5_messages_to_hermes(raw_messages)
        if not validate_messages(messages):
            skipped += 1
            continue

        if not messages_are_english(messages):
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

    logger.info("  Slice C: %d records (%d skipped)", len(records), skipped)
    return records


# ── Slice D: English instruction following (OpenHermes-2.5) ───────────────

OPENHERMES_ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system"}

EXCLUDED_CATEGORIES = {"roleplay", "song", "poem", "story"}


def load_slice_d(
    dataset_id: str = "teknium/OpenHermes-2.5",
    max_rows: int = 8000,
    seed: int = 42,
) -> list[dict]:
    """Load OpenHermes-2.5 English instruction-following examples."""
    logger.info("Slice D: %s (max %d)", dataset_id, max_rows)

    target_categories = [
        "orca", "coding", "multiple_choice", "general", "writing",
        "brainstorming", "summarization", "classification",
    ]

    ds = load_dataset(dataset_id, split="train")
    logger.info("  Loaded %d total rows", len(ds))

    indices = list(range(len(ds)))
    random.seed(seed)
    random.shuffle(indices)

    records = []
    skipped = 0

    for idx in indices:
        if len(records) >= max_rows:
            break

        row = ds[idx]
        category = row.get("category", "")
        if category in EXCLUDED_CATEGORIES:
            skipped += 1
            continue

        convs = row.get("conversations", [])
        if not convs:
            skipped += 1
            continue

        messages = []
        sys_prompt = row.get("system_prompt")
        if sys_prompt and isinstance(sys_prompt, str) and sys_prompt.strip():
            messages.append({"role": "system", "content": sys_prompt.strip()})

        for turn in convs:
            role = OPENHERMES_ROLE_MAP.get(turn.get("from", ""), turn.get("from", ""))
            content = turn.get("value", "")
            if not content:
                continue
            messages.append({"role": role, "content": content})

        if not validate_messages(messages):
            skipped += 1
            continue

        if not messages_are_english(messages):
            skipped += 1
            continue

        records.append({
            "messages": messages,
            "slice": "instruction",
            "source": dataset_id,
            "category": category or "general",
            "has_tools": False,
            "n_turns": len(messages),
        })

    logger.info("  Slice D: %d records (%d skipped)", len(records), skipped)
    return records


# ── Build pipeline ────────────────────────────────────────────────────────


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
    stats = {}
    for split_name, ds in dataset_dict.items():
        slice_counts = Counter(ds["slice"])
        tool_counts = Counter(str(x) for x in ds["has_tools"])
        n_turns = ds["n_turns"]
        category_counts = Counter(ds["category"])

        stats[split_name] = {
            "total": len(ds),
            "by_slice": dict(slice_counts),
            "has_tools": dict(tool_counts),
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
        description="Build V8 mixed training dataset (native tool-call format + English-only)"
    )
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--output-dir", default="data/v8_mixed")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--slice-a-size", type=int, default=8000)
    parser.add_argument("--slice-b-size", type=int, default=12000)
    parser.add_argument("--slice-c-size", type=int, default=10000)
    parser.add_argument("--slice-d-size", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hub-repo", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    v8_cfg = config.get("v8_mixed", {})
    hub_repo = args.hub_repo or v8_cfg.get("dataset_name", "cmndcntrlcyber/code-trainer-v8-mixed")
    val_ratio = float(v8_cfg.get("val_split", 0.1))

    logger.info("=" * 60)
    logger.info("V8 Mixed Dataset Builder")
    logger.info("  Slice A: code-gen (%d)", args.slice_a_size)
    logger.info("  Slice B: tool-calling (%d)", args.slice_b_size)
    logger.info("  Slice C: agent traces (%d)", args.slice_c_size)
    logger.info("  Slice D: instruction (%d)", args.slice_d_size)
    logger.info("  Hub repo: %s", hub_repo)
    logger.info("=" * 60)

    records_a = load_slice_a(
        v8_cfg.get("slice_a", {}).get("source", "cmndcntrlcyber/code-trainer-offsec-dataset"),
        args.slice_a_size, args.seed,
    )
    records_b = load_slice_b(
        v8_cfg.get("slice_b", {}).get("source", "glaiveai/glaive-function-calling-v2"),
        args.slice_b_size, args.seed,
    )
    records_c = load_slice_c(
        v8_cfg.get("slice_c", {}).get("source", "greghavens/fable-5-coding-and-debugging-traces"),
        args.slice_c_size, args.seed,
    )
    records_d = load_slice_d(
        v8_cfg.get("slice_d", {}).get("source", "teknium/OpenHermes-2.5"),
        args.slice_d_size, args.seed,
    )

    all_records = records_a + records_b + records_c + records_d
    logger.info("Total: %d (A=%d B=%d C=%d D=%d)",
                len(all_records), len(records_a), len(records_b), len(records_c), len(records_d))

    dataset_dict = build_dataset_dict(all_records, args.seed, val_ratio)

    stats = compute_statistics(dataset_dict)
    for split_name, s in stats.items():
        logger.info("  %s: %d rows | slices: %s | turns: %s",
                     split_name, s["total"], s["by_slice"], s["n_turns"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving to %s", output_dir)
    dataset_dict.save_to_disk(str(output_dir))

    stats_path = output_dir / "statistics.json"
    stats_path.write_text(json.dumps(stats, indent=2))

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
                             commit_message="V8 mixed dataset: code_gen + tool_calling + agentic + instruction")

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(stats_path),
        path_in_repo="statistics.json",
        repo_id=hub_repo,
        repo_type="dataset",
        commit_message="Upload V8 statistics",
    )
    logger.info("Pushed: https://huggingface.co/datasets/%s", hub_repo)


if __name__ == "__main__":
    main()
