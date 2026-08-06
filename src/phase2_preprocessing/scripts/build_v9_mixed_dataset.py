"""
phase2_preprocessing/scripts/build_v9_mixed_dataset.py

Build the V9 mixed training dataset — fixes tool_call tag emission.

Key changes from V8:
  - Slice B increased from 12K to 19K (tool-calling density from 31% to ~45%)
  - Synthetic multi-tool-call examples (~2K) teach 2-3 calls per assistant turn
  - All assistant turns with </tool_call> are stripped of trailing text (stop-after-tag)
  - Tag completeness validation: drop records with unmatched <tool_call> tags
  - V8 builder is left untouched for reproducibility

Four slices + synthetic multi-call:
  A: Code generation     (~8K  from cmndcntrlcyber/code-trainer-offsec-dataset)
  B: Tool calling         (~19K from glaiveai/glaive-function-calling-v2, native format)
  B+: Multi-tool-call     (~2K  synthetic from Slice B pairs)
  C: Agent traces         (~10K from greghavens/fable-5-coding-and-debugging-traces)
  D: English instruction  (~8K  from teknium/OpenHermes-2.5)

Usage:
    uv run python -m src.phase2_preprocessing.scripts.build_v9_mixed_dataset \\
        --config src/config/config.yaml --no-push

    uv run python -m src.phase2_preprocessing.scripts.build_v9_mixed_dataset \\
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
    if not text:
        return True
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return (non_ascii / len(text)) < NON_ASCII_THRESHOLD


def messages_are_english(messages: list[dict]) -> bool:
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


QWEN_TOOLS_TEMPLATE = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_json_lines}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""


def parse_glaive_system(system_text: str) -> tuple[str, list[dict]]:
    system_text = system_text.replace("SYSTEM: ", "", 1).strip()

    tools = []
    depth = 0
    start = None
    for ci, ch in enumerate(system_text):
        if ch == "{":
            if depth == 0:
                start = ci
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(system_text[start : ci + 1])
                    if "name" in obj and "parameters" in obj:
                        tools.append({"type": "function", "function": obj})
                except json.JSONDecodeError:
                    pass
                start = None

    base_prompt = system_text
    if tools:
        first_brace = system_text.find("{")
        if first_brace > 0:
            base_prompt = system_text[:first_brace].strip().rstrip("-").strip()

    return base_prompt, tools


def build_qwen_system_with_tools(base_prompt: str, tools: list[dict]) -> str:
    tool_lines = "\n".join(json.dumps(t) for t in tools)
    tools_block = QWEN_TOOLS_TEMPLATE.format(tool_json_lines=tool_lines)
    if base_prompt:
        return f"{base_prompt}\n\n{tools_block}"
    return tools_block


def parse_glaive_chat(chat_text: str) -> list[dict]:
    messages = []
    parts = re.split(r"\n(?=USER:|FUNCTION RESPONSE:|ASSISTANT:)", chat_text.strip())

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
            fc_match = re.search(r"<functioncall>\s*(\{.*\})", content, re.DOTALL)
            if fc_match:
                raw = fc_match.group(1)
                raw = re.sub(r"'(\{.*?\})'", lambda m: '"' + m.group(1).replace('"', '\\"') + '"', raw)
                try:
                    call = json.loads(raw)
                except json.JSONDecodeError:
                    try:
                        call = json.loads(raw.replace("'", '"'))
                    except json.JSONDecodeError:
                        messages.append({"role": "assistant", "content": content})
                        continue
                name = call.get("name", "")
                args_raw = call.get("arguments", {})
                if isinstance(args_raw, str):
                    try:
                        args_raw = json.loads(args_raw)
                    except json.JSONDecodeError:
                        args_raw = {}
                tool_call = {"name": name, "arguments": args_raw}
                messages.append({
                    "role": "assistant",
                    "content": f"<tool_call>\n{json.dumps(tool_call)}\n</tool_call>",
                })
            else:
                messages.append({"role": "assistant", "content": content})

        elif part.startswith("FUNCTION RESPONSE:"):
            content = part[len("FUNCTION RESPONSE:") :].strip()
            messages.append({
                "role": "user",
                "content": f"<tool_response>\n{content}\n</tool_response>",
            })

    return messages


def load_slice_b(
    dataset_id: str = "glaiveai/glaive-function-calling-v2",
    max_rows: int = 19000,
    seed: int = 42,
) -> list[dict]:
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

        if not tools:
            skipped_parse += 1
            continue

        messages = parse_glaive_chat(chat)
        if not messages:
            skipped_parse += 1
            continue

        system_content = build_qwen_system_with_tools(base_prompt, tools)
        if not any(m["role"] == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": system_content})

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


# ── Slice B+: Synthetic multi-tool-call examples ──────────────────────────


def _extract_single_tool_call(content: str) -> dict | None:
    """Extract JSON from a single <tool_call> block, or None."""
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def _extract_tool_response(content: str) -> str | None:
    """Extract content from a <tool_response> block, or None."""
    match = re.search(r"<tool_response>\s*(.*?)\s*</tool_response>", content, re.DOTALL)
    return match.group(1) if match else None


def _merge_tool_definitions(tools_a: list[dict], tools_b: list[dict]) -> list[dict]:
    """Merge two tool definition lists, deduplicating by function name."""
    seen = set()
    merged = []
    for t in tools_a + tools_b:
        name = t.get("function", {}).get("name", "")
        if name and name not in seen:
            seen.add(name)
            merged.append(t)
    return merged


def _extract_tools_from_system(system_content: str) -> list[dict]:
    """Extract tool definitions from a Qwen-format system prompt."""
    tools = []
    tools_match = re.search(r"<tools>\s*(.*?)\s*</tools>", system_content, re.DOTALL)
    if not tools_match:
        return tools
    for line in tools_match.group(1).strip().split("\n"):
        line = line.strip()
        if line:
            try:
                tools.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return tools


def synthesize_multi_tool_calls(
    records: list[dict],
    target_count: int = 2000,
    seed: int = 42,
) -> list[dict]:
    """Synthesize multi-tool-call examples from single-call glaive records.

    Strategy 1: Find conversations with consecutive (assistant+tool_call, user+response)
    pairs and merge them into one multi-call assistant turn.

    Strategy 2 (fallback): Randomly pair two single-call records, merge their tool
    definitions and tool calls into a synthetic multi-call conversation.
    """
    rng = random.Random(seed)
    tool_records = [r for r in records if r.get("has_tools") and r["slice"] == "tool_calling"]
    synthetic = []

    # Strategy 1: merge consecutive tool-call turns within existing conversations
    for record in tool_records:
        if len(synthetic) >= target_count:
            break
        msgs = record["messages"]
        i = 0
        while i < len(msgs) - 3:
            a1 = msgs[i]
            r1 = msgs[i + 1] if i + 1 < len(msgs) else None
            a2 = msgs[i + 2] if i + 2 < len(msgs) else None
            r2 = msgs[i + 3] if i + 3 < len(msgs) else None

            if (a1["role"] == "assistant" and r1 and r1["role"] == "user"
                    and a2 and a2["role"] == "assistant" and r2 and r2["role"] == "user"):
                tc1 = _extract_single_tool_call(a1["content"])
                tr1 = _extract_tool_response(r1["content"])
                tc2 = _extract_single_tool_call(a2["content"])
                tr2 = _extract_tool_response(r2["content"])

                if tc1 and tc2 and tr1 is not None and tr2 is not None:
                    merged_call = (
                        f"<tool_call>\n{json.dumps(tc1)}\n</tool_call>\n"
                        f"<tool_call>\n{json.dumps(tc2)}\n</tool_call>"
                    )
                    merged_response = (
                        f"<tool_response>\n{tr1}\n</tool_response>\n"
                        f"<tool_response>\n{tr2}\n</tool_response>"
                    )
                    new_msgs = list(msgs[:i])
                    new_msgs.append({"role": "assistant", "content": merged_call})
                    new_msgs.append({"role": "user", "content": merged_response})
                    if i + 4 < len(msgs):
                        new_msgs.extend(msgs[i + 4:])

                    if validate_messages(new_msgs):
                        synthetic.append({
                            "messages": new_msgs,
                            "slice": "tool_calling_multi",
                            "source": "synthetic",
                            "category": "function_calling_multi",
                            "has_tools": True,
                            "n_turns": len(new_msgs),
                        })
                        break
            i += 1

    logger.info("  Strategy 1 (consecutive merge): %d multi-call examples", len(synthetic))

    # Strategy 2: random pairing of single-call records
    if len(synthetic) < target_count:
        single_call_records = [
            r for r in tool_records
            if any(
                m["role"] == "assistant" and _extract_single_tool_call(m["content"])
                for m in r["messages"]
            )
        ]
        rng.shuffle(single_call_records)

        pairs = list(zip(single_call_records[::2], single_call_records[1::2]))
        for rec_a, rec_b in pairs:
            if len(synthetic) >= target_count:
                break

            sys_a = next((m for m in rec_a["messages"] if m["role"] == "system"), None)
            sys_b = next((m for m in rec_b["messages"] if m["role"] == "system"), None)
            if not sys_a or not sys_b:
                continue

            tools_a = _extract_tools_from_system(sys_a["content"])
            tools_b = _extract_tools_from_system(sys_b["content"])
            merged_tools = _merge_tool_definitions(tools_a, tools_b)
            if not merged_tools:
                continue

            base_a = sys_a["content"].split("\n\n# Tools")[0] if "\n\n# Tools" in sys_a["content"] else ""
            merged_system = build_qwen_system_with_tools(base_a, merged_tools)

            tc_a = tc_b = None
            user_a = user_b = None
            resp_a = resp_b = None

            for m in rec_a["messages"]:
                if m["role"] == "user" and "<tool_response>" not in m["content"] and not user_a:
                    user_a = m["content"]
                elif m["role"] == "assistant" and not tc_a:
                    tc_a = _extract_single_tool_call(m["content"])
                elif m["role"] == "user" and "<tool_response>" in m["content"] and not resp_a:
                    resp_a = _extract_tool_response(m["content"])

            for m in rec_b["messages"]:
                if m["role"] == "user" and "<tool_response>" not in m["content"] and not user_b:
                    user_b = m["content"]
                elif m["role"] == "assistant" and not tc_b:
                    tc_b = _extract_single_tool_call(m["content"])
                elif m["role"] == "user" and "<tool_response>" in m["content"] and not resp_b:
                    resp_b = _extract_tool_response(m["content"])

            if not all([tc_a, tc_b, user_a, resp_a is not None, resp_b is not None]):
                continue

            merged_call = (
                f"<tool_call>\n{json.dumps(tc_a)}\n</tool_call>\n"
                f"<tool_call>\n{json.dumps(tc_b)}\n</tool_call>"
            )
            merged_response = (
                f"<tool_response>\n{resp_a}\n</tool_response>\n"
                f"<tool_response>\n{resp_b}\n</tool_response>"
            )

            combined_query = f"{user_a}\n\nAlso, {user_b[0].lower()}{user_b[1:]}" if user_b else user_a

            new_msgs = [
                {"role": "system", "content": merged_system},
                {"role": "user", "content": combined_query},
                {"role": "assistant", "content": merged_call},
                {"role": "user", "content": merged_response},
            ]

            if validate_messages(new_msgs) and messages_are_english(new_msgs):
                synthetic.append({
                    "messages": new_msgs,
                    "slice": "tool_calling_multi",
                    "source": "synthetic",
                    "category": "function_calling_multi",
                    "has_tools": True,
                    "n_turns": len(new_msgs),
                })

    logger.info("  Total multi-call synthetic: %d (target: %d)", len(synthetic), target_count)
    return synthetic


# ── Slice C: Agent traces (Fable 5, English-only) ─────────────────────────


def load_slice_c(
    dataset_id: str = "greghavens/fable-5-coding-and-debugging-traces",
    max_rows: int = 10000,
    seed: int = 42,
) -> list[dict]:
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


# ── V9 post-processing: stop-after-tag + tag validation ──────────────────


def strip_after_tool_call_close(content: str) -> str:
    """Strip everything after the last </tool_call> in an assistant turn."""
    last_close = content.rfind("</tool_call>")
    if last_close == -1:
        return content
    return content[:last_close + len("</tool_call>")]


def enforce_stop_after_tag(records: list[dict]) -> list[dict]:
    """Strip trailing text after </tool_call> in all assistant turns."""
    stripped = 0
    for record in records:
        for msg in record["messages"]:
            if msg["role"] == "assistant" and "</tool_call>" in msg["content"]:
                cleaned = strip_after_tool_call_close(msg["content"])
                if cleaned != msg["content"]:
                    stripped += 1
                msg["content"] = cleaned
    logger.info("  Stop-after-tag: stripped trailing text from %d assistant turns", stripped)
    return records


def validate_tag_completeness(records: list[dict]) -> list[dict]:
    """Drop records where an assistant message has <tool_call> without </tool_call>."""
    valid = []
    dropped = 0
    for record in records:
        ok = True
        for msg in record["messages"]:
            if msg["role"] == "assistant":
                opens = msg["content"].count("<tool_call>")
                closes = msg["content"].count("</tool_call>")
                if opens != closes:
                    ok = False
                    break
        if ok:
            valid.append(record)
        else:
            dropped += 1
    logger.info("  Tag validation: dropped %d records with unmatched tags", dropped)
    return valid


# ── Build pipeline ────────────────────────────────────────────────────────


def build_dataset_dict(
    records: list[dict],
    seed: int = 42,
    val_ratio: float = 0.1,
) -> DatasetDict:
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
        description="Build V9 mixed training dataset (tool_call tag emission fix)"
    )
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--output-dir", default="data/v9_mixed")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--slice-a-size", type=int, default=8000)
    parser.add_argument("--slice-b-size", type=int, default=19000)
    parser.add_argument("--slice-b-multi", type=int, default=2000)
    parser.add_argument("--slice-c-size", type=int, default=10000)
    parser.add_argument("--slice-d-size", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hub-repo", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    v9_cfg = config.get("v9_mixed", {})
    hub_repo = args.hub_repo or v9_cfg.get("dataset_name", "cmndcntrlcyber/code-trainer-v9-mixed")
    val_ratio = float(v9_cfg.get("val_split", 0.1))

    logger.info("=" * 60)
    logger.info("V9 Mixed Dataset Builder")
    logger.info("  Slice A: code-gen (%d)", args.slice_a_size)
    logger.info("  Slice B: tool-calling (%d)", args.slice_b_size)
    logger.info("  Slice B+: multi-call synthetic (%d)", args.slice_b_multi)
    logger.info("  Slice C: agent traces (%d)", args.slice_c_size)
    logger.info("  Slice D: instruction (%d)", args.slice_d_size)
    logger.info("  Hub repo: %s", hub_repo)
    logger.info("=" * 60)

    records_a = load_slice_a(
        v9_cfg.get("slice_a", {}).get("source", "cmndcntrlcyber/code-trainer-offsec-dataset"),
        args.slice_a_size, args.seed,
    )
    records_b = load_slice_b(
        v9_cfg.get("slice_b", {}).get("source", "glaiveai/glaive-function-calling-v2"),
        args.slice_b_size, args.seed,
    )

    multi_target = args.slice_b_multi or int(v9_cfg.get("slice_b_multi", {}).get("synthetic_count", 2000))
    records_b_multi = synthesize_multi_tool_calls(records_b, multi_target, args.seed)

    records_c = load_slice_c(
        v9_cfg.get("slice_c", {}).get("source", "greghavens/fable-5-coding-and-debugging-traces"),
        args.slice_c_size, args.seed,
    )
    records_d = load_slice_d(
        v9_cfg.get("slice_d", {}).get("source", "teknium/OpenHermes-2.5"),
        args.slice_d_size, args.seed,
    )

    all_records = records_a + records_b + records_b_multi + records_c + records_d
    logger.info("Total before post-processing: %d (A=%d B=%d B+=%d C=%d D=%d)",
                len(all_records), len(records_a), len(records_b),
                len(records_b_multi), len(records_c), len(records_d))

    # V9 post-processing
    all_records = enforce_stop_after_tag(all_records)
    all_records = validate_tag_completeness(all_records)
    logger.info("Total after post-processing: %d", len(all_records))

    dataset_dict = build_dataset_dict(all_records, args.seed, val_ratio)

    stats = compute_statistics(dataset_dict)
    for split_name, s in stats.items():
        logger.info("  %s: %d rows | slices: %s | turns: %s",
                     split_name, s["total"], s["by_slice"], s["n_turns"])

    project_root = Path(__file__).resolve().parents[3]
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
        logger.info("Cleared existing output dir: %s", output_dir)
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
                             commit_message="V9 mixed dataset: tool_call tag fix + multi-call + stop-after-tag")

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(stats_path),
        path_in_repo="statistics.json",
        repo_id=hub_repo,
        repo_type="dataset",
        commit_message="Upload V9 statistics",
    )
    logger.info("Pushed: https://huggingface.co/datasets/%s", hub_repo)


if __name__ == "__main__":
    main()
