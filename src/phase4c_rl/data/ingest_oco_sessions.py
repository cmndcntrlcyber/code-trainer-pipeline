"""
phase4c_rl/data/ingest_oco_sessions.py

Ingest offensive security session histories and convert them into
structured HuggingFace dataset format with <tool_call> tags mapped to
NEXUS_TOOLS_V10 schema names.

Supports four session sources (organized as subdirectories):
    data/cot_rl_sessions/htb/        — HackTheBox completed room histories
    data/cot_rl_sessions/thm/        — TryHackMe completed room histories
    data/cot_rl_sessions/claude/     — Claude session histories (all hosts)
    data/cot_rl_sessions/bugbounty/  — Bug bounty program sessions

All subdirectories are scanned recursively. Each session file is tagged
with its source (htb, thm, claude, bugbounty) based on its parent directory.

JSON format (expected):
    [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "...", "tool_calls": [...]},
        {"role": "tool", "name": "...", "content": "..."},
        ...
    ]

Text format (expected):
    Lines prefixed with "USER:", "ASSISTANT:", or "TOOL:" markers,
    separated by blank lines between turns.

Usage:
    python -m src.phase4c_rl.data.ingest_oco_sessions \
        --input-dir data/cot_rl_sessions \
        --output-dir data/oco_converted \
        --format json
"""
import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from src.phase4_qwen_finetuning.hf_skills.nexus_tools import NEXUS_TOOLS_V10

# Build lookup maps for tool name normalization.
VALID_TOOL_NAMES: set[str] = {
    t["function"]["name"] for t in NEXUS_TOOLS_V10
}

# Common aliases from various chat/tool-use formats -> NEXUS_TOOLS_V10 names.
TOOL_NAME_ALIASES: dict[str, str] = {
    # File operations
    "read_file": "Read",
    "read": "Read",
    "cat": "Read",
    "write_file": "Write",
    "write": "Write",
    "create_file": "Write",
    "edit_file": "Edit",
    "edit": "Edit",
    "replace": "Edit",
    "patch": "Edit",
    "list_dir": "LS",
    "ls": "LS",
    "list_directory": "LS",
    "listdir": "LS",
    # Shell
    "bash": "Bash",
    "shell": "Bash",
    "run_command": "Bash",
    "execute": "Bash",
    "exec": "Bash",
    "terminal": "Bash",
    "run": "Bash",
    "command": "Bash",
    # Search
    "grep": "Grep",
    "search": "Grep",
    "search_files": "Grep",
    "ripgrep": "Grep",
    "rg": "Grep",
    "glob": "Glob",
    "find_files": "Glob",
    "find": "Glob",
    # Web
    "web_fetch": "WebFetch",
    "webfetch": "WebFetch",
    "fetch": "WebFetch",
    "curl": "WebFetch",
    "http_get": "WebFetch",
    # Session
    "todo_write": "TodoWrite",
    "todowrite": "TodoWrite",
    "todo": "TodoWrite",
    "add_todo": "TodoWrite",
    "skill": "Skill",
    "invoke_skill": "Skill",
    "task": "Task",
    "delegate": "Task",
    "subagent": "Task",
    # Scope
    "scope_check": "ScopeCheck",
    "scopecheck": "ScopeCheck",
    "check_scope": "ScopeCheck",
}


def normalize_tool_name(name: str) -> str | None:
    """Map a tool name to its NEXUS_TOOLS_V10 canonical form, or None."""
    if name in VALID_TOOL_NAMES:
        return name
    lower = name.lower().strip()
    return TOOL_NAME_ALIASES.get(lower)


def _parse_json_session(path: Path) -> list[dict] | None:
    """Parse a JSON session file into a list of message dicts."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        logger.warning("Failed to read %s: %s", path, e)
        return None

    # Try single JSON document first.
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ("messages", "conversation", "turns", "chat", "history"):
                if key in data and isinstance(data[key], list):
                    return data[key]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Try JSONL (one JSON object per line) — Claude Code session format.
    return _parse_claude_jsonl(path, text)


def _content_blocks_to_text(content) -> str:
    """Convert Anthropic API content blocks to plain text."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content else ""
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            btype = block.get("type", "")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "tool_result":
                result_content = block.get("content", "")
                if isinstance(result_content, list):
                    for sub in result_content:
                        if isinstance(sub, dict) and sub.get("type") == "text":
                            parts.append(sub.get("text", ""))
                elif isinstance(result_content, str):
                    parts.append(result_content)
    return "\n".join(parts)


def _extract_tool_calls(content) -> list[dict]:
    """Extract tool_use blocks from Anthropic API content."""
    if not isinstance(content, list):
        return []
    calls = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            calls.append({
                "name": block.get("name", ""),
                "arguments": block.get("input", {}),
            })
    return calls


def _parse_claude_jsonl(path: Path, text: str) -> list[dict] | None:
    """Parse Claude Code JSONL session into a list of message dicts."""
    messages = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        etype = event.get("type", "")
        if etype not in ("user", "assistant"):
            continue

        msg = event.get("message", {})
        role = msg.get("role", etype)
        content_raw = msg.get("content", "")

        text_content = _content_blocks_to_text(content_raw)
        tool_calls = _extract_tool_calls(content_raw)

        entry = {"role": role, "content": text_content}
        if tool_calls:
            entry["tool_calls"] = tool_calls

        if text_content.strip() or tool_calls:
            messages.append(entry)

    return messages if messages else None


def _parse_text_session(path: Path) -> list[dict] | None:
    """Parse a plain-text session into a list of message dicts.

    Expected format:
        USER: <message>
        ASSISTANT: <message>
        TOOL: <tool_name>: <result>
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("Cannot read %s as UTF-8", path)
        return None

    messages = []
    role_pattern = re.compile(
        r"^(USER|ASSISTANT|TOOL|SYSTEM):\s*(.*)", re.IGNORECASE
    )

    current_role = None
    current_content = []

    for line in text.splitlines():
        match = role_pattern.match(line)
        if match:
            # Save previous turn.
            if current_role is not None:
                messages.append({
                    "role": current_role,
                    "content": "\n".join(current_content).strip(),
                })
            role_label = match.group(1).lower()
            if role_label == "tool":
                current_role = "tool"
            elif role_label == "user":
                current_role = "user"
            elif role_label == "system":
                current_role = "system"
            else:
                current_role = "assistant"
            current_content = [match.group(2)]
        elif current_role is not None:
            current_content.append(line)

    # Save last turn.
    if current_role is not None:
        messages.append({
            "role": current_role,
            "content": "\n".join(current_content).strip(),
        })

    return messages if messages else None


def _convert_tool_calls_in_message(message: dict) -> dict:
    """Convert tool_calls in a message dict to <tool_call> tag format."""
    role = message.get("role", "")
    content = message.get("content", "")

    # If the message already has <tool_call> tags, normalize tool names.
    if "<tool_call>" in content:
        def _normalize_tag(m):
            try:
                parsed = json.loads(m.group(1))
                if "name" in parsed:
                    canonical = normalize_tool_name(parsed["name"])
                    if canonical:
                        parsed["name"] = canonical
                    return f'<tool_call>{json.dumps(parsed)}</tool_call>'
            except json.JSONDecodeError:
                pass
            return m.group(0)

        content = re.sub(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
            _normalize_tag,
            content,
            flags=re.DOTALL,
        )
        return {"role": role, "content": content}

    # Convert explicit tool_calls list to <tool_call> tags.
    tool_calls = message.get("tool_calls", [])
    if tool_calls and role == "assistant":
        tag_parts = []
        for tc in tool_calls:
            # Handle various tool_call formats.
            if isinstance(tc, dict):
                name = tc.get("name") or tc.get("function", {}).get("name", "")
                arguments = (
                    tc.get("arguments")
                    or tc.get("function", {}).get("arguments", {})
                )
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw": arguments}

                canonical = normalize_tool_name(name)
                if canonical:
                    tag_parts.append(
                        f'<tool_call>{json.dumps({"name": canonical, "arguments": arguments})}</tool_call>'
                    )
                else:
                    logger.debug("Skipping unmapped tool: %s", name)

        if tag_parts:
            # Append tool call tags after any existing content.
            prefix = content.strip() + "\n\n" if content.strip() else ""
            content = prefix + "\n".join(tag_parts)

    return {"role": role, "content": content}


def convert_session(messages: list[dict]) -> dict | None:
    """Convert a raw session into HF dataset format with <tool_call> tags.

    Returns a dict with {"messages": [...]} or None if invalid.
    """
    converted = []
    has_tool_call = False

    for msg in messages:
        role = msg.get("role", "").lower()
        if role not in ("system", "user", "assistant", "tool"):
            continue

        new_msg = _convert_tool_calls_in_message(msg)

        # Skip empty messages.
        if not new_msg.get("content", "").strip():
            continue

        if "<tool_call>" in new_msg.get("content", ""):
            has_tool_call = True

        converted.append(new_msg)

    # Require at least one user and one assistant message.
    roles = {m["role"] for m in converted}
    if "user" not in roles or "assistant" not in roles:
        return None

    # Require at least one tool call (this is tool-use training data).
    if not has_tool_call:
        return None

    return {"messages": converted}


def main():
    parser = argparse.ArgumentParser(
        description="Ingest OCO (offensive cyber operations) sessions into HF dataset format"
    )
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing session files")
    parser.add_argument("--output-dir", default="data/oco_converted",
                        help="Output directory for HF dataset")
    parser.add_argument("--format", choices=["json", "text"], default="json",
                        help="Input file format")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    # Discover session files.
    if args.format == "json":
        extensions = {".json", ".jsonl"}
    else:
        extensions = {".txt", ".log", ".md"}

    files = sorted(
        f for f in input_dir.rglob("*") if f.suffix.lower() in extensions
    )
    logger.info("Found %d session files in %s", len(files), input_dir)

    if not files:
        raise SystemExit(f"No {args.format} files found in {input_dir}")

    # Parse and convert.
    converted = []
    skipped = 0

    for path in files:
        if args.format == "json":
            messages = _parse_json_session(path)
        else:
            messages = _parse_text_session(path)

        if messages is None:
            skipped += 1
            continue

        result = convert_session(messages)
        if result is not None:
            result["source_file"] = str(path.relative_to(input_dir))
            converted.append(result)
        else:
            skipped += 1

    logger.info(
        "Converted %d sessions, skipped %d", len(converted), skipped
    )

    if not converted:
        raise SystemExit("No valid sessions found after conversion")

    # Split into train/validation.
    import random
    random.seed(args.seed)
    random.shuffle(converted)

    val_size = max(1, int(len(converted) * args.val_split))
    val_data = converted[:val_size]
    train_data = converted[val_size:]

    # Save as JSONL (datasets-compatible).
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "validation.jsonl"

    with open(train_path, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    with open(val_path, "w") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")

    stats = {
        "total_files": len(files),
        "converted": len(converted),
        "skipped": skipped,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "tools_mapped": sorted(VALID_TOOL_NAMES),
    }
    (output_dir / "ingestion_stats.json").write_text(
        json.dumps(stats, indent=2)
    )

    logger.info("Output: %s (%d train, %d val)", output_dir, len(train_data), len(val_data))


if __name__ == "__main__":
    main()
