"""
phase2_preprocessing/converters/tool_format_converter.py

Converts messages from various source formats into unified ChatML-compatible
messages for Qwen2.5 fine-tuning with Hermes tool-calling format.

Source formats handled:
  - Hermes (NousResearch): conversations with from/value keys, tools in system prompt
  - OpenAI (Fable 5): messages with role/content/tool_calls/tool_call_id keys
  - Code-Trainer V6: messages with role/content keys (pass-through)
"""
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

HERMES_ROLE_MAP = {
    "human": "user",
    "gpt": "assistant",
    "system": "system",
    "tool": "tool",
}


def openai_tool_calls_to_hermes_xml(tool_calls: list[dict]) -> str:
    """
    Convert OpenAI-format tool_calls to Hermes <tool_call> XML tags.

    Input (OpenAI format):
        [{"id": "...", "type": "function",
          "function": {"name": "Read", "arguments": '{"file_path": "x.py"}'}}]

    Output (Hermes XML):
        <tool_call>
        {"name": "Read", "arguments": {"file_path": "x.py"}}
        </tool_call>
    """
    blocks = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        raw_args = func.get("arguments", "{}")

        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Malformed tool_call arguments for %s: %s", name, raw_args)
                args = raw_args
        else:
            args = raw_args

        call_obj = {"name": name, "arguments": args}
        blocks.append(f"<tool_call>\n{json.dumps(call_obj)}\n</tool_call>")

    return "\n".join(blocks)


def convert_hermes_conversations_to_messages(
    conversations: list[dict[str, str]],
) -> list[dict[str, str]]:
    """
    Convert Hermes from/value format to standard role/content format.

    Hermes uses {"from": "human", "value": "..."} etc.
    Converts to {"role": "user", "content": "..."} etc.
    """
    messages = []
    for turn in conversations:
        raw_role = turn.get("from", "")
        role = HERMES_ROLE_MAP.get(raw_role, raw_role)
        content = turn.get("value", "")
        messages.append({"role": role, "content": content})
    return messages


def convert_fable5_messages_to_hermes(
    messages: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """
    Convert Fable 5 OpenAI-format messages to Hermes ChatML format.

    Transformations:
    1. Assistant messages with tool_calls: content + <tool_call> XML appended
    2. Tool result messages (role="tool"): content wrapped in <tool_response> tags
    3. reasoning_content: discarded (not part of Qwen2.5's format)
    4. All messages flattened to {"role": str, "content": str}
    """
    converted = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls") or []

        if role == "assistant" and tool_calls:
            xml = openai_tool_calls_to_hermes_xml(tool_calls)
            if content:
                content = f"{content}\n{xml}"
            else:
                content = xml

        elif role == "tool":
            content = f"<tool_response>\n{content}\n</tool_response>"

        converted.append({"role": role, "content": content})

    return converted


def detect_tools_in_messages(messages: list[dict[str, str]]) -> bool:
    """Check whether any message contains <tool_call> or <tools> tags."""
    for msg in messages:
        content = msg.get("content", "")
        if "<tool_call>" in content or "<tools>" in content:
            return True
    return False


def validate_messages(messages: list[dict[str, str]]) -> bool:
    """
    Validate that a messages list is well-formed for SFT.

    Requirements:
    - At least 2 messages
    - All messages have 'role' and 'content' keys
    - content is a string (not None)
    - Has at least one assistant message
    """
    if len(messages) < 2:
        return False

    has_assistant = False
    for msg in messages:
        if "role" not in msg or "content" not in msg:
            return False
        if not isinstance(msg["content"], str):
            return False
        if msg["role"] == "assistant":
            has_assistant = True

    return has_assistant
