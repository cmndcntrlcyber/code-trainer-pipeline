"""
phase4c_rl/rewards/tool_call_reward.py

GRPO reward function that scores model responses on 5 criteria for
tool-call quality. Used by GRPOTrainer during Phase 4c RL training.

Each response is scored 0.0–1.0 as a weighted sum:
  - has_valid_tool_call_tags   (0.30): proper <tool_call>{"name":...,"arguments":...}</tool_call>
  - tool_name_in_schema        (0.20): tool name exists in NEXUS_TOOLS_V10
  - has_reasoning_prefix       (0.20): reasoning text before the first <tool_call>
  - no_hallucinated_tools      (0.15): no tool names outside the schema
  - ends_cleanly_after_tag     (0.15): no trailing text/garbage after </tool_call>
"""
import json
import re
from typing import Optional

from src.phase4_qwen_finetuning.hf_skills.nexus_tools import NEXUS_TOOLS_V10

# Pre-compute the set of valid tool names from the V10 schema.
VALID_TOOL_NAMES: set[str] = {
    t["function"]["name"] for t in NEXUS_TOOLS_V10
}

# Regex patterns for tool-call extraction.
TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
)
# Matches any {"name": "Something", ...} blocks (even malformed ones).
ANY_TOOL_NAME_RE = re.compile(
    r'"name"\s*:\s*"([^"]+)"', re.DOTALL
)

# Reward component weights.
WEIGHTS = {
    "has_valid_tool_call_tags": 0.30,
    "tool_name_in_schema": 0.20,
    "has_reasoning_prefix": 0.20,
    "no_hallucinated_tools": 0.15,
    "ends_cleanly_after_tag": 0.15,
}


def _parse_tool_calls(response: str) -> list[dict]:
    """Extract well-formed tool calls from <tool_call>...</tool_call> tags."""
    calls = []
    for match in TOOL_CALL_RE.finditer(response):
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, dict) and "name" in parsed:
                calls.append(parsed)
        except json.JSONDecodeError:
            pass
    return calls


def _has_valid_tool_call_tags(response: str) -> float:
    """1.0 if at least one <tool_call>{"name":"...","arguments":{...}}</tool_call> found."""
    calls = _parse_tool_calls(response)
    if not calls:
        return 0.0
    # Check that at least one call has both name and arguments.
    for call in calls:
        if (
            isinstance(call.get("name"), str)
            and isinstance(call.get("arguments"), dict)
        ):
            return 1.0
    return 0.0


def _tool_name_in_schema(response: str) -> float:
    """1.0 if the first valid tool call uses a name from NEXUS_TOOLS_V10."""
    calls = _parse_tool_calls(response)
    if not calls:
        return 0.0
    first_name = calls[0].get("name", "")
    return 1.0 if first_name in VALID_TOOL_NAMES else 0.0


def _has_reasoning_prefix(response: str) -> float:
    """1.0 if there is non-whitespace text before the first <tool_call> tag."""
    idx = response.find("<tool_call>")
    if idx < 0:
        # No tool call at all -- no reasoning needed but also no credit.
        return 0.0
    prefix = response[:idx].strip()
    # Require at least 10 chars of reasoning (not just a newline or token).
    return 1.0 if len(prefix) >= 10 else 0.0


def _no_hallucinated_tools(response: str) -> float:
    """1.0 if every tool name referenced in the response is in the schema."""
    all_names = set(ANY_TOOL_NAME_RE.findall(response))
    if not all_names:
        # No tool names at all -- vacuously true.
        return 1.0
    hallucinated = all_names - VALID_TOOL_NAMES
    return 1.0 if not hallucinated else 0.0


def _ends_cleanly_after_tag(response: str) -> float:
    """1.0 if no meaningful trailing text exists after the last </tool_call>."""
    last_close = response.rfind("</tool_call>")
    if last_close < 0:
        # No closing tag at all -- fail.
        return 0.0
    trailing = response[last_close + len("</tool_call>"):].strip()
    # Allow up to 5 chars of trailing whitespace/punctuation (e.g. newline,
    # EOS token artifact) but penalize actual text.
    return 1.0 if len(trailing) <= 5 else 0.0


def tool_call_reward(
    completions: list[str],
    schema: Optional[list[dict]] = None,
    **kwargs,
) -> list[float]:
    """Score a batch of model completions for tool-call quality.

    Args:
        completions: List of generated response strings.
        schema: Optional tool schema list. If provided, overrides
                NEXUS_TOOLS_V10 for valid tool names. Each entry must have
                the shape {"function": {"name": "..."}}.
        **kwargs: Ignored (allows GRPOTrainer to pass extra context).

    Returns:
        List of float rewards in [0.0, 1.0], one per completion.
    """
    # Allow runtime schema override.
    global VALID_TOOL_NAMES
    if schema is not None:
        valid_names = {t["function"]["name"] for t in schema}
    else:
        valid_names = VALID_TOOL_NAMES

    # Temporarily swap for the per-call checkers that read the global.
    original_valid = VALID_TOOL_NAMES
    if schema is not None:
        VALID_TOOL_NAMES = valid_names

    rewards = []
    for response in completions:
        score = (
            WEIGHTS["has_valid_tool_call_tags"] * _has_valid_tool_call_tags(response)
            + WEIGHTS["tool_name_in_schema"] * _tool_name_in_schema(response)
            + WEIGHTS["has_reasoning_prefix"] * _has_reasoning_prefix(response)
            + WEIGHTS["no_hallucinated_tools"] * _no_hallucinated_tools(response)
            + WEIGHTS["ends_cleanly_after_tag"] * _ends_cleanly_after_tag(response)
        )
        rewards.append(score)

    # Restore global.
    VALID_TOOL_NAMES = original_valid

    return rewards


def tool_call_reward_detailed(
    response: str,
    schema: Optional[list[dict]] = None,
) -> dict:
    """Score a single response and return per-component breakdown.

    Useful for diagnostics and negative collection.
    """
    global VALID_TOOL_NAMES
    original_valid = VALID_TOOL_NAMES
    if schema is not None:
        VALID_TOOL_NAMES = {t["function"]["name"] for t in schema}

    components = {
        "has_valid_tool_call_tags": _has_valid_tool_call_tags(response),
        "tool_name_in_schema": _tool_name_in_schema(response),
        "has_reasoning_prefix": _has_reasoning_prefix(response),
        "no_hallucinated_tools": _no_hallucinated_tools(response),
        "ends_cleanly_after_tag": _ends_cleanly_after_tag(response),
    }
    total = sum(WEIGHTS[k] * v for k, v in components.items())

    VALID_TOOL_NAMES = original_valid

    return {
        "total_reward": total,
        "components": components,
        "weights": dict(WEIGHTS),
    }
