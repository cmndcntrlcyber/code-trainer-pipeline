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

from src.config.nexus_identity import OFFSEC_TERMS, PERSONA_BREAK_PHRASES
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
    "has_valid_tool_call_tags": 0.25,
    "tool_name_in_schema": 0.20,
    "has_reasoning_prefix": 0.15,
    "no_hallucinated_tools": 0.15,
    "ends_cleanly_after_tag": 0.15,
    "persona_aligned_reasoning": 0.10,
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


def _tool_selection_quality(response: str, prompt: str | None = None) -> float:
    """TTCA P0: Score tool-task alignment (0.0–1.0).

    Three sub-signals combined: scope-before-action (0.4),
    tool-task alignment (0.4), no-redundancy (0.2).

    Only called when a role-specific weight table includes this key.
    """
    calls = _parse_tool_calls(response)
    if not calls:
        return 0.0

    # Sub-signal 1: scope before action
    scope_score = _scope_before_action(response, prompt)

    # Sub-signal 2: tool-task alignment
    alignment_score = _tool_task_alignment(calls, prompt)

    # Sub-signal 3: no redundant tool calls
    redundancy_score = _no_redundancy(calls)

    return 0.4 * scope_score + 0.4 * alignment_score + 0.2 * redundancy_score


def _scope_before_action(response: str, prompt: str | None) -> float:
    """1.0 if ScopeCheck precedes action tools when new targets are engaged."""
    calls = _parse_tool_calls(response)
    if not calls:
        return 1.0

    action_tools = {"Bash", "Skill", "WebFetch"}
    scope_check_seen = False
    for call in calls:
        name = call.get("name", "")
        if name == "ScopeCheck":
            scope_check_seen = True
        elif name in action_tools and not scope_check_seen:
            if prompt and any(
                kw in prompt.lower()
                for kw in ("new target", "new ip", "new domain", "new host")
            ):
                return 0.0
    return 1.0


_TOOL_TASK_MAP = {
    "read": "Read", "check file": "Read", "show file": "Read", "view": "Read",
    "search": "Grep", "find": "Grep", "grep": "Grep", "look for": "Grep",
    "list": "LS", "directory": "LS", "ls": "LS",
    "edit": "Edit", "change": "Edit", "modify": "Edit", "replace": "Edit",
    "write": "Write", "create file": "Write", "save": "Write",
    "scan": "Bash", "run": "Bash", "execute": "Bash", "install": "Bash",
    "fetch": "WebFetch", "download": "WebFetch", "url": "WebFetch",
    "scope": "ScopeCheck", "authorized": "ScopeCheck",
}


def _tool_task_alignment(calls: list[dict], prompt: str | None) -> float:
    """1.0 if tool matches task keywords, 0.5 for Bash fallback, 0.0 for mismatch."""
    if not prompt or not calls:
        return 0.5

    lower = prompt.lower()
    expected = None
    for keyword, tool in _TOOL_TASK_MAP.items():
        if keyword in lower:
            expected = tool
            break

    if expected is None:
        return 0.5

    first_tool = calls[0].get("name", "")
    if first_tool == expected:
        return 1.0
    if first_tool == "Bash":
        return 0.5
    return 0.0


def _no_redundancy(calls: list[dict]) -> float:
    """1.0 if no duplicate tool+target pairs detected."""
    seen: set[str] = set()
    for call in calls:
        name = call.get("name", "")
        args = call.get("arguments", {})
        target = args.get("path") or args.get("command") or args.get("url") or ""
        key = f"{name}:{target}"
        if key in seen:
            return 0.0
        seen.add(key)
    return 1.0


def _reasoning_efficiency(response: str) -> float:
    """TTCA P2: Soft conciseness bonus (0.5–1.0).

    1.0 for <100 chars reasoning, linear decay to 0.5 at 400+ chars.
    Gated on having a valid tool call — no credit without correctness.

    Only called when a role-specific weight table includes this key.
    """
    if _has_valid_tool_call_tags(response) < 1.0:
        return 0.0

    idx = response.find("<tool_call>")
    if idx < 0:
        return 0.0

    reasoning_len = len(response[:idx].strip())
    if reasoning_len <= 100:
        return 1.0
    if reasoning_len >= 400:
        return 0.5
    return 1.0 - 0.5 * (reasoning_len - 100) / 300


def _persona_aligned_reasoning(
    response: str,
    domain_terms: frozenset[str] | None = None,
    break_phrases: list[str] | None = None,
) -> float:
    """Score persona alignment of the reasoning prefix.

    1.0 if domain terms present AND no persona-breaking phrases.
    0.5 if no persona-breaking phrases but no domain terms.
    0.0 if persona-breaking phrases detected.

    When *domain_terms* or *break_phrases* are None, falls back to
    the module-level OFFSEC_TERMS / PERSONA_BREAK_PHRASES constants.
    """
    terms = domain_terms if domain_terms is not None else OFFSEC_TERMS
    phrases = break_phrases if break_phrases is not None else PERSONA_BREAK_PHRASES

    lower = response.lower()

    for phrase in phrases:
        if phrase in lower:
            return 0.0

    idx = response.find("<tool_call>")
    prefix = response[:idx].lower() if idx >= 0 else lower
    words = set(re.findall(r"[a-z&]+", prefix))
    if words & terms:
        return 1.0
    return 0.5


def tool_call_reward(
    completions: list[str],
    schema: Optional[list[dict]] = None,
    persona_reward: bool = True,
    prompts: list[str] | None = None,
    role: str | None = None,
    weights: dict[str, float] | None = None,
    domain: str | None = None,
    **kwargs,
) -> list[float]:
    """Score a batch of model completions for tool-call quality.

    Args:
        completions: List of generated response strings.
        schema: Optional tool schema list. If provided, overrides
                NEXUS_TOOLS_V10 for valid tool names. Each entry must have
                the shape {"function": {"name": "..."}}.
        prompts: Optional list of user prompts (for TTCA P0 tool-task alignment).
        role: Optional swarm role name. When set, loads role-specific weights
              from role_reward_weights. When None, uses default WEIGHTS.
        weights: Explicit weight dict override (for testing/custom configs).
        domain: Optional domain name (e.g. "offsec"). Loads domain-specific
                terms and break phrases for persona scoring. When None, uses
                the module-level OFFSEC_TERMS / PERSONA_BREAK_PHRASES.
        **kwargs: Ignored (allows GRPOTrainer to pass extra context).

    Returns:
        List of float rewards in [0.0, 1.0], one per completion.
    """
    # Select weight table: explicit override > role-specific > default
    if weights is not None:
        active_weights = weights
    elif role is not None:
        from src.phase4c_rl.rewards.role_reward_weights import get_weights
        active_weights = get_weights(role)
    else:
        active_weights = WEIGHTS

    # Resolve domain-specific vocabulary for persona scoring.
    _domain_terms: frozenset[str] | None = None
    _break_phrases: list[str] | None = None
    if domain is not None:
        from src.config.domain_loader import get_domain
        d = get_domain(domain)
        _domain_terms = d.domain_terms
        _break_phrases = d.break_phrases

    # Allow runtime schema override.
    global VALID_TOOL_NAMES
    original_valid = VALID_TOOL_NAMES
    if schema is not None:
        VALID_TOOL_NAMES = {t["function"]["name"] for t in schema}

    # Map component names to scoring functions.
    _COMPONENT_FNS = {
        "has_valid_tool_call_tags": lambda r, p: _has_valid_tool_call_tags(r),
        "tool_name_in_schema": lambda r, p: _tool_name_in_schema(r),
        "has_reasoning_prefix": lambda r, p: _has_reasoning_prefix(r),
        "no_hallucinated_tools": lambda r, p: _no_hallucinated_tools(r),
        "ends_cleanly_after_tag": lambda r, p: _ends_cleanly_after_tag(r),
        "persona_aligned_reasoning": lambda r, p: _persona_aligned_reasoning(
            r, domain_terms=_domain_terms, break_phrases=_break_phrases,
        ),
        "tool_selection_quality": lambda r, p: _tool_selection_quality(r, p),
        "reasoning_efficiency": lambda r, p: _reasoning_efficiency(r),
    }

    rewards = []
    for i, response in enumerate(completions):
        prompt = prompts[i] if prompts and i < len(prompts) else None
        score = 0.0
        for component, weight in active_weights.items():
            if weight == 0.0:
                continue
            if not persona_reward and component == "persona_aligned_reasoning":
                continue
            fn = _COMPONENT_FNS.get(component)
            if fn:
                score += weight * fn(response, prompt)
        rewards.append(score)

    VALID_TOOL_NAMES = original_valid
    return rewards


def tool_call_reward_detailed(
    response: str,
    schema: Optional[list[dict]] = None,
    persona_reward: bool = True,
    prompt: str | None = None,
    role: str | None = None,
    weights: dict[str, float] | None = None,
    domain: str | None = None,
) -> dict:
    """Score a single response and return per-component breakdown.

    Useful for diagnostics and negative collection.
    """
    if weights is not None:
        active_weights = weights
    elif role is not None:
        from src.phase4c_rl.rewards.role_reward_weights import get_weights
        active_weights = get_weights(role)
    else:
        active_weights = WEIGHTS

    _domain_terms: frozenset[str] | None = None
    _break_phrases: list[str] | None = None
    if domain is not None:
        from src.config.domain_loader import get_domain
        d = get_domain(domain)
        _domain_terms = d.domain_terms
        _break_phrases = d.break_phrases

    global VALID_TOOL_NAMES
    original_valid = VALID_TOOL_NAMES
    if schema is not None:
        VALID_TOOL_NAMES = {t["function"]["name"] for t in schema}

    _COMPONENT_FNS = {
        "has_valid_tool_call_tags": lambda: _has_valid_tool_call_tags(response),
        "tool_name_in_schema": lambda: _tool_name_in_schema(response),
        "has_reasoning_prefix": lambda: _has_reasoning_prefix(response),
        "no_hallucinated_tools": lambda: _no_hallucinated_tools(response),
        "ends_cleanly_after_tag": lambda: _ends_cleanly_after_tag(response),
        "persona_aligned_reasoning": lambda: _persona_aligned_reasoning(
            response, domain_terms=_domain_terms, break_phrases=_break_phrases,
        ),
        "tool_selection_quality": lambda: _tool_selection_quality(response, prompt),
        "reasoning_efficiency": lambda: _reasoning_efficiency(response),
    }

    components = {}
    for component, weight in active_weights.items():
        if not persona_reward and component == "persona_aligned_reasoning":
            continue
        fn = _COMPONENT_FNS.get(component)
        if fn:
            components[component] = fn()

    total = sum(active_weights.get(k, 0) * v for k, v in components.items())

    VALID_TOOL_NAMES = original_valid

    return {
        "total_reward": total,
        "components": components,
        "weights": dict(active_weights),
    }
