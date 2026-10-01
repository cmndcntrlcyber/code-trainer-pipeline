"""
Per-role reward weight tables for the NEXUS swarm pipeline.

Only loaded when ``--role`` is specified. The default WEIGHTS in
tool_call_reward.py is never modified — this module provides
alternative weight dicts that include the TTCA P0+P2 components.
"""

DEFAULT_WEIGHTS = {
    "has_valid_tool_call_tags": 0.25,
    "tool_name_in_schema": 0.20,
    "has_reasoning_prefix": 0.15,
    "no_hallucinated_tools": 0.15,
    "ends_cleanly_after_tag": 0.15,
    "persona_aligned_reasoning": 0.10,
}

ORCHESTRATOR_WEIGHTS = {
    "has_valid_tool_call_tags": 0.15,
    "tool_name_in_schema": 0.10,
    "has_reasoning_prefix": 0.15,
    "no_hallucinated_tools": 0.10,
    "ends_cleanly_after_tag": 0.05,
    "persona_aligned_reasoning": 0.10,
    "tool_selection_quality": 0.25,
    "reasoning_efficiency": 0.10,
}

WORKER_WEIGHTS = {
    "has_valid_tool_call_tags": 0.25,
    "tool_name_in_schema": 0.20,
    "has_reasoning_prefix": 0.05,
    "no_hallucinated_tools": 0.15,
    "ends_cleanly_after_tag": 0.15,
    "persona_aligned_reasoning": 0.00,
    "tool_selection_quality": 0.10,
    "reasoning_efficiency": 0.10,
}

_ROLE_MAP = {
    "orchestrator": ORCHESTRATOR_WEIGHTS,
    "worker": WORKER_WEIGHTS,
}


def get_weights(role: str | None = None) -> dict[str, float]:
    """Return reward weights for the given role.

    None returns the existing default (no TTCA components).
    """
    if role is None:
        return DEFAULT_WEIGHTS
    if role not in _ROLE_MAP:
        raise ValueError(
            f"Unknown role {role!r}. Valid roles: {sorted(_ROLE_MAP)}"
        )
    return _ROLE_MAP[role]
