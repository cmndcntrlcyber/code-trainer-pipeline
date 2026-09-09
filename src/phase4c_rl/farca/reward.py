"""FARCA Module 4: Three-component reward design.

R_i = R^format + R^task + R^fact

Where:
  R^format = existing tool_call_reward remapped to [-1, +1]
  R^task   = +1/-1 for correct/incorrect tool (0 if unknown)
  R^fact   = mean reliability-weighted factual score across claims
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .claim_extractor import AtomicClaim
from .counterfactual_attribution import ReliabilityResult

# Import existing reward function
_tcr_module = None


def _get_tool_call_reward():
    global _tcr_module
    if _tcr_module is None:
        project_root = Path(__file__).resolve().parents[3]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from src.phase4c_rl.rewards.tool_call_reward import tool_call_reward
        _tcr_module = tool_call_reward
    return _tcr_module


@dataclass
class FARCAReward:
    format_reward: float
    task_reward: float
    fact_reward: float
    total_reward: float
    claim_scores: list[float] = field(default_factory=list)
    claim_weights: list[float] = field(default_factory=list)
    claim_weighted_scores: list[float] = field(default_factory=list)


def compute_farca_reward(
    completion: str,
    claims: list[AtomicClaim],
    reliability_results: list[ReliabilityResult],
    expected_tool: str | None = None,
    format_weight: float = 1.0,
    answer_weight: float = 1.0,
    fact_weight: float = 1.0,
) -> FARCAReward:
    # R^format: existing tool_call_reward remapped from [0,1] to [-1,+1]
    try:
        tcr = _get_tool_call_reward()
        raw_format = tcr([completion])[0]
    except Exception:
        raw_format = 0.5
    r_format = (2.0 * raw_format - 1.0) * format_weight

    # R^task: check first tool call against expected
    r_task = 0.0
    if expected_tool is not None:
        first_tool = _extract_first_tool_name(completion)
        if first_tool is not None:
            r_task = (1.0 if first_tool.lower() == expected_tool.lower() else -1.0)
        else:
            r_task = -1.0
        r_task *= answer_weight

    # R^fact: mean reliability-weighted factual score
    claim_scores = [rr.original_score for rr in reliability_results]
    claim_weights = [rr.reliability_weight for rr in reliability_results]
    claim_weighted = [rr.weighted_score for rr in reliability_results]

    if claim_weighted:
        r_fact = sum(claim_weighted) / len(claim_weighted) * fact_weight
    else:
        r_fact = 0.0

    total = r_format + r_task + r_fact

    return FARCAReward(
        format_reward=r_format,
        task_reward=r_task,
        fact_reward=r_fact,
        total_reward=total,
        claim_scores=claim_scores,
        claim_weights=claim_weights,
        claim_weighted_scores=claim_weighted,
    )


def _extract_first_tool_name(completion: str) -> str | None:
    match = re.search(
        r'<tool_call>\s*\{[^}]*"name"\s*:\s*"([^"]+)"', completion
    )
    return match.group(1) if match else None
