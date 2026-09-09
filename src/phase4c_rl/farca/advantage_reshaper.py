"""FARCA Module 5: Per-token advantage construction.

Converts scalar (B,) GRPO advantages into per-token (B, T) advantages
using claim-level factual scores and reliability weights.

For each token t in completion i:
  C(i,t) = {claims covering token t}
  if C(i,t) non-empty:
    A_fact[k] = r[k] * |A_i|
    A_tilde[k] = (1 - w[k]) * A_i + w[k] * A_fact[k]
    A_hat[i,t] = mean(A_tilde for claims in C(i,t))
  else:
    A_hat[i,t] = A_i
"""

from __future__ import annotations

import torch

from .claim_extractor import AtomicClaim


def reshape_advantages(
    advantage: float,
    claims: list[AtomicClaim],
    claim_scores: list[float],
    claim_weights: list[float],
    completion_length: int,
) -> torch.Tensor:
    """Reshape a single scalar advantage into per-token advantages.

    Returns: Tensor of shape (completion_length,)
    """
    token_adv = torch.full((completion_length,), advantage, dtype=torch.float32)

    if not claims:
        return token_adv

    abs_a = abs(advantage)

    # Build per-claim interpolated advantages
    claim_tilde: list[float] = []
    for r, w in zip(claim_scores, claim_weights):
        a_fact = r * abs_a
        a_interp = (1.0 - w) * advantage + w * a_fact
        claim_tilde.append(a_interp)

    # Accumulate per-token: sum of covering claim advantages and count
    accum = torch.zeros(completion_length, dtype=torch.float64)
    count = torch.zeros(completion_length, dtype=torch.int32)

    for claim, a_tilde in zip(claims, claim_tilde):
        start, end = claim.token_span
        start = max(0, start)
        end = min(end, completion_length)
        if start >= end:
            continue
        accum[start:end] += a_tilde
        count[start:end] += 1

    # Average where covered, fallback to scalar elsewhere
    covered = count > 0
    token_adv[covered] = (accum[covered] / count[covered].double()).float()

    return token_adv


def reshape_advantages_batch(
    sequence_advantages: torch.Tensor,
    all_claims: list[list[AtomicClaim]],
    all_scores: list[list[float]],
    all_weights: list[list[float]],
    completion_lengths: list[int],
    max_length: int,
    warmup_alpha: float = 1.0,
) -> torch.Tensor:
    """Batch reshape: returns (B, T) tensor of per-token advantages.

    Args:
        sequence_advantages: (B,) scalar advantages from GRPO
        all_claims: per-completion claim lists
        all_scores: per-completion claim factual scores r_{i,j,k}
        all_weights: per-completion claim reliability weights w_{i,j,k}
        completion_lengths: actual token count per completion
        max_length: padded sequence length T
        warmup_alpha: mixing coefficient in [0, 1]. 0 = pure GRPO, 1 = full FARCA.
    """
    B = sequence_advantages.shape[0]
    result = torch.zeros(B, max_length, dtype=torch.float32)

    for i in range(B):
        adv_i = sequence_advantages[i].item()
        clen = min(completion_lengths[i], max_length)

        farca_adv = reshape_advantages(
            advantage=adv_i,
            claims=all_claims[i],
            claim_scores=all_scores[i],
            claim_weights=all_weights[i],
            completion_length=clen,
        )

        if warmup_alpha < 1.0:
            grpo_adv = torch.full((clen,), adv_i, dtype=torch.float32)
            farca_adv = (1.0 - warmup_alpha) * grpo_adv + warmup_alpha * farca_adv

        result[i, :clen] = farca_adv

    return result
