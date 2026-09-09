"""Tests for FARCA Module 5: Advantage Reshaper."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.phase4c_rl.farca.claim_extractor import AtomicClaim
from src.phase4c_rl.farca.advantage_reshaper import reshape_advantages, reshape_advantages_batch


def _claim(start: int, end: int, idx: int = 0) -> AtomicClaim:
    return AtomicClaim(
        text="test claim", source_sentence="test sentence",
        sentence_index=0, claim_index=idx,
        token_span=(start, end), claim_type="tool_selection",
    )


class TestReshapeAdvantages:
    def test_no_claims_returns_uniform(self):
        result = reshape_advantages(
            advantage=2.0, claims=[], claim_scores=[], claim_weights=[],
            completion_length=10,
        )
        assert result.shape == (10,)
        assert torch.allclose(result, torch.full((10,), 2.0))

    def test_high_reliability_positive_claim_boosts(self):
        claims = [_claim(2, 5)]
        result = reshape_advantages(
            advantage=1.0, claims=claims,
            claim_scores=[0.8], claim_weights=[1.0],
            completion_length=10,
        )
        # Covered tokens (2-4): A_fact = 0.8 * |1.0| = 0.8
        # A_tilde = (1-1.0)*1.0 + 1.0*0.8 = 0.8
        assert result[3].item() < 1.0  # pushed toward 0.8 from 1.0
        # Uncovered tokens: remain at 1.0
        assert result[0].item() == 1.0
        assert result[7].item() == 1.0

    def test_high_reliability_negative_claim_suppresses(self):
        claims = [_claim(2, 5)]
        result = reshape_advantages(
            advantage=1.0, claims=claims,
            claim_scores=[-0.9], claim_weights=[1.0],
            completion_length=10,
        )
        # Covered tokens: A_fact = -0.9 * |1.0| = -0.9
        # A_tilde = (1-1.0)*1.0 + 1.0*(-0.9) = -0.9
        assert result[3].item() < 0  # suppressed

    def test_low_reliability_stays_near_original(self):
        claims = [_claim(2, 5)]
        result = reshape_advantages(
            advantage=1.0, claims=claims,
            claim_scores=[0.8], claim_weights=[0.01],
            completion_length=10,
        )
        # A_tilde = (1-0.01)*1.0 + 0.01*0.8 = 0.998
        assert abs(result[3].item() - 1.0) < 0.05

    def test_overlapping_claims_averaged(self):
        claims = [_claim(2, 6, 0), _claim(4, 8, 1)]
        result = reshape_advantages(
            advantage=1.0, claims=claims,
            claim_scores=[0.5, -0.5], claim_weights=[1.0, 1.0],
            completion_length=10,
        )
        # Tokens 4-5 covered by both claims: average of 0.5 and -0.5 = 0.0
        assert abs(result[4].item()) < 0.01
        assert abs(result[5].item()) < 0.01


class TestReshapeBatch:
    def test_output_shape(self):
        adv = torch.tensor([1.0, -0.5])
        claims = [[_claim(0, 3)], []]
        scores = [[0.8], []]
        weights = [[1.0], []]
        result = reshape_advantages_batch(
            adv, claims, scores, weights,
            completion_lengths=[5, 5], max_length=8,
        )
        assert result.shape == (2, 8)

    def test_warmup_alpha_zero_equals_grpo(self):
        adv = torch.tensor([2.0])
        claims = [[_claim(0, 5)]]
        scores = [[0.9]]
        weights = [[1.0]]
        result = reshape_advantages_batch(
            adv, claims, scores, weights,
            completion_lengths=[5], max_length=5,
            warmup_alpha=0.0,
        )
        assert torch.allclose(result[0, :5], torch.full((5,), 2.0))

    def test_warmup_alpha_one_equals_full_farca(self):
        adv = torch.tensor([2.0])
        claims = [[_claim(0, 5)]]
        scores = [[0.5]]
        weights = [[1.0]]
        result_full = reshape_advantages_batch(
            adv, claims, scores, weights,
            completion_lengths=[5], max_length=5,
            warmup_alpha=1.0,
        )
        result_direct = reshape_advantages(2.0, claims[0], scores[0], weights[0], 5)
        assert torch.allclose(result_full[0, :5], result_direct)

    def test_padding_tokens_are_zero(self):
        adv = torch.tensor([1.0])
        result = reshape_advantages_batch(
            adv, [[]], [[]], [[]],
            completion_lengths=[3], max_length=8,
        )
        assert result[0, 3:].sum().item() == 0.0
