"""FARCA Module 3: Counterfactual evidence attribution and reliability estimation.

Determines how much a verification score depends on specific evidence.
High dependency = high reliability (the verifier used real evidence).
Low dependency = low reliability (surface-pattern match, downweight).

Implements:
  E* = TopK cos(phi(e_l), phi(c))       most relevant evidence
  K' = K \\ E*                           counterfactual evidence set
  h_tilde = Verifier(K', c)             counterfactual score
  delta = |r - r_tilde|                 evidence dependency
  w = sigmoid((delta - mu) / tau)       reliability weight
  r_fact = w * r                        reliability-weighted score
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .claim_extractor import AtomicClaim
from .claim_verifier import ClaimVerifier, VerificationResult

# Lazy-loaded sentence encoder
_encoder = None


@dataclass
class ReliabilityResult:
    original_score: float       # r
    counterfactual_score: float  # r_tilde
    evidence_dependency: float   # delta = |r - r_tilde|
    reliability_weight: float    # w = sigmoid((delta - mu) / tau)
    weighted_score: float        # r_fact = w * r
    relevant_evidence: list[str]


class CounterfactualAttributor:
    def __init__(
        self,
        verifier: ClaimVerifier,
        sentence_encoder_id: str = "all-MiniLM-L6-v2",
        k_rel: int = 1,
        mu: float = 0.16,
        tau: float = 0.20,
        device: str = "cpu",
    ):
        self.verifier = verifier
        self.encoder_id = sentence_encoder_id
        self.k_rel = k_rel
        self.mu = mu
        self.tau = tau
        self.device = device
        self._encoder_loaded = False

    def compute_reliability(
        self,
        claim: AtomicClaim,
        verification: VerificationResult,
        evidence_sentences: list[str],
    ) -> ReliabilityResult:
        # Rule-based verifications are inherently reliable
        if verification.method == "rule":
            return ReliabilityResult(
                original_score=verification.factual_score,
                counterfactual_score=0.0,
                evidence_dependency=1.0,
                reliability_weight=1.0,
                weighted_score=verification.factual_score,
                relevant_evidence=[],
            )

        if len(evidence_sentences) <= self.k_rel:
            return ReliabilityResult(
                original_score=verification.factual_score,
                counterfactual_score=0.0,
                evidence_dependency=abs(verification.factual_score),
                reliability_weight=self._sigmoid_weight(
                    abs(verification.factual_score)
                ),
                weighted_score=(
                    self._sigmoid_weight(abs(verification.factual_score))
                    * verification.factual_score
                ),
                relevant_evidence=evidence_sentences[:],
            )

        relevant, relevant_indices = self._find_relevant_evidence(
            claim.text, evidence_sentences, self.k_rel
        )

        counterfactual_evidence = self._remove_evidence(
            evidence_sentences, relevant_indices
        )

        cf_verification = self.verifier.verify(
            claim, counterfactual_evidence
        )
        r_tilde = cf_verification.factual_score

        r = verification.factual_score
        delta = abs(r - r_tilde)
        w = self._sigmoid_weight(delta)
        r_fact = w * r

        return ReliabilityResult(
            original_score=r,
            counterfactual_score=r_tilde,
            evidence_dependency=delta,
            reliability_weight=w,
            weighted_score=r_fact,
            relevant_evidence=relevant,
        )

    def compute_batch(
        self,
        claims: list[AtomicClaim],
        verifications: list[VerificationResult],
        evidence_sentences: list[str],
    ) -> list[ReliabilityResult]:
        return [
            self.compute_reliability(c, v, evidence_sentences)
            for c, v in zip(claims, verifications)
        ]

    # ── internals ──

    def _ensure_encoder(self):
        if self._encoder_loaded:
            return
        global _encoder
        if _encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                _encoder = SentenceTransformer(self.encoder_id, device=self.device)
            except Exception:
                _encoder = None
        self._encoder_loaded = True

    def _find_relevant_evidence(
        self,
        claim_text: str,
        evidence_sentences: list[str],
        k: int,
    ) -> tuple[list[str], list[int]]:
        self._ensure_encoder()

        if _encoder is None:
            indices = list(range(min(k, len(evidence_sentences))))
            return [evidence_sentences[i] for i in indices], indices

        import numpy as np

        claim_emb = _encoder.encode([claim_text], normalize_embeddings=True)
        ev_emb = _encoder.encode(evidence_sentences, normalize_embeddings=True)
        sims = np.dot(ev_emb, claim_emb.T).squeeze(-1)
        top_indices = np.argsort(sims)[-k:][::-1].tolist()

        return [evidence_sentences[i] for i in top_indices], top_indices

    @staticmethod
    def _remove_evidence(
        evidence_sentences: list[str],
        remove_indices: list[int],
    ) -> str:
        remove_set = set(remove_indices)
        kept = [s for i, s in enumerate(evidence_sentences) if i not in remove_set]
        return " ".join(kept)

    def _sigmoid_weight(self, delta: float) -> float:
        x = (delta - self.mu) / self.tau
        x = max(-20.0, min(20.0, x))
        return 1.0 / (1.0 + math.exp(-x))
