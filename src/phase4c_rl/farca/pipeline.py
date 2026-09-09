"""FARCA Pipeline: Orchestrates Modules 1-5 for a batch of completions.

Called from FARCAGRPOTrainer._generate_and_score_completions after the
standard GRPO computation produces scalar advantages.
"""

from __future__ import annotations

import json

import torch

from .claim_extractor import ClaimExtractor
from .claim_verifier import ClaimVerifier
from .config import FARCAConfig
from .counterfactual_attribution import CounterfactualAttributor
from .reward import compute_farca_reward
from .advantage_reshaper import reshape_advantages_batch


def _schema_to_evidence_sentences(tool_schema: list[dict]) -> list[str]:
    """Convert structured tool schemas into evidence sentences for NLI."""
    sentences: list[str] = []
    for entry in tool_schema:
        func = entry.get("function", {})
        name = func.get("name", "")
        desc = func.get("description", "")
        if name and desc:
            sentences.append(f"The {name} tool: {desc}")
        params = func.get("parameters", {}).get("properties", {})
        for pname, pinfo in params.items():
            pdesc = pinfo.get("description", "")
            ptype = pinfo.get("type", "")
            if pdesc:
                sentences.append(
                    f"The {name} tool has a {ptype} parameter '{pname}': {pdesc}"
                )
    return sentences


def _schema_to_evidence_string(tool_schema: list[dict]) -> str:
    """Flatten tool schemas into a single evidence string."""
    return " ".join(_schema_to_evidence_sentences(tool_schema))


class FARCAPipeline:
    def __init__(self, config: FARCAConfig, tool_schema: list[dict]):
        self.config = config
        self.tool_schema = tool_schema
        self.evidence_sentences = _schema_to_evidence_sentences(tool_schema)
        self.evidence_string = " ".join(self.evidence_sentences)

        self.extractor = ClaimExtractor(
            tool_names={
                e.get("function", {}).get("name", "")
                for e in tool_schema
                if e.get("function", {}).get("name")
            }
        )

        self.verifier = ClaimVerifier(
            tool_schema=tool_schema,
            nli_model_id=config.nli_model_id,
            nli_weight=config.nli_weight,
            rule_weight=config.rule_weight,
            device=config.device,
        )

        self.attributor = CounterfactualAttributor(
            verifier=self.verifier,
            sentence_encoder_id=config.sentence_encoder_id,
            k_rel=config.k_rel,
            mu=config.mu,
            tau=config.tau,
            device=config.device,
        )

        self._step = 0

    def process_batch(
        self,
        completions: list[str],
        completion_ids: torch.Tensor,
        tokenizer,
        sequence_advantages: torch.Tensor,
        prompts: list[str] | None = None,
        system_prompt: str | None = None,
        expected_tools: list[str | None] | None = None,
    ) -> tuple[torch.Tensor, list[float]]:
        """Run the full FARCA pipeline on a batch.

        Returns:
            per_token_advantages: (B, T) tensor
            farca_fact_rewards: list of R^fact values for logging
        """
        B, T = completion_ids.shape

        # Build evidence with optional prompt/system context
        evidence = self.evidence_string
        if system_prompt:
            evidence = system_prompt + " " + evidence

        # M1: Extract claims
        all_claims = []
        all_extraction_results = []
        for i in range(B):
            ids_i = completion_ids[i].tolist()
            er = self.extractor.extract(completions[i], ids_i, tokenizer)
            all_claims.append(er.claims)
            all_extraction_results.append(er)

        # M2 + M3: Verify and compute reliability for each completion's claims
        all_scores: list[list[float]] = []
        all_weights: list[list[float]] = []
        farca_fact_rewards: list[float] = []

        for i in range(B):
            claims_i = all_claims[i]

            if not claims_i:
                all_scores.append([])
                all_weights.append([])
                farca_fact_rewards.append(0.0)
                continue

            prompt_evidence = evidence
            if prompts and i < len(prompts):
                prompt_evidence = prompts[i] + " " + evidence

            # M2: Verify
            verifications = self.verifier.verify_batch(claims_i, prompt_evidence)

            # M3: Reliability
            reliabilities = self.attributor.compute_batch(
                claims_i, verifications, self.evidence_sentences
            )

            # M4: Reward (fact component only — format/task handled by GRPOTrainer)
            expected = None
            if expected_tools and i < len(expected_tools):
                expected = expected_tools[i]

            farca_reward = compute_farca_reward(
                completion=completions[i],
                claims=claims_i,
                reliability_results=reliabilities,
                expected_tool=expected,
                format_weight=self.config.format_weight,
                answer_weight=self.config.answer_weight,
                fact_weight=self.config.fact_weight,
            )

            all_scores.append(farca_reward.claim_scores)
            all_weights.append(farca_reward.claim_weights)
            farca_fact_rewards.append(farca_reward.fact_reward)

        # M5: Reshape advantages
        completion_lengths = [
            int((completion_ids[i] != tokenizer.pad_token_id).sum().item())
            if tokenizer.pad_token_id is not None
            else T
            for i in range(B)
        ]

        # Warmup: ramp alpha from 0 to 1 over warmup_steps
        if self.config.warmup_steps > 0 and self._step < self.config.warmup_steps:
            alpha = self._step / self.config.warmup_steps
        else:
            alpha = 1.0

        per_token_advantages = reshape_advantages_batch(
            sequence_advantages=sequence_advantages,
            all_claims=all_claims,
            all_scores=all_scores,
            all_weights=all_weights,
            completion_lengths=completion_lengths,
            max_length=T,
            warmup_alpha=alpha,
        )

        self._step += 1

        return per_token_advantages.to(sequence_advantages.device), farca_fact_rewards
