"""FARCA Module 2: Hybrid claim verification.

Verifies atomic claims against structured evidence (tool schemas +
user prompt) using a combination of rule-based schema matching and
NLI-based entailment scoring.

Implements: h_{i,j,k} = Verifier(K, c_{i,j,k}) in [0,1]
            r_{i,j,k} = 2*h - 1 in [-1,1]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .claim_extractor import AtomicClaim

# Lazy-loaded to avoid import overhead when not using NLI
_nli_model = None
_nli_tokenizer = None


@dataclass
class VerificationResult:
    claim_text: str
    entailment_score: float  # h in [0, 1]
    factual_score: float     # r = 2h - 1 in [-1, 1]
    method: str              # "rule" | "nli" | "hybrid"
    rule_checks: dict[str, bool] = field(default_factory=dict)


def _build_tool_index(schema: list[dict]) -> dict[str, dict]:
    """Index tool schemas by lowercase name for fast lookup."""
    index: dict[str, dict] = {}
    for entry in schema:
        func = entry.get("function", {})
        name = func.get("name", "")
        if name:
            index[name.lower()] = func
    return index


class ClaimVerifier:
    def __init__(
        self,
        tool_schema: list[dict],
        nli_model_id: str = "vectara/hallucination_evaluation_model",
        nli_weight: float = 0.4,
        rule_weight: float = 0.6,
        device: str = "cpu",
    ):
        self.tool_schema = tool_schema
        self.tool_index = _build_tool_index(tool_schema)
        self.tool_names_lower = set(self.tool_index.keys())
        self.nli_model_id = nli_model_id
        self.nli_weight = nli_weight
        self.rule_weight = rule_weight
        self.device = device
        self._nli_loaded = False

    def verify(
        self,
        claim: AtomicClaim,
        evidence: str,
    ) -> VerificationResult:
        if claim.claim_type == "tool_selection":
            return self._verify_tool_selection(claim, evidence)
        elif claim.claim_type == "argument_claim":
            return self._verify_argument(claim, evidence)
        else:
            return self._verify_reasoning(claim, evidence)

    def verify_batch(
        self,
        claims: list[AtomicClaim],
        evidence: str,
    ) -> list[VerificationResult]:
        return [self.verify(c, evidence) for c in claims]

    # ── rule-based verification ──

    def _verify_tool_selection(
        self, claim: AtomicClaim, evidence: str
    ) -> VerificationResult:
        mentioned = self._extract_tool_names(claim.text)
        if not mentioned:
            # Claim type says tool_selection but no known tools found —
            # check if any capitalized word looks like a tool name attempt
            import re as _re
            candidates = _re.findall(r"\b[A-Z][a-zA-Z]+\b", claim.text)
            tool_like = [c for c in candidates if c not in (
                "I", "The", "This", "First", "Then", "Next", "After",
                "Before", "Each", "Every", "Some", "Any", "All",
            )]
            if tool_like:
                checks = {f"tool_exists:{t}": False for t in tool_like}
                return VerificationResult(
                    claim_text=claim.text,
                    entailment_score=0.0,
                    factual_score=-1.0,
                    method="rule",
                    rule_checks=checks,
                )
            return self._verify_reasoning(claim, evidence)

        checks: dict[str, bool] = {}
        for name in mentioned:
            checks[f"tool_exists:{name}"] = name.lower() in self.tool_names_lower

        all_valid = all(checks.values())
        h = 1.0 if all_valid else 0.0
        return VerificationResult(
            claim_text=claim.text,
            entailment_score=h,
            factual_score=2.0 * h - 1.0,
            method="rule",
            rule_checks=checks,
        )

    def _verify_argument(
        self, claim: AtomicClaim, evidence: str
    ) -> VerificationResult:
        tool_names = self._extract_tool_names(claim.text)
        checks: dict[str, bool] = {}

        if tool_names:
            for tname in tool_names:
                func = self.tool_index.get(tname.lower(), {})
                params = func.get("parameters", {}).get("properties", {})
                param_names_lower = {k.lower() for k in params}
                arg_matches = re.findall(
                    r"\b(" + "|".join(re.escape(p) for p in params) + r")\b",
                    claim.text,
                    re.IGNORECASE,
                )
                for arg in arg_matches:
                    checks[f"arg_valid:{tname}.{arg}"] = (
                        arg.lower() in param_names_lower
                    )

        if checks:
            h = 1.0 if all(checks.values()) else 0.0
            return VerificationResult(
                claim_text=claim.text,
                entailment_score=h,
                factual_score=2.0 * h - 1.0,
                method="rule",
                rule_checks=checks,
            )

        return self._verify_reasoning(claim, evidence)

    def _verify_reasoning(
        self, claim: AtomicClaim, evidence: str
    ) -> VerificationResult:
        h = self._nli_score(claim.text, evidence)
        return VerificationResult(
            claim_text=claim.text,
            entailment_score=h,
            factual_score=2.0 * h - 1.0,
            method="nli",
        )

    # ── NLI verification ──

    def _ensure_nli_loaded(self):
        if self._nli_loaded:
            return
        global _nli_model, _nli_tokenizer
        if _nli_model is None:
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                _nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_model_id)
                _nli_model = AutoModelForSequenceClassification.from_pretrained(
                    self.nli_model_id
                )
                _nli_model.to(self.device)
                _nli_model.eval()
            except Exception:
                _nli_model = None
                _nli_tokenizer = None
        self._nli_loaded = True

    def _nli_score(self, hypothesis: str, premise: str) -> float:
        self._ensure_nli_loaded()
        if _nli_model is None or _nli_tokenizer is None:
            return 0.5

        import torch

        inputs = _nli_tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = _nli_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            # HHEM: class 0 = not hallucinated (entailed), class 1 = hallucinated
            entailment_prob = probs[0, 0].item()

        return entailment_prob

    # ── helpers ──

    def _extract_tool_names(self, text: str) -> list[str]:
        found: list[str] = []
        for name in self.tool_index:
            pattern = re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
            if pattern.search(text):
                # Return the canonical-case name from the schema
                found.append(self.tool_index[name]["name"])
        return found
