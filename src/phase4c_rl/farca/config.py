from dataclasses import dataclass, field


@dataclass
class FARCAConfig:
    """Configuration for the FARCA pipeline.

    Hyperparameter defaults match the FARCA paper (arXiv:2608.24350v1).
    """

    # M1: Claim extraction
    claim_extraction_backend: str = "rule_based"

    # M2: Claim verification
    nli_model_id: str = "vectara/hallucination_evaluation_model"
    nli_weight: float = 0.4
    rule_weight: float = 0.6

    # M3: Counterfactual attribution
    sentence_encoder_id: str = "all-MiniLM-L6-v2"
    k_rel: int = 1
    mu: float = 0.16
    tau: float = 0.20

    # M4: Reward weights
    format_weight: float = 1.0
    answer_weight: float = 1.0
    fact_weight: float = 1.0

    # M5: Advantage reshaping
    warmup_steps: int = 0

    # General
    device: str = "cpu"
    cache_claims: bool = True

    # Tool schema (populated at runtime)
    tool_schema: list[dict] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "FARCAConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})
