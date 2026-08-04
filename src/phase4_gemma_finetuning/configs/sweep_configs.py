"""
phase4_gemma_finetuning/configs/sweep_configs.py

Two hyperparameter configurations for the Phase 4A Gemma validation sweep.
Reduced from 3 configs (Qwen) to 2: community Gemma fine-tunes converge
well with lower learning rates, so conservative is dropped.
"""
from dataclasses import dataclass


@dataclass
class SweepConfig:
    name: str
    lora_r: int
    lora_alpha: int
    learning_rate: float
    batch_size: int
    gradient_accumulation: int
    # Derived
    effective_batch: int = 0

    def __post_init__(self):
        self.effective_batch = self.batch_size * self.gradient_accumulation


SWEEP_CONFIGS = [
    SweepConfig(
        name="standard",
        lora_r=32,
        lora_alpha=64,
        learning_rate=1e-4,
        batch_size=2,
        gradient_accumulation=8,
    ),
    SweepConfig(
        name="aggressive",
        lora_r=64,
        lora_alpha=128,
        learning_rate=2e-4,
        batch_size=4,
        gradient_accumulation=4,
    ),
]

SWEEP_CONFIG_MAP = {c.name: c for c in SWEEP_CONFIGS}

# LoRA target modules for Gemma-4-12B-it (same names as Qwen2.5-Coder-14B)
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
