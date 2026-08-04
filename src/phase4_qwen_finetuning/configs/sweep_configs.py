"""
phase4_qwen_finetuning/configs/sweep_configs.py

Hyperparameter configurations for the Phase 4 validation sweep and V7 training.
Each runs for 1 epoch on A100-large (~2h each).

Best config (lowest eval_loss) is used for Phase 4B full training.
"""
from dataclasses import dataclass, field


@dataclass
class SweepConfig:
    name: str
    lora_r: int
    lora_alpha: int
    learning_rate: float
    batch_size: int
    gradient_accumulation: int
    lora_dropout: float = 0.05
    max_seq_length: int = 0  # 0 = use config.yaml default
    dataset_id: str | None = None  # None = use config.yaml default
    dataset_revision: str | None = None
    epochs: int = 0  # 0 = use config.yaml default
    # Derived
    effective_batch: int = 0

    def __post_init__(self):
        self.effective_batch = self.batch_size * self.gradient_accumulation


SWEEP_CONFIGS = [
    SweepConfig(
        name="conservative",
        lora_r=16,
        lora_alpha=32,
        learning_rate=1.5e-4,
        batch_size=1,
        gradient_accumulation=16,
    ),
    SweepConfig(
        name="standard",
        lora_r=32,
        lora_alpha=64,
        learning_rate=2e-4,
        batch_size=2,
        gradient_accumulation=8,
    ),
    SweepConfig(
        name="aggressive",
        lora_r=64,
        lora_alpha=128,
        learning_rate=3e-4,
        batch_size=4,
        gradient_accumulation=4,
    ),
    SweepConfig(
        name="v7_mixed",
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        learning_rate=1.5e-4,
        batch_size=2,
        gradient_accumulation=8,
        max_seq_length=8192,
        dataset_id="cmndcntrlcyber/code-trainer-v7-mixed",
        dataset_revision="main",
        epochs=1,
    ),
]

SWEEP_CONFIG_MAP = {c.name: c for c in SWEEP_CONFIGS}

# LoRA target modules for Qwen2.5-Coder-14B
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
