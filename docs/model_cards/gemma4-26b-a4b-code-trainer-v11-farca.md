---
base_model: google/gemma-4-26B-A4B-it
library_name: peft
license: apache-2.0
tags:
- reinforcement-learning
- farca
- grpo
- tool-calling
- lora
- peft
- gemma4
- code-trainer
- agentic
datasets:
- cmndcntrlcyber/code-trainer-v10-grpo-prompts
pipeline_tag: text-generation
---

# gemma4-26b-a4b-code-trainer-v11-farca

LoRA adapter for **google/gemma-4-26B-A4B-it**, reinforcement-learned with
**FARCA-GRPO** (Fact-Aligned Reliability-Aware Credit Assignment + Group
Relative Policy Optimization) on a combined rule-based and NLI reward
signal. This is the **V11 FARCA release** — the first RL stage in the Gemma
pipeline, trained on top of the merged DAPT + SFT adapter chain.

Part of the Code-Trainer / RTPI pipeline
([GitHub](https://github.com/cmndcntrlcyber/code-trainer-pipeline)).

## Model architecture notes

Gemma 4 26B-A4B is a **Mixture-of-Experts** model: 128 experts + 1 shared
expert, 8 active per layer, 30 layers. Total parameters: 25.8B; active per
forward pass: 3.8B.

**LoRA targeting constraint:** the routed expert FFN layers use 3D
`nn.Parameter` tensors that PEFT cannot target. LoRA is applied only to
shared attention + shared MLP modules. All learning rates are halved vs the
Qwen pipeline for MoE routing stability.

**Gemma4ClippableLinear:** modules must be unwrapped before PEFT operations
(handled by `src/utils.py:unwrap_clippable_linear`).

## What is FARCA-GRPO

FARCA extends standard GRPO with two additional reward components:

1. **Fact-aligned claim extraction** — extracts verifiable claims from each
   completion (rule-based backend for speed and determinism).
2. **Reliability-aware credit assignment** — blends NLI-based factuality
   scoring with rule-based tool-call formatting reward, weighted by
   configurable `nli_weight` and `rule_weight` parameters.

The advantage estimate mixes standard GRPO group-relative advantages with
FARCA's per-claim reliability scores, controlled by `mu` (mixing
coefficient) and `tau` (temperature). A `warmup_steps` ramp (0 to 1)
prevents advantage variance spikes during early training — critical for MoE
routing stability.

## Adapter chain

The V11 FARCA adapter cannot be loaded directly on the base model — it
expects the DAPT and SFT adapters to be merged first:

```
google/gemma-4-26B-A4B-it
  └─ merge: cmndcntrlcyber/gemma4-26b-a4b-dapt-offsec              (DAPT)
      └─ merge: cmndcntrlcyber/gemma4-26b-a4b-code-trainer-aggressive-full1  (SFT)
          └─ merge: cmndcntrlcyber/gemma4-26b-a4b-code-trainer-vision-sft  (Vision SFT)
              └─ LoRA: cmndcntrlcyber/gemma4-26b-a4b-code-trainer-v11-farca  (this adapter)
```

For deployment, all four are merged into the base and quantized to GGUF
(see [`gemma26b-offsec-coder-gguf`](https://huggingface.co/cmndcntrlcyber/gemma26b-offsec-coder-gguf)).

## Training data

* **Prompt dataset:** [`cmndcntrlcyber/code-trainer-v10-grpo-prompts`](https://huggingface.co/datasets/cmndcntrlcyber/code-trainer-v10-grpo-prompts)
* **Max prompts:** 400 (reduced from Qwen's 500 for budget constraints)

Each prompt is formatted via `apply_chat_template` with the Nexus system
prompt and full tool definitions.

## Reward function

FARCA blends two reward signals:

| Component | Weight | Source |
|---|---|---|
| Rule-based tool-call formatting | 0.6 | 6-component reward (V4.0: includes persona_aligned_reasoning at 10% weight) |
| NLI factuality scoring | 0.4 | HHEM (`vectara/hallucination_evaluation_model`) |

### FARCA parameters

| Parameter | Value |
|---|---|
| `claim_extraction_backend` | `rule_based` |
| `nli_weight` | 0.4 |
| `rule_weight` | 0.6 |
| `mu` (mixing coefficient) | 0.16 |
| `tau` (temperature) | 0.20 |
| `warmup_steps` | 50 (ramps FARCA mixing 0 to 1 — required for MoE stability) |

## Training procedure

| Knob | Value |
|---|---|
| Base model | `google/gemma-4-26B-A4B-it` (with DAPT + SFT merged) |
| Method | FARCA-GRPO |
| Adapter | LoRA (PEFT), `r = 32`, `alpha = 64`, `dropout = 0.05` |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Learning rate | 2.5e-7 (halved from Qwen's 5e-7 for MoE routing stability) |
| KL coefficient (beta) | 0.1 |
| Generations per prompt | 4 |
| Max completion length | 768 tokens |
| Warmup steps | 50 |
| Precision | bfloat16 + gradient checkpointing |

| Meta | Value |
|---|---|
| Hardware | HF Jobs `a100-large` (1x A100 80 GB) |
| Entry point | `src/phase4c_rl/hf_skills/farca_grpo_entry.py` |
| Config | `src/config/pipeline-gemma26b.yml` (`rl_training.farca_grpo` section) |
| NLI model | `vectara/hallucination_evaluation_model` (~550 MB) |
| Sentence encoder | `all-MiniLM-L6-v2` (~80 MB) |

## Intended use

* **Direct use:** load the full adapter chain on top of
  `google/gemma-4-26B-A4B-it` for instruction-following code generation,
  tool calling, and multi-turn agent behaviour with improved `<tool_call>`
  tag formatting and factual grounding.
* **Downstream:** merge the full adapter chain into the base model, then
  apply DPO
  ([`gemma4-26b-a4b-code-trainer-v10-dpo`](https://huggingface.co/cmndcntrlcyber/gemma4-26b-a4b-code-trainer-v10-dpo))
  and quantize to GGUF.
* **Out of scope:** this adapter was not trained for safety alignment or
  non-code tasks. The FARCA-GRPO stage optimizes formatting and factual
  grounding, not task completion.

## Deployment notes

* **Inference target:** RTX 5060 Ti 16 GB (after full merge + GGUF
  quantization).
* **Context length:** 4,096 tokens recommended; trained at 768 max
  completion length but the base model supports 256K.

## How to use

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import unwrap_clippable_linear

base_id = "google/gemma-4-26B-A4B-it"
dapt_id = "cmndcntrlcyber/gemma4-26b-a4b-dapt-offsec"
sft_id = "cmndcntrlcyber/gemma4-26b-a4b-code-trainer-aggressive-full1"
farca_id = "cmndcntrlcyber/gemma4-26b-a4b-code-trainer-v11-farca"

tokenizer = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForCausalLM.from_pretrained(
    base_id, torch_dtype=torch.bfloat16, device_map="auto",
)
unwrap_clippable_linear(model)

# Merge DAPT and SFT adapters into base weights
model = PeftModel.from_pretrained(model, dapt_id)
model = model.merge_and_unload()
model = PeftModel.from_pretrained(model, sft_id)
model = model.merge_and_unload()

# Load FARCA-GRPO adapter (active LoRA)
model = PeftModel.from_pretrained(model, farca_id)
model.eval()

messages = [
    {"role": "user", "content": "Read the file main.py and summarise its structure."},
]
inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", add_generation_prompt=True,
).to(model.device)
out = model.generate(inputs, max_new_tokens=512, do_sample=False)
print(tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
```

## Limitations

* **Shared layers only.** LoRA cannot target routed expert FFN (3D
  `nn.Parameter`).
* **No safety tuning.** Inherits the base model's safety properties.
* **Budget-constrained.** 400 prompts (vs Qwen's 500) due to larger model
  throughput overhead.
* **FARCA warmup required.** Removing `warmup_steps` on MoE can cause
  advantage variance spikes and training instability.

## Reproducibility

* **Code:** [github.com/cmndcntrlcyber/code-trainer-pipeline](https://github.com/cmndcntrlcyber/code-trainer-pipeline)
  (`src/phase4c_rl/`)
* **Training launch:**
  ```bash
  python -m src.phase4c_rl.scripts.launch_farca_grpo \
      --config src/config/pipeline-gemma26b.yml --wait
  ```
