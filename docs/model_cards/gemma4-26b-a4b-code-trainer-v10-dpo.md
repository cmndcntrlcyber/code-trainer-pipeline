---
base_model: google/gemma-4-26B-A4B-it
library_name: peft
license: apache-2.0
tags:
- dpo
- preference-learning
- tool-calling
- lora
- peft
- gemma4
- code-trainer
datasets:
- cmndcntrlcyber/code-trainer-v10-dpo-pairs
pipeline_tag: text-generation
---

# gemma4-26b-a4b-code-trainer-v10-dpo

LoRA adapter for **google/gemma-4-26B-A4B-it**, trained with **DPO**
(Direct Preference Optimization) on offensive security agent session
preference pairs. This is the **second and final RL stage** in the Gemma
pipeline — applied after FARCA-GRPO to refine reasoning quality and tool-use
patterns using human-derived preference signals.

Part of the Code-Trainer / RTPI pipeline
([GitHub](https://github.com/cmndcntrlcyber/code-trainer-pipeline)).

## Model architecture notes

Gemma 4 26B-A4B is a **Mixture-of-Experts** model: 128 experts + 1 shared
expert, 8 active per layer, 30 layers. Total parameters: 25.8B; active per
forward pass: 3.8B.

**LoRA targeting constraint:** the routed expert FFN layers use 3D
`nn.Parameter` tensors that PEFT cannot target. LoRA is applied only to
shared attention + shared MLP modules.

**DPO VRAM constraint:** DPO requires both policy and reference forward
passes. At ~52 GB BF16, two full copies exceed A100 80 GB. This adapter
uses `ref_model=None` (PEFT implicit reference mode) — DPOTrainer
automatically disables LoRA for reference forward passes, keeping a single
model copy in memory.

**Gemma4ClippableLinear:** modules must be unwrapped before PEFT operations
(handled by `src/utils.py:unwrap_clippable_linear`).

## Adapter chain

The V10 DPO adapter sits at the end of a 5-stage chain:

```
google/gemma-4-26B-A4B-it
  └─ merge: cmndcntrlcyber/gemma4-26b-a4b-dapt-offsec              (DAPT)
      └─ merge: cmndcntrlcyber/gemma4-26b-a4b-code-trainer-aggressive-full1  (SFT)
          └─ merge: cmndcntrlcyber/gemma4-26b-a4b-code-trainer-vision-sft  (Vision SFT)
              └─ merge: cmndcntrlcyber/gemma4-26b-a4b-code-trainer-v11-farca  (FARCA-GRPO)
                  └─ LoRA: cmndcntrlcyber/gemma4-26b-a4b-code-trainer-v10-dpo  (this adapter)
```

For deployment, all four are merged into the base and quantized to GGUF
(see [`gemma26b-offsec-coder-gguf`](https://huggingface.co/cmndcntrlcyber/gemma26b-offsec-coder-gguf)).

## Training data

* **Preference dataset:** [`cmndcntrlcyber/code-trainer-v10-dpo-pairs`](https://huggingface.co/datasets/cmndcntrlcyber/code-trainer-v10-dpo-pairs)
* **Total pairs:** 870 (783 train / 87 validation)
* **Format:** `{"prompt": "...", "chosen": "...", "rejected": "..."}`
* **Positives:** extracted from 221 OCO sessions (HackTheBox, TryHackMe,
  bug bounty — real offensive security agent traces with tool calls)
* **Negatives:** synthetically generated via 4 degradation strategies
  (refusal, stripped tool calls, truncated, hallucinated commands)
* **Persona pairs (V4.0):** ~400 additional pairs where the chosen response
  identifies as Nexus (offsec framing, MITRE ATT&CK) and the rejected
  response is a vanilla AI assistant reply. Built from identity training
  examples via `build_dpo_pairs.py --identity-examples`.

## Training procedure

| Knob | Value |
|---|---|
| Base model | `google/gemma-4-26B-A4B-it` (with DAPT + SFT + FARCA merged) |
| Method | DPO (Direct Preference Optimization) |
| Reference model | PEFT implicit (`ref_model=None`) — single model copy |
| Adapter | LoRA (PEFT), `r = 32`, `alpha = 64`, `dropout = 0.05` |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Learning rate | 5e-7 (halved from Qwen's 1e-6 for MoE routing stability) |
| Beta | 0.1 |
| Max length | 2,048 (reduced from 4,096 due to DPO dual forward pass VRAM constraints on 52 GB model) |
| Batch size | 1 |
| Gradient accumulation | 8 (effective batch = 8) |
| Precision | bfloat16 + gradient checkpointing |

| Meta | Value |
|---|---|
| Hardware | HF Jobs `a100-large` (1x A100 80 GB) |
| Entry point | `src/phase4c_rl/hf_skills/dpo_entry.py` |
| Config | `src/config/pipeline-gemma26b.yml` (`rl_training.dpo` section) |

## Intended use

* **Direct use:** load the full adapter chain on top of
  `google/gemma-4-26B-A4B-it` for instruction-following code generation,
  tool calling, and multi-turn agent behaviour with preference-aligned
  reasoning.
* **Downstream:** merge the full adapter chain into the base model and
  quantize to Q4_K_M / IQ4_XS GGUF for local serving via llama.cpp,
  Ollama, or LM Studio.
* **Out of scope:** this adapter was not trained for safety alignment or
  non-code tasks.

## Deployment notes

* **Inference target:** RTX 5060 Ti 16 GB (after full merge + GGUF
  quantization).
* **Recommended quant:** Q4_K_M (~14.5 GB, 28/30 layers GPU) or IQ4_XS
  (~13 GB, fully GPU-resident).
* **Context length:** 4,096 tokens recommended.

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
dpo_id = "cmndcntrlcyber/gemma4-26b-a4b-code-trainer-v10-dpo"

tokenizer = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForCausalLM.from_pretrained(
    base_id, torch_dtype=torch.bfloat16, device_map="auto",
)
unwrap_clippable_linear(model)

# Merge DAPT, SFT, and FARCA adapters into base weights
model = PeftModel.from_pretrained(model, dapt_id)
model = model.merge_and_unload()
model = PeftModel.from_pretrained(model, sft_id)
model = model.merge_and_unload()
model = PeftModel.from_pretrained(model, farca_id)
model = model.merge_and_unload()

# Load DPO adapter (active LoRA)
model = PeftModel.from_pretrained(model, dpo_id)
model.eval()

messages = [
    {"role": "user", "content": "Scan the target 10.10.10.5 for open ports and identify running services."},
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
* **Reduced max_length.** 2,048 instead of 4,096 due to DPO dual forward
  pass VRAM constraints — longer preference pairs are truncated.
* **PEFT implicit reference.** The reference model is approximated by
  disabling LoRA weights; this is not identical to a separate frozen copy
  but is the only viable approach within 80 GB VRAM for a 52 GB model.
* **No safety tuning.** Inherits the base model's safety properties.

## Reproducibility

* **Code:** [github.com/cmndcntrlcyber/code-trainer-pipeline](https://github.com/cmndcntrlcyber/code-trainer-pipeline)
  (`src/phase4c_rl/`)
* **Preference dataset build:**
  ```bash
  python -m src.phase4c_rl.data.collect_negatives --synthetic
  python -m src.phase4c_rl.data.build_dpo_pairs
  ```
* **Training launch:**
  ```bash
  python -m src.phase4c_rl.scripts.launch_dpo \
      --config src/config/pipeline-gemma26b.yml --wait
  ```
