---
base_model: google/gemma-4-26B-A4B-it
library_name: peft
license: apache-2.0
tags:
- code-generation
- tool-calling
- lora
- peft
- gemma4
- code-trainer
- agentic
datasets:
- cmndcntrlcyber/code-trainer-v9-mixed
pipeline_tag: text-generation
---

# gemma4-26b-a4b-code-trainer-aggressive-full1

LoRA adapter for **google/gemma-4-26B-A4B-it**, fine-tuned on the
[`code-trainer-v9-mixed`](https://huggingface.co/datasets/cmndcntrlcyber/code-trainer-v9-mixed)
dataset with the `aggressive` LoRA configuration. This is the **full
1-epoch SFT** run — the Gemma pipeline's equivalent of the Qwen V9 SFT
stage. The DAPT adapter
([`gemma4-26b-a4b-dapt-offsec`](https://huggingface.co/cmndcntrlcyber/gemma4-26b-a4b-dapt-offsec))
is merged into the base model before SFT training begins.

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

## Intended use

* **Direct use:** load the adapter on top of `google/gemma-4-26B-A4B-it`
  (with DAPT merged) for instruction-following code generation, tool calling,
  and multi-turn agent behaviour in the 8 dataset languages (Python,
  JavaScript, TypeScript, Java, Go, Rust, C++, C#).
* **Downstream:** merge into the base model for RL stages (FARCA-GRPO, DPO)
  and eventual GGUF quantization — see
  [`gemma26b-offsec-coder-gguf`](https://huggingface.co/cmndcntrlcyber/gemma26b-offsec-coder-gguf).
* **Out of scope:** this adapter was not trained for safety alignment, RLHF,
  or non-code tasks.

## Training data

* **Dataset:** [`cmndcntrlcyber/code-trainer-v9-mixed`](https://huggingface.co/datasets/cmndcntrlcyber/code-trainer-v9-mixed)
  (same dataset as the Qwen V9 SFT stage)
* **Format:** Unified ChatML, tool calls formatted via `apply_chat_template(tools=...)`
* **DAPT foundation:** the DAPT adapter
  ([`gemma4-26b-a4b-dapt-offsec`](https://huggingface.co/cmndcntrlcyber/gemma4-26b-a4b-dapt-offsec))
  is merged into the base model before SFT, grounding the model in offensive
  security patterns.

## Training procedure

| Knob | Value |
|---|---|
| Base model | `google/gemma-4-26B-A4B-it` (with DAPT merged) |
| Adapter | LoRA (PEFT), `r = 64`, `alpha = 128`, `dropout = 0.05` |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Learning rate | 2e-4 (cosine decay) |
| Batch size | 4 |
| Gradient accumulation | 4 (effective batch = 16) |
| Epochs | 1 |
| Sequence length | 4,096 |
| Precision | bfloat16 + gradient checkpointing |

| Meta | Value |
|---|---|
| Hardware | HF Jobs `a100-large` (1x A100 80 GB) |
| Entry point | `src/phase4_gemma_finetuning/hf_skills/train_entry.py` |
| Config | `src/config/pipeline-gemma26b.yml` (`gemma_finetuning` section) |
| eval_loss | **0.4195** (from sweep) |

## Evaluation

### Sweep result

| Metric | Value |
|---|---|
| eval_loss | **0.4195** |

### Adapter chain position

```
google/gemma-4-26B-A4B-it
  └─ merge: cmndcntrlcyber/gemma4-26b-a4b-dapt-offsec         (DAPT)
      └─ LoRA: cmndcntrlcyber/gemma4-26b-a4b-code-trainer-aggressive-full1  (this adapter)
```

## Deployment notes

* **Inference target:** RTX 5060 Ti 16 GB (after full merge + GGUF
  quantization to Q4_K_M or IQ4_XS).
* **Context length:** 4,096 tokens recommended for inference; the base model
  supports 256K but training used 4,096.

## How to use

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import unwrap_clippable_linear

base_id = "google/gemma-4-26B-A4B-it"
dapt_id = "cmndcntrlcyber/gemma4-26b-a4b-dapt-offsec"
sft_id = "cmndcntrlcyber/gemma4-26b-a4b-code-trainer-aggressive-full1"

tokenizer = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForCausalLM.from_pretrained(
    base_id, torch_dtype=torch.bfloat16, device_map="auto",
)
unwrap_clippable_linear(model)

# Merge DAPT adapter into base weights
model = PeftModel.from_pretrained(model, dapt_id)
model = model.merge_and_unload()

# Load SFT adapter (active LoRA)
model = PeftModel.from_pretrained(model, sft_id)
model.eval()

messages = [
    {"role": "user", "content": "Write a Python function that performs an ARP scan on a /24 subnet using scapy."},
]
inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", add_generation_prompt=True,
).to(model.device)
out = model.generate(inputs, max_new_tokens=512, do_sample=False)
print(tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
```

## Limitations

* **Shared layers only.** LoRA cannot target routed expert FFN (3D
  `nn.Parameter`), so fine-tuning adapts shared attention + shared MLP only.
* **No safety tuning.** Inherits the base model's safety properties.
* **Adapter, not full weights.** You need the base model (~52 GB BF16) plus
  this adapter.
* **Halved LR.** All learning rates are halved vs the Qwen pipeline to
  preserve MoE routing stability.

## Reproducibility

* **Code:** [github.com/cmndcntrlcyber/code-trainer-pipeline](https://github.com/cmndcntrlcyber/code-trainer-pipeline)
  (`src/phase4_gemma_finetuning/`)
* **Training launch:**
  ```bash
  python -m src.phase4_gemma_finetuning.scripts.launch_full_training \
      --config src/config/pipeline-gemma26b.yml --best-config aggressive --wait
  ```
