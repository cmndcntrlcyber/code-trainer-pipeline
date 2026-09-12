---
base_model: google/gemma-4-26B-A4B-it
library_name: peft
license: apache-2.0
tags:
- dapt
- domain-adaptation
- offensive-security
- lora
- peft
- gemma4
- code-trainer
datasets:
- cmndcntrlcyber/gemma4-26b-a4b-dapt-offsec-corpus
pipeline_tag: text-generation
---

# gemma4-26b-a4b-dapt-offsec

LoRA adapter for **google/gemma-4-26B-A4B-it**, domain-adaptive continued
pretrained (DAPT) on an offensive security code corpus. This adapter grounds the
model in security tooling patterns — exploit frameworks, C2 infrastructure,
network reconnaissance, privilege escalation — before downstream SFT and RL
stages.

Part of the Code-Trainer / RTPI pipeline
([GitHub](https://github.com/cmndcntrlcyber/code-trainer-pipeline)).

## Model architecture notes

Gemma 4 26B-A4B is a **Mixture-of-Experts** model: 128 experts + 1 shared
expert, 8 active per layer, 30 layers (25 sliding-window @ 1024 + 5 full
attention). Total parameters: 25.8B; active per forward pass: 3.8B.

**LoRA targeting constraint:** the routed expert FFN layers use 3D
`nn.Parameter` tensors that PEFT cannot target. LoRA is applied only to
shared attention + shared MLP modules (`q_proj`, `k_proj`, `v_proj`,
`o_proj`, `gate_proj`, `up_proj`, `down_proj`). Domain vocabulary is still
injected through these shared pathways.

**Gemma4ClippableLinear:** Gemma 4 wraps linear layers in
`Gemma4ClippableLinear` modules that must be unwrapped before PEFT
operations. This is handled by `src/utils.py:unwrap_clippable_linear`.

## Intended use

* **Direct use:** not recommended — this is a domain-adaptation adapter, not
  instruction-tuned beyond the base model's existing capabilities.
* **Downstream:** merge into the base model before SFT training. The Gemma
  pipeline merges this adapter first, then trains the SFT adapter
  ([`gemma4-26b-a4b-code-trainer-aggressive-full1`](https://huggingface.co/cmndcntrlcyber/gemma4-26b-a4b-code-trainer-aggressive-full1))
  on the merged weights.
* **Out of scope:** general-purpose chat, safety-critical applications, or
  non-code tasks.

## Training data

* **Dataset:** [`cmndcntrlcyber/gemma4-26b-a4b-dapt-offsec-corpus`](https://huggingface.co/datasets/cmndcntrlcyber/gemma4-26b-a4b-dapt-offsec-corpus)
* **Rows:** 20,680 offensive security code documents
* **Format:** `{"text": "..."}` — plain text continuation (no chat formatting)
* **Source:** offensive-security GitHub repositories, cloned and chunked by
  `src/phase3b_dapt/data/prepare_corpus.py`
* **Shared with:** the Qwen DAPT pipeline uses the same underlying corpus

## Training procedure

| Knob | Value |
|---|---|
| Base model | `google/gemma-4-26B-A4B-it` |
| Adapter | LoRA (PEFT), `r = 32`, `alpha = 64`, `dropout = 0.05` |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Learning rate | 2.5e-5 (halved from Qwen's 5e-5 for MoE routing stability) |
| Batch size | 2 |
| Gradient accumulation | 8 (effective batch = 16) |
| Epochs | 1 |
| Sequence length | 2,048 |
| Precision | bfloat16 + gradient checkpointing |

| Meta | Value |
|---|---|
| Hardware | HF Jobs `a100-large` (1x A100 80 GB) |
| Entry point | `src/phase3b_dapt/hf_skills/dapt_entry.py` |
| Config | `src/config/pipeline-gemma26b.yml` (`gemma_dapt` section) |

## Limitations

* **Shared layers only.** LoRA cannot target routed expert FFN (3D
  `nn.Parameter`), so domain adaptation is limited to shared attention and
  shared MLP pathways.
* **No safety tuning.** Inherits the base model's safety properties.
* **Not for direct use.** This adapter is a pipeline intermediate — merge it
  before SFT for best results.

## How to use

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "google/gemma-4-26B-A4B-it"
dapt_id = "cmndcntrlcyber/gemma4-26b-a4b-dapt-offsec"

tokenizer = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForCausalLM.from_pretrained(
    base_id, torch_dtype=torch.bfloat16, device_map="auto",
)

# Unwrap Gemma4ClippableLinear before PEFT operations
from src.utils import unwrap_clippable_linear
unwrap_clippable_linear(model)

model = PeftModel.from_pretrained(model, dapt_id)
model = model.merge_and_unload()  # merge into base for downstream SFT
```

## Reproducibility

* **Code:** [github.com/cmndcntrlcyber/code-trainer-pipeline](https://github.com/cmndcntrlcyber/code-trainer-pipeline)
  (`src/phase3b_dapt/`)
* **Corpus build:**
  ```bash
  python -m src.phase3b_dapt.data.prepare_corpus \
      --config src/config/pipeline-gemma26b.yml
  ```
* **Training launch:**
  ```bash
  python -m src.phase3b_dapt.scripts.launch_dapt \
      --config src/config/pipeline-gemma26b.yml --wait
  ```
