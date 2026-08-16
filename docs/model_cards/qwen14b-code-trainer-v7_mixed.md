---
base_model: Qwen/Qwen2.5-Coder-14B-Instruct
library_name: peft
license: apache-2.0
tags:
- code-generation
- tool-calling
- lora
- peft
- qwen2.5-coder
- code-trainer
- agentic
datasets:
- cmndcntrlcyber/code-trainer-v7-mixed
pipeline_tag: text-generation
---

# qwen14b-code-trainer-v7_mixed

LoRA adapter for **Qwen/Qwen2.5-Coder-14B-Instruct**, fine-tuned on the
[`code-trainer-v7-mixed`](https://huggingface.co/datasets/cmndcntrlcyber/code-trainer-v7-mixed)
dataset. This is the **V7 corrective-action** release — the first mixed-capability
training run after V6's distributional collapse wiped tool-calling and
agent behaviour from the model.

Part of the Code-Trainer / RTPI pipeline
([GitHub](https://github.com/cmndcntrlcyber/code-trainer-pipeline)).

> **Status:** superseded by
> [`qwen14b-code-trainer-v8_mixed`](https://huggingface.co/cmndcntrlcyber/qwen14b-code-trainer-v8_mixed),
> which fixes multilingual hallucination and tool-call format mismatches
> discovered during V7 validation. Kept on the Hub for reproducibility.

## Motivation — V6 distributional collapse

The V6 adapter was trained on 26K single-turn code-transcription examples
(100% "extract the code from this screenshot"). This created three capability
gaps:

1. **No tool calling** — zero training examples with function/tool calls
2. **No multi-turn reasoning** — every example was one-shot input→output
3. **No complex instruction following** — every system prompt was identical

V7 corrects this by blending three capability slices into a single
mixed dataset.

## Training data

* **Dataset:** [`cmndcntrlcyber/code-trainer-v7-mixed`](https://huggingface.co/datasets/cmndcntrlcyber/code-trainer-v7-mixed)
* **Total:** 28,862 train / 3,206 validation (90/10 split, seed 42)

| Slice | Source | Rows (train) | Purpose |
|---|---|---|---|
| A — Code generation | `cmndcntrlcyber/code-trainer-offsec-dataset` (8K subsample) | 7,191 | Preserve V6 code-gen quality |
| B — Tool/function calling | `NousResearch/hermes-function-calling-v1` (5 configs) | 10,381 | Restore `<tool_call>` emission |
| C — Agentic multi-turn | `greghavens/fable-5-coding-and-debugging-traces` | 11,290 | Teach multi-step agent behaviour |

* **Tool coverage:** 18,413 / 28,862 train rows (63.8%) contain tool definitions
* **Format:** Unified ChatML with Hermes-style `<tool_call>` XML tags, compatible
  with Qwen2.5's native chat template.

## Training procedure

| Knob | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-14B-Instruct` |
| Adapter | LoRA (PEFT), `r = 32`, `alpha = 64`, `dropout = 0.05` |
| Learning rate | 1.5e-4 (cosine decay, warmup ratio 0.05) |
| Batch size × grad accum | 2 × 8 (effective batch = 16) |
| Epochs | 1 |
| Sequence length | 8,192 |
| Precision | bfloat16 + gradient checkpointing |
| Hardware | HF Skills `a100-large` (1× A100 80 GB) |
| Frameworks | `transformers`, `peft`, `trl` (SFTTrainer) |

Key changes from V6 (`aggressive`):
- **`lora_r: 32`** (was 64) — less aggressive LoRA to reduce capability overwrite
- **`learning_rate: 1.5e-4`** (was 3e-4) — gentler to preserve tool-calling circuits
- **`max_seq_length: 8192`** (was 2048) — agent prompts + tool specs need more context

## Validation

V7 validation tested all three capabilities via separate HF Jobs:

| Check | Job ID | Status | Target |
|---|---|---|---|
| Tool calling (10 scenarios) | `6a7606d43e1f34a7e32bd7e7` | COMPLETED | ≥80% valid calls |
| Agent behaviour (5 scenarios) | `6a7606d73e1f34a7e32bd7e9` | COMPLETED | Progress on ≥3 tasks |
| eval_loss | — | — | < 0.50 |
| GSM8K (forgetting check) | — | — | ≥ 0.60 flexible-extract |

## Known issues (fixed in V8)

1. **Multilingual hallucination** — the model appends Thai, Russian, or
   Chinese garbage tokens at sequence boundaries after tool-call JSON.
   Root cause: Qwen2.5's large multilingual vocabulary + low confidence at
   end-of-generation.
2. **Tool-call format mismatch** — Slice B (Hermes format) uses `<tool_call>`
   tags tokenized as regular text tokens, not the native Qwen2.5 tool-call
   template produced by `apply_chat_template(tools=...)`. The model learned
   the JSON payload but not the exact token sequence Ollama expects.

## How to use

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen2.5-Coder-14B-Instruct"
adapter_id = "cmndcntrlcyber/qwen14b-code-trainer-v7_mixed"

tokenizer = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForCausalLM.from_pretrained(
    base_id, dtype=torch.bfloat16, device_map="auto",
)
model = PeftModel.from_pretrained(model, adapter_id)
model.eval()

messages = [
    {"role": "system", "content": "You are a coding assistant with tool access."},
    {"role": "user", "content": "Read the file main.py and summarise its structure."},
]
inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", add_generation_prompt=True,
).to(model.device)
out = model.generate(inputs, max_new_tokens=512, do_sample=False)
print(tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
```

## Reproducibility

* **Code:** [github.com/cmndcntrlcyber/code-trainer-pipeline](https://github.com/cmndcntrlcyber/code-trainer-pipeline)
* **Dataset build:**
  ```bash
  python -m src.phase2_preprocessing.scripts.build_v7_mixed_dataset \
      --config src/config/config.yaml
  ```
* **Training launch:**
  ```bash
  python -m src.phase4_qwen_finetuning.scripts.launch_validation_sweep \
      --config src/config/config.yaml --only v7_mixed --wait
  ```
* **Validation launch:**
  ```bash
  python -m src.phase4_qwen_finetuning.scripts.launch_v7_validation \
      --config src/config/config.yaml --wait
  ```
* **W&B project:** [`rtpi-phase4-qwen14b`](https://wandb.ai/cmndcntrlcyber-c3s-consulting/rtpi-phase4-qwen14b)
