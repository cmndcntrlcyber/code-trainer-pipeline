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
- cmndcntrlcyber/code-trainer-v8-mixed
pipeline_tag: text-generation
---

# qwen14b-code-trainer-v8_mixed

LoRA adapter for **Qwen/Qwen2.5-Coder-14B-Instruct**, fine-tuned on the
[`code-trainer-v8-mixed`](https://huggingface.co/datasets/cmndcntrlcyber/code-trainer-v8-mixed)
dataset. This is the **V8 release** — a corrective iteration over V7 that
fixes multilingual hallucination and aligns tool-call formatting with Qwen2.5's
native template.

Part of the Code-Trainer / RTPI pipeline
([GitHub](https://github.com/cmndcntrlcyber/code-trainer-pipeline)).

> **Status:** the current Phase 5 GGUF conversion source. Superseded for
> training by V9, which improves `<tool_call>` tag emission fidelity.

## What changed from V7

1. **Native Qwen2.5 tool-call format** — all tool-calling examples are
   formatted via `tokenizer.apply_chat_template(tools=...)` to produce the
   exact token sequence the model expects natively. V7 used Hermes-style
   `<tool_call>` tags tokenized as regular text, which the model learned
   inconsistently.
2. **English-only filtering** — a `< 5%` non-ASCII threshold on all slices
   eliminates the multilingual garbage tokens V7 emitted at sequence boundaries.
3. **Added Slice D** — 8K English instruction-following examples from
   OpenHermes-2.5 to anchor the model's language and prevent distributional
   drift toward tool-calling-only behaviour.
4. **Switched Slice B source** — from `NousResearch/hermes-function-calling-v1`
   (Hermes format) to `glaiveai/glaive-function-calling-v2` (cleaner,
   single-source, easier to reformat into native Qwen2.5 template).

## Training data

* **Dataset:** [`cmndcntrlcyber/code-trainer-v8-mixed`](https://huggingface.co/datasets/cmndcntrlcyber/code-trainer-v8-mixed)
* **Total:** 34,104 train / 3,789 validation (90/10 split, seed 42)

| Slice | Source | Rows (train) | Purpose |
|---|---|---|---|
| A — Code generation | `cmndcntrlcyber/code-trainer-offsec-dataset` (8K subsample) | 7,118 | Preserve code-gen quality |
| B — Tool calling | `glaiveai/glaive-function-calling-v2` (12K cap) | 10,789 | Native-format tool calling |
| C — Agentic multi-turn | `greghavens/fable-5-coding-and-debugging-traces` (10K cap) | 9,015 | Multi-step agent behaviour |
| D — English instruction | `teknium/OpenHermes-2.5` (8K cap) | 7,182 | Language anchor |

* **Tool coverage:** 19,651 / 34,104 train rows (57.6%) contain tool definitions
* **Format:** Unified ChatML, tool calls formatted via `apply_chat_template(tools=...)`

## Training procedure

| Knob | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-14B-Instruct` |
| Adapter | LoRA (PEFT), `r = 32`, `alpha = 64`, `dropout = 0.05` |
| Learning rate | 1.0e-4 (cosine decay, warmup ratio 0.03) |
| Batch size × grad accum | 1 × 16 (effective batch = 16) |
| Epochs | 1 |
| Sequence length | 4,096 |
| Precision | bfloat16 + gradient checkpointing |
| Hardware | HF Skills `a100-large` (1× A100 80 GB) |
| Frameworks | `transformers`, `peft`, `trl` (SFTTrainer) |

Key changes from V7:
- **`learning_rate: 1.0e-4`** (was 1.5e-4) — even gentler to preserve base capabilities
- **`max_seq_length: 4096`** (was 8192) — reduced to fix A100 OOM; agent prompts
  fit at 4K after truncation
- **`batch_size: 1`** (was 2) — compensated by gradient accumulation 16

## Evaluation

| Metric | Value |
|---|---|
| eval_loss | **0.4837** |

* **HF Job:** [`6a73e7e66b79c09949c23c7e`](https://huggingface.co/jobs/cmndcntrlcyber/6a73e7e66b79c09949c23c7e)
  (status: CANCELED — adapter was pushed to Hub before the job was killed by
  HF Skills timeout enforcement)
* **Fixes confirmed:** multilingual hallucination eliminated; tool-call JSON
  emitted in native Qwen2.5 format.

### Comparison across versions

| Version | Dataset rows | eval_loss | Key fix |
|---|---|---|---|
| V6 `aggressive` | 26,126 | 0.4724 | Baseline (code-only) |
| V7 `v7_mixed` | 28,862 | — | Restore tool-calling + agent |
| **V8 `v8_mixed` (this)** | **34,104** | **0.4837** | Fix multilingual + native format |
| V9 `v9_mixed` | 40,401 | — | Fix tag emission + curriculum |

## Known issues (fixed in V9)

1. **`<tool_call>` tag omission** — the model outputs correct tool name and
   JSON arguments but sometimes omits the `<tool_call>` / `</tool_call>`
   wrapper tags that Ollama needs to parse structured `tool_calls`.
2. **Trailing garbage after tool calls** — some assistant turns include text
   after `</tool_call>`, weakening the stop signal.
3. **Occasional tool-name lowercasing** — `ls` instead of `LS`.

Root cause: `<tool_call>` tags are tokenized as multi-token sequences; Q4_K_M
quantization loses some fidelity on these. The JSON payload pattern is learned
strongly, the XML wrapper weakly.

## How to use

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen2.5-Coder-14B-Instruct"
adapter_id = "cmndcntrlcyber/qwen14b-code-trainer-v8_mixed"

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
  python -m src.phase2_preprocessing.scripts.build_v8_mixed_dataset \
      --config src/config/config.yaml
  ```
* **Training launch:**
  ```bash
  python -m src.phase4_qwen_finetuning.scripts.launch_validation_sweep \
      --config src/config/config.yaml --only v8_mixed --wait
  ```
* **W&B project:** [`rtpi-phase4-qwen14b`](https://wandb.ai/cmndcntrlcyber-c3s-consulting/rtpi-phase4-qwen14b)
