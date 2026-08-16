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
- curriculum-training
datasets:
- cmndcntrlcyber/code-trainer-v9-mixed
pipeline_tag: text-generation
---

# qwen14b-code-trainer-v9_mixed

LoRA adapter for **Qwen/Qwen2.5-Coder-14B-Instruct**, fine-tuned on the
[`code-trainer-v9-mixed`](https://huggingface.co/datasets/cmndcntrlcyber/code-trainer-v9-mixed)
dataset with a **two-phase curriculum**. This is the **V9 release** — focused on
fixing `<tool_call>` tag emission so the model produces the exact wrapper tags
Ollama needs to parse structured tool calls.

Part of the Code-Trainer / RTPI pipeline
([GitHub](https://github.com/cmndcntrlcyber/code-trainer-pipeline)).

## What changed from V8

1. **Increased tool-calling density** — Slice B expanded from 12K to 19K rows
   (tool-calling share: 31% → ~45% of the dataset).
2. **Synthetic multi-tool-call examples** — ~2K synthetic examples with 2–3
   tool calls per assistant turn, teaching the model sequential tool emission.
3. **Stop-after-tag cleanup** — all assistant turns containing `</tool_call>`
   are stripped of trailing text, teaching a clean stop signal.
4. **Tag completeness validation** — records with unmatched `<tool_call>` /
   `</tool_call>` tags are dropped during dataset build.
5. **Two-phase curriculum training** — Phase A (80% of steps) trains on the
   full mixed dataset; Phase B (20% of steps) trains on the tool-calling-only
   subset at a higher learning rate for a final "polish" pass on tag formatting.
6. **Q5_K_M default** — the deployment quantization is bumped from Q4_K_M to
   Q5_K_M for improved tag-pattern fidelity (~1.5 GB larger, still fits
   RTX 5060 Ti 16GB).

## Training data

* **Dataset:** [`cmndcntrlcyber/code-trainer-v9-mixed`](https://huggingface.co/datasets/cmndcntrlcyber/code-trainer-v9-mixed)
* **Total:** 40,401 train / 4,489 validation (90/10 split, seed 42)

| Slice | Source | Rows (train) | Purpose |
|---|---|---|---|
| A — Code generation | `cmndcntrlcyber/code-trainer-offsec-dataset` (8K subsample) | 7,074 | Preserve code-gen quality |
| B — Tool calling | `glaiveai/glaive-function-calling-v2` (19K cap) | ~15,125 | High-density tool calling |
| B+ — Multi-tool synthetic | Synthetic from Slice B pairs (~2K) | ~2,000 | Multi-call per turn |
| C — Agentic multi-turn | `greghavens/fable-5-coding-and-debugging-traces` (10K cap) | 8,994 | Multi-step agent behaviour |
| D — English instruction | `teknium/OpenHermes-2.5` (8K cap) | 7,208 | Language anchor |

* **Tool coverage:** 25,964 / 40,401 train rows (64.3%) contain tool definitions
* **Format:** Unified ChatML, tool calls formatted via `apply_chat_template(tools=...)`

## Training procedure

### Phase A — Full mixed dataset (80% of total steps)

| Knob | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-14B-Instruct` |
| Adapter | LoRA (PEFT), `r = 32`, `alpha = 64`, `dropout = 0.05` |
| Learning rate | 1.0e-4 (cosine decay, warmup ratio 0.03) |
| Batch size × grad accum | 1 × 16 (effective batch = 16) |
| Sequence length | 4,096 |
| Precision | bfloat16 + gradient checkpointing |

### Phase B — Tool-calling polish (20% of total steps)

| Knob | Value |
|---|---|
| Data | Tool-calling subset only (Slice B + B+) |
| Learning rate | 2.0e-4 |
| Warmup ratio | 0.10 |
| All other knobs | Same as Phase A |

| Meta | Value |
|---|---|
| Hardware | HF Skills `a100-large` (1× A100 80 GB) |
| Frameworks | `transformers`, `peft`, `trl` (SFTTrainer) |
| Entry point | `train_entry_v9.py` (curriculum orchestrator) |
| HF Job | [`6a74de123e1f34a7e32bb955`](https://huggingface.co/jobs/cmndcntrlcyber/6a74de123e1f34a7e32bb955) |

## Evaluation

### Version comparison

| Version | Dataset rows | Tool % | eval_loss | Key improvement |
|---|---|---|---|---|
| V6 `aggressive` | 26,126 | 0% | 0.4724 | Code-only baseline |
| V7 `v7_mixed` | 28,862 | 63.8% | — | Restore tool-calling + agent |
| V8 `v8_mixed` | 34,104 | 57.6% | 0.4837 | Fix multilingual + native format |
| **V9 `v9_mixed` (this)** | **40,401** | **64.3%** | — | Fix tag emission + curriculum |

## Intended use

* **Direct use:** load the adapter on top of `Qwen/Qwen2.5-Coder-14B-Instruct`
  for instruction-following code generation, tool calling, and multi-turn
  agent behaviour.
* **Downstream:** merge into the base model and quantize to Q5_K_M GGUF for
  local serving via llama.cpp, Ollama, or LM Studio.
* **Out of scope:** this adapter was not trained for safety alignment, RLHF,
  or non-code tasks.

## Deployment notes

* **Recommended quant:** Q5_K_M (preserves multi-token `<tool_call>` tag
  patterns better than Q4_K_M). Q4_K_M is available as a fallback.
* **Context length:** 8,192 tokens recommended; the model was trained at
  4,096 but the base model supports 32K.

## How to use

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen2.5-Coder-14B-Instruct"
adapter_id = "cmndcntrlcyber/qwen14b-code-trainer-v9_mixed"

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
  python -m src.phase2_preprocessing.scripts.build_v9_mixed_dataset \
      --config src/config/config.yaml
  ```
* **Training launch:**
  ```bash
  python -m src.phase4_qwen_finetuning.scripts.launch_v9_training \
      --config src/config/config.yaml --wait
  ```
* **W&B project:** [`rtpi-phase4-qwen14b`](https://wandb.ai/cmndcntrlcyber-c3s-consulting/rtpi-phase4-qwen14b)
