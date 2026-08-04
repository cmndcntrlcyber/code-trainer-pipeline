---
base_model: Qwen/Qwen2.5-Coder-14B-Instruct
library_name: peft
license: apache-2.0
tags:
- code-generation
- lora
- peft
- qwen2.5-coder
- code-trainer
datasets:
- cmndcntrlcyber/code-trainer-offsec-dataset
pipeline_tag: text-generation
---

# qwen14b-code-trainer-aggressive-full3

LoRA adapter for **Qwen/Qwen2.5-Coder-14B-Instruct**, trained for 3 epochs
over an 8 K-row slice of the
[`code-trainer-offsec-dataset`](https://huggingface.co/datasets/cmndcntrlcyber/code-trainer-offsec-dataset)
as the Phase 4B follow-up to the Phase 4A `aggressive` sweep winner.

> **Status:** kept on the Hub for transparency, but **not the canonical
> Phase 5 conversion target.** The 1-epoch / full-data sibling
> [`qwen14b-code-trainer-aggressive`](https://huggingface.co/cmndcntrlcyber/qwen14b-code-trainer-aggressive)
> outperformed this adapter on the full validation split (eval_loss 0.4724
> vs **0.5126**) — see the comparison below.

## Intended use

Same as the Phase 4A sibling — instruction-following code generation across
the 8 dataset languages. Use this adapter only if you specifically want to
study the 3-epoch / sliced-data behaviour.

## Training data

* **Dataset:** [`cmndcntrlcyber/code-trainer-offsec-dataset`](https://huggingface.co/datasets/cmndcntrlcyber/code-trainer-offsec-dataset)
  (revision `main`, text-only chat format).
* **Splits seen:** 8,000 train rows (`--train-limit 8000` slice of 26,126),
  500 val rows (`--val-limit 500` slice of 3,265).
* **Total samples seen:** 24,000 (3 epochs × 8 K). The Phase 4A run saw 26,126
  samples in 1 epoch.

## Training procedure

| Knob | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-14B-Instruct` |
| Adapter | LoRA (PEFT), `r = 64`, `alpha = 128`, `dropout = 0.05` |
| Learning rate | 3e-4 (cosine decay, warmup ratio 0.03) |
| Batch size × grad accum | 4 × 4 (effective batch = 16) |
| Epochs | **3** |
| Sequence length | 2,048 |
| Precision | bfloat16 + gradient checkpointing |
| Hardware | HF Skills `a100-large` |
| Frameworks | `transformers`, `peft`, `trl` (SFTTrainer) |
| Job runtime | 4 h 53 m (COMPLETED) |
| HF Job | [`69f8188e9d85bec4d76f1c5e`](https://huggingface.co/jobs/cmndcntrlcyber/69f8188e9d85bec4d76f1c5e) |

## Evaluation — Phase 4A vs Phase 4B (full val 3,265 rows)

| Metric | Phase 4A `aggressive` | Phase 4B `aggressive-full3` (this) |
|---|---|---|
| eval_loss (full val) | **0.4724** | 0.5126 |
| eval_loss (500-row slice during training) | — | 0.5102 |
| Total samples seen | 26,126 | 24,000 |
| Epochs | 1 | 3 |

The 500-row training-time eval already pointed in this direction; the full-val
eval confirmed it. **Conclusion:** more unique examples beats more passes for
this dataset and this base model. The 3-epoch slice produced a measurably
weaker adapter despite seeing roughly the same compute budget.

See [`docs/sweep/phase4b-summary.md`](https://github.com/cmndcntrlcyber/code-trainer-offsec-pipeline/blob/main/docs/sweep/phase4b-summary.md)
for the apples-to-apples writeup.

## Limitations

* **Worse than `qwen14b-code-trainer-aggressive`** on the same val split —
  use that adapter unless you need to reproduce this experiment.
* All other limitations from the Phase 4A card carry over (no safety tuning,
  multilingual eval approximate, adapter-only).

## How to use

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen2.5-Coder-14B-Instruct"
adapter_id = "cmndcntrlcyber/qwen14b-code-trainer-aggressive-full3"

tokenizer = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForCausalLM.from_pretrained(
    base_id, dtype=torch.bfloat16, device_map="auto",
)
model = PeftModel.from_pretrained(model, adapter_id)
model.eval()
```

## Reproducibility

* **Code:** [github.com/cmndcntrlcyber/code-trainer-offsec-pipeline](https://github.com/cmndcntrlcyber/code-trainer-offsec-pipeline)
* **Launch command:**
  ```bash
  python -m src.phase4_qwen_finetuning.scripts.launch_full_training \
      --config src/config/config.yaml --best-config aggressive \
      --train-limit 8000 --val-limit 500 --wait
  ```
* **Cost:** ~$15.60 on `a100-large` (4 h 53 m).
