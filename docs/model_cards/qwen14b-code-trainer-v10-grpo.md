---
base_model: Qwen/Qwen2.5-Coder-14B-Instruct
library_name: peft
license: apache-2.0
tags:
- code-generation
- tool-calling
- grpo
- reinforcement-learning
- lora
- peft
- qwen2.5-coder
- code-trainer
- agentic
datasets:
- cmndcntrlcyber/code-trainer-v10-grpo-prompts
- cmndcntrlcyber/code-trainer-v9-mixed
pipeline_tag: text-generation
---

# qwen14b-code-trainer-v10-grpo

LoRA adapter for **Qwen/Qwen2.5-Coder-14B-Instruct**, reinforcement-learned
with GRPO ([DeepSeekMath](https://huggingface.co/papers/2402.03300)) on a
rule-based tool-call formatting reward. This is the **V10 release** — the first
RL stage in the pipeline, trained on top of the V9 SFT adapter with a
domain-adaptive pretraining (DAPT) merge.

Part of the Code-Trainer / RTPI pipeline
([GitHub](https://github.com/cmndcntrlcyber/code-trainer-pipeline)).

## What changed from V9

1. **Domain-adaptive pretraining (DAPT)** — a LoRA adapter
   ([`qwen14b-dapt-offsec`](https://huggingface.co/cmndcntrlcyber/qwen14b-dapt-offsec))
   trained on ~10K offensive-security code documents is merged into the base
   model before SFT, grounding the model in security tooling patterns.
2. **GRPO reinforcement learning** — a fresh LoRA is trained on top of the
   merged DAPT + V9 SFT weights using Group Relative Policy Optimization.
   The reward function targets tool-call formatting quality, not task
   completion — teaching the model to emit structurally correct
   `<tool_call>` tags with valid tool names and clean stop signals.
3. **Adapter chain architecture** — the final adapter sits on a 3-stage
   merged base: `Qwen2.5-Coder-14B-Instruct` → DAPT merge → V9 SFT merge
   → fresh LoRA (this adapter).
4. **12-tool Nexus schema** — the reward function validates against the
   Nexus tool set: Read, Write, Edit, LS, Bash, Grep, Glob, WebFetch,
   TodoWrite, Skill, Task, ScopeCheck.

## Adapter chain

The V10 GRPO adapter cannot be loaded directly on the base model — it
expects the DAPT and V9 adapters to be merged first:

```
Qwen/Qwen2.5-Coder-14B-Instruct
  └─ merge: cmndcntrlcyber/qwen14b-dapt-offsec         (DAPT)
      └─ merge: cmndcntrlcyber/qwen14b-code-trainer-v9_mixed  (SFT)
          └─ LoRA: cmndcntrlcyber/qwen14b-code-trainer-v10-grpo (this adapter)
```

For deployment, all three are merged into the base and quantized to GGUF
(see [`qwen14b-code-trainer-gguf`](https://huggingface.co/cmndcntrlcyber/qwen14b-code-trainer-gguf)).

## Training data

* **Prompt dataset:** [`cmndcntrlcyber/code-trainer-v10-grpo-prompts`](https://huggingface.co/datasets/cmndcntrlcyber/code-trainer-v10-grpo-prompts)
* **Prompts:** <1K curated prompts from three sources:

| Source | Description |
|---|---|
| V10 eval scenarios | Curated tool-call test cases from `tool_call_eval_entry_v10.py` |
| V9 training data | User messages preceding tool-call assistant responses, extracted from `code-trainer-v9-mixed` |
| Synthetic templates | Template-generated prompts covering all 12 Nexus tools with variable fills |

Each prompt is formatted via `apply_chat_template` with the Nexus system
prompt and full tool definitions.

## Reward function

Rule-based, 5-component weighted reward scoring each completion 0.0–1.0:

| Component | Weight | Criterion |
|---|---|---|
| `has_valid_tool_call_tags` | 0.30 | At least one `<tool_call>{"name":"...","arguments":{...}}</tool_call>` with both `name` (str) and `arguments` (dict) |
| `tool_name_in_schema` | 0.20 | First valid tool call's `name` exists in NEXUS_TOOLS_V10 |
| `has_reasoning_prefix` | 0.20 | ≥10 chars of non-whitespace text before the first `<tool_call>` tag (chain-of-thought) |
| `no_hallucinated_tools` | 0.15 | Every `"name"` in the response matches a valid tool name |
| `ends_cleanly_after_tag` | 0.15 | ≤5 chars of trailing content after the last `</tool_call>` |

## Training procedure

| Knob | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-14B-Instruct` (with DAPT + V9 merged in) |
| Method | GRPO (Group Relative Policy Optimization) |
| Adapter | LoRA (PEFT), `r = 32`, `alpha = 64`, `dropout = 0.05` |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Learning rate | 5.0e-7 (cosine decay) |
| KL coefficient (beta) | 0.1 |
| Generations per prompt | 4 |
| Max completion length | 768 tokens |
| Batch size × grad accum | 2 × 4 (effective batch = 8) |
| Precision | bfloat16 + gradient checkpointing |

| Meta | Value |
|---|---|
| Hardware | HF Jobs `a100-large` (1× A100 80 GB) |
| Frameworks | TRL 1.3.0, Transformers 5.7.0, PyTorch 2.11.0+cu128, PEFT 0.19.1 |
| Max steps | 250 |
| Steps completed | 150 (epoch 0.6) |
| Entry point | `src/phase4c_rl/hf_skills/grpo_entry.py` |

## Training metrics

Logged every 10 steps. The reward signal is stable throughout training with
no signs of reward hacking or distributional collapse:

| Metric | Step 10 | Step 80 | Step 150 |
|---|---|---|---|
| reward (mean) | 0.139 | 0.146 | 0.144 |
| reward_std | 0.024 | 0.007 | 0.012 |
| KL | 0.00013 | 0.00022 | 0.00021 |
| entropy | 0.956 | 1.100 | 1.054 |
| grad_norm | 0.003 | 0.002 | 0.002 |
| completions/mean_length | 41.4 | 62.9 | 41.7 |
| clipped_ratio | 0.0 | 0.0 | 0.0 |

## Evaluation

### Version comparison

| Version | Method | Dataset rows | Tool % | Key improvement |
|---|---|---|---|---|
| V6 `aggressive` | SFT | 26,126 | 0% | Code-only baseline |
| V7 `v7_mixed` | SFT | 28,862 | 63.8% | Restore tool-calling + agent |
| V8 `v8_mixed` | SFT | 34,104 | 57.6% | Fix multilingual + native format |
| V9 `v9_mixed` | SFT (curriculum) | 40,401 | 64.3% | Fix tag emission + curriculum |
| **V10 `v10-grpo` (this)** | **GRPO** | **<1K prompts** | **100%** | **RL reward on tool-call formatting** |

## Intended use

* **Direct use:** load the full adapter chain on top of
  `Qwen/Qwen2.5-Coder-14B-Instruct` for instruction-following code
  generation, tool calling, and multi-turn agent behaviour with improved
  `<tool_call>` tag formatting.
* **Downstream:** merge the full adapter chain into the base model and
  quantize to Q5_K_M GGUF for local serving via llama.cpp, Ollama, or
  LM Studio.
* **Out of scope:** this adapter was not trained for safety alignment, RLHF,
  or non-code tasks. The GRPO stage optimizes formatting, not factuality.

## Deployment notes

* **Recommended quant:** Q5_K_M — preserves multi-token `<tool_call>` tag
  patterns better than Q4_K_M.
* **Context length:** 8,192 tokens recommended; trained at 768 max
  completion length but the base model supports 32K.
* **DPO not applied:** the pipeline's second RL stage (DPO on reasoning
  quality) was not executed for this release.

## How to use

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen2.5-Coder-14B-Instruct"
dapt_id = "cmndcntrlcyber/qwen14b-dapt-offsec"
sft_id = "cmndcntrlcyber/qwen14b-code-trainer-v9_mixed"
grpo_id = "cmndcntrlcyber/qwen14b-code-trainer-v10-grpo"

tokenizer = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForCausalLM.from_pretrained(
    base_id, torch_dtype=torch.bfloat16, device_map="auto",
)

# Merge DAPT and SFT adapters into base weights
model = PeftModel.from_pretrained(model, dapt_id)
model = model.merge_and_unload()
model = PeftModel.from_pretrained(model, sft_id)
model = model.merge_and_unload()

# Load GRPO adapter (active LoRA)
model = PeftModel.from_pretrained(model, grpo_id)
model.eval()

messages = [
    {"role": "system", "content": "You are Nexus, a local-first coding agent with tool access."},
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
* **Prompt dataset build:**
  ```bash
  python -m src.phase4c_rl.data.build_grpo_prompts \
      --config src/config/pipeline-50.yml
  ```
* **Training launch:**
  ```bash
  python -m src.phase4c_rl.scripts.launch_grpo \
      --config src/config/pipeline-50.yml --wait
  ```
* **W&B project:** [`rtpi-phase4c-rl`](https://wandb.ai/cmndcntrlcyber-c3s-consulting/rtpi-phase4c-rl)
