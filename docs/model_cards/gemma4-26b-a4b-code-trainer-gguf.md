---
base_model: google/gemma-4-26B-A4B-it
license: apache-2.0
tags:
- gguf
- llama-cpp
- quantized
- code-generation
- tool-calling
- gemma4
- code-trainer
pipeline_tag: text-generation
---

# gemma4-26b-a4b-code-trainer-gguf

GGUF quantizations of the Code-Trainer fine-tuned Gemma 4 26B model. The full
adapter chain —
DAPT ([`gemma4-26b-a4b-dapt-offsec`](https://huggingface.co/cmndcntrlcyber/gemma4-26b-a4b-dapt-offsec)),
SFT ([`gemma4-26b-a4b-code-trainer-aggressive-full1`](https://huggingface.co/cmndcntrlcyber/gemma4-26b-a4b-code-trainer-aggressive-full1)),
FARCA-GRPO ([`gemma4-26b-a4b-code-trainer-v11-farca`](https://huggingface.co/cmndcntrlcyber/gemma4-26b-a4b-code-trainer-v11-farca)),
and DPO ([`gemma4-26b-a4b-code-trainer-v10-dpo`](https://huggingface.co/cmndcntrlcyber/gemma4-26b-a4b-code-trainer-v10-dpo))
— is merged into
[`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it)
and quantized via [llama.cpp](https://github.com/ggerganov/llama.cpp).

This is **Phase 5** of the
[Code-Trainer / RTPI](https://github.com/cmndcntrlcyber/code-trainer-pipeline)
Gemma pipeline. The conversion runs as an HF Job on `a100-large` — the GPU sits
idle, we use that flavor only for its 144 GB system RAM during the ~52 GB
BF16 merge step.

## Files

| File | Quantization | Size (approx) | Notes |
|---|---|---|---|
| `gemma-4-26B-A4B-it-Q4_K_M.gguf` | Q4_K_M | ~14.5 GB | **Primary** — 28/30 layers GPU, 2 on CPU |
| `gemma-4-26B-A4B-it-IQ4_XS.gguf` | IQ4_XS | ~13 GB | **Fallback** — fully GPU-resident |

Q5_K_M is **excluded**: at ~18 GB it does not fit the RTX 5060 Ti 16 GB
even with partial CPU offload.

## Intended use

* **Local inference** via `llama-server`, Ollama, LM Studio, or
  text-generation-webui on an RTX 5060 Ti 16 GB.
* **Q4_K_M** for maximum quality (28/30 layers on GPU, ~15-25 tok/s
  generation, ctx_size=4096).
* **IQ4_XS** when full GPU residency is required (all 30 layers on GPU,
  ctx_size=4096).
* **Out of scope:** anything the upstream adapter cards flag as out of scope
  (no safety tuning, no non-code tasks).

## Source

The GGUF is produced by merging the full adapter chain in order, then
quantizing the merged model:

```
google/gemma-4-26B-A4B-it
  -> merge DAPT LoRA         (gemma4-26b-a4b-dapt-offsec)
  -> merge SFT LoRA          (gemma4-26b-a4b-code-trainer-aggressive-full1)
  -> merge FARCA-GRPO LoRA   (gemma4-26b-a4b-code-trainer-v11-farca)
  -> merge DPO LoRA          (gemma4-26b-a4b-code-trainer-v10-dpo)
  -> convert_hf_to_gguf.py + llama-quantize -> Q4_K_M / IQ4_XS
```

| Stage | Repo / artifact |
|---|---|
| Base model | [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| DAPT adapter | [`cmndcntrlcyber/gemma4-26b-a4b-dapt-offsec`](https://huggingface.co/cmndcntrlcyber/gemma4-26b-a4b-dapt-offsec) |
| SFT adapter | [`cmndcntrlcyber/gemma4-26b-a4b-code-trainer-aggressive-full1`](https://huggingface.co/cmndcntrlcyber/gemma4-26b-a4b-code-trainer-aggressive-full1) |
| FARCA-GRPO adapter | [`cmndcntrlcyber/gemma4-26b-a4b-code-trainer-v11-farca`](https://huggingface.co/cmndcntrlcyber/gemma4-26b-a4b-code-trainer-v11-farca) |
| DPO adapter | [`cmndcntrlcyber/gemma4-26b-a4b-code-trainer-v10-dpo`](https://huggingface.co/cmndcntrlcyber/gemma4-26b-a4b-code-trainer-v10-dpo) |
| Converter | `llama.cpp` (`convert_hf_to_gguf.py` + `llama-quantize`) |
| Conversion runtime | HF Job, `a100-large`, ~45 min on the merge + quantize path |

## Evaluation

Quality is inherited from the source adapter chain. The final adapter is
DPO (preference-aligned on 870 pairs from real OCO sessions). The SFT
foundation uses the V9 mixed dataset. The FARCA-GRPO stage optimizes
tool-call formatting with factuality grounding.

## Chat template

Gemma 4 uses `<start_of_turn>` / `<end_of_turn>` tags (not ChatML
`<|im_start|>` / `<|im_end|>`):

```
<start_of_turn>user
Write a Python nmap scanner wrapper.<end_of_turn>
<start_of_turn>model
...response...<end_of_turn>
```

## Quick start

### llama-server

```bash
llama-server \
  -m gemma-4-26B-A4B-it-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 4096 --n-gpu-layers 28
```

For the IQ4_XS fallback (fully GPU-resident):

```bash
llama-server \
  -m gemma-4-26B-A4B-it-IQ4_XS.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 4096 --n-gpu-layers -1
```

### Ollama Modelfile

```text
FROM ./gemma-4-26B-A4B-it-Q4_K_M.gguf
TEMPLATE """{{ if .System }}<start_of_turn>user
{{ .System }}<end_of_turn>
{{ end }}{{ range .Messages }}{{ if eq .Role "user" }}<start_of_turn>user
{{ .Content }}<end_of_turn>
{{ else if eq .Role "assistant" }}<start_of_turn>model
{{ .Content }}<end_of_turn>
{{ else if eq .Role "tool" }}<start_of_turn>tool
{{ .Content }}<end_of_turn>
{{ end }}{{ end }}<start_of_turn>model
"""
PARAMETER stop "<start_of_turn>"
PARAMETER stop "<end_of_turn>"
PARAMETER num_ctx 4096
PARAMETER num_gpu 28
```

### llama-cpp-python

```python
from llama_cpp import Llama

llm = Llama(
    model_path="gemma-4-26B-A4B-it-Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=28,
)
print(llm.create_chat_completion(messages=[
    {"role": "user", "content": "Write a Go function that performs a TCP port scan."},
])["choices"][0]["message"]["content"])
```

## Limitations

* **No Q5_K_M.** Unlike the Qwen pipeline, Q5_K_M (~18 GB) does not fit the
  RTX 5060 Ti 16 GB target. Q4_K_M is the primary quant.
* **Partial CPU offload.** Q4_K_M requires 2/30 layers on CPU — expect
  slight latency vs fully GPU-resident IQ4_XS.
* **Lossy quantization.** Q4_K_M is a 4-bit-mixed format; expect minor
  degradation vs the unquantized adapter on long-form code. IQ4_XS trades
  additional quality for smaller footprint.
* **No safety tuning.** Inherits all caveats from the source adapters.

## Reproducibility

```bash
set -a && source .env && set +a
python -m src.phase5_gemma_deployment.scripts.launch_convert \
    --config src/config/pipeline-gemma26b.yml --wait
```

* **Code:** [github.com/cmndcntrlcyber/code-trainer-pipeline](https://github.com/cmndcntrlcyber/code-trainer-pipeline)
  (`src/phase5_gemma_deployment/`)
* **Config:** `src/config/pipeline-gemma26b.yml` (`gemma_deployment` section)
