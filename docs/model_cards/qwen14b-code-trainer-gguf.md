---
base_model: Qwen/Qwen2.5-Coder-14B-Instruct
license: apache-2.0
tags:
- gguf
- llama-cpp
- quantized
- code-generation
- qwen2.5-coder
- code-trainer
pipeline_tag: text-generation
---

# qwen14b-code-trainer-gguf

GGUF quantizations of the Code-Trainer fine-tuned model. The Phase 4A LoRA
adapter [`qwen14b-code-trainer-aggressive`](https://huggingface.co/cmndcntrlcyber/qwen14b-code-trainer-aggressive)
has been merged into [`Qwen/Qwen2.5-Coder-14B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct)
and quantized via [llama.cpp](https://github.com/ggerganov/llama.cpp).

This is **Phase 5** of the
[Code-Trainer / RTPI](https://github.com/cmndcntrlcyber/code-trainer-offsec-pipeline)
pipeline. The conversion runs as an HF Job on `a100-large` — the GPU sits
idle, we use that flavor only for its 144 GB system RAM during the float16
merge step.

## Files

| File | Quantization | Size (≈) | Notes |
|---|---|---|---|
| `Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf` | Q4_K_M | ~9 GB | Recommended default — balanced quality / footprint |
| `Qwen2.5-Coder-14B-Instruct-Q5_K_M.gguf` | Q5_K_M | ~10.5 GB | Higher quality — recommended for structured output / tool calling |

Additional quantizations (Q8_0, F16) can be produced by passing
`--quants` to `launch_convert.py`.

## Intended use

* **Local inference** via `llama-cli`, `llama-server`, Ollama, LM Studio, or
  text-generation-webui.
* **Phase 6 hot-swap target** for the project's vLLM + Qwen-Agent stack —
  swapped in for compiled-language tasks alongside a smaller primary model.
* **Out of scope:** anything the upstream
  [`qwen14b-code-trainer-aggressive`](https://huggingface.co/cmndcntrlcyber/qwen14b-code-trainer-aggressive)
  card flags as out of scope (no safety tuning, no non-code tasks).

## Source

| Stage | Repo / artifact |
|---|---|
| Base model | [`Qwen/Qwen2.5-Coder-14B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct) |
| LoRA adapter | [`cmndcntrlcyber/qwen14b-code-trainer-aggressive`](https://huggingface.co/cmndcntrlcyber/qwen14b-code-trainer-aggressive) |
| Converter | `llama.cpp` (`convert_hf_to_gguf.py` + `llama-quantize`) |
| Conversion runtime | HF Job, `a100-large`, ~1 h on the merge + quantize path |

## Evaluation

Quality is inherited from the source LoRA adapter (eval_loss = 0.4724 on the
3,265-row validation split — see the
[upstream model card](https://huggingface.co/cmndcntrlcyber/qwen14b-code-trainer-aggressive#evaluation)).
Quantization to Q4_K_M typically introduces a small additional perplexity
penalty (~1 – 3 %) for 14 B coder models; we have not separately re-measured
this here because the adapter eval is the canonical signal.

## Quick start

### llama-server

```bash
llama-server \
  -m Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 8192 --n-gpu-layers 999
```

### Ollama Modelfile

```text
FROM ./Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ range .Messages }}{{ if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ .Content }}<|im_end|>
{{ else if eq .Role "tool" }}<|im_start|>tool
{{ .Content }}<|im_end|>
{{ end }}{{ end }}<|im_start|>assistant
"""
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER num_ctx 8192
```

### llama-cpp-python

```python
from llama_cpp import Llama

llm = Llama(
    model_path="Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf",
    n_ctx=8192,
    n_gpu_layers=999,
)
print(llm.create_chat_completion(messages=[
    {"role": "user", "content": "Write a Go function that reverses a UTF-8 string."},
])["choices"][0]["message"]["content"])
```

## Limitations

* **Lossy quantization.** Q4_K_M is a 4-bit-mixed format; expect minor
  degradation vs. the unquantized adapter on long-form code.
* **No safety tuning.** Inherits all caveats from the source adapter.
* **Single quant shipped.** If you need Q5_K_M / Q8_0 / F16, regenerate with
  `python -m src.phase5_deployment.scripts.launch_convert --quants Q5_K_M Q8_0`.

## Reproducibility

```bash
set -a && source .env && set +a
python -m src.phase5_deployment.scripts.launch_convert \
    --config src/config/config.yaml --wait
```

* **Code:** [github.com/cmndcntrlcyber/code-trainer-offsec-pipeline](https://github.com/cmndcntrlcyber/code-trainer-offsec-pipeline)
  (`src/phase5_deployment/`)
* **Cost:** ~$2 on `a100-large` once the job runs.
