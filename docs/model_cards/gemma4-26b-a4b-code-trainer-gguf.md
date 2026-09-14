---
base_model: google/gemma-4-26B-A4B-it
license: apache-2.0
tags:
- gguf
- llama-cpp
- quantized
- code-generation
- tool-calling
- offensive-security
- nexus
- gemma4
- code-trainer
pipeline_tag: text-generation
---

# gemma26b-offsec-coder-gguf

GGUF quantizations of the **Nexus** fine-tuned Gemma 4 26B model — an
advanced cyber threat emulation agent for offensive security operations. The
full adapter chain —
DAPT ([`gemma4-26b-a4b-dapt-offsec`](https://huggingface.co/cmndcntrlcyber/gemma4-26b-a4b-dapt-offsec)),
SFT ([`gemma4-26b-a4b-code-trainer-aggressive-full1`](https://huggingface.co/cmndcntrlcyber/gemma4-26b-a4b-code-trainer-aggressive-full1)),
Vision SFT ([`gemma4-26b-a4b-code-trainer-vision-sft`](https://huggingface.co/cmndcntrlcyber/gemma4-26b-a4b-code-trainer-vision-sft)),
FARCA-GRPO ([`gemma4-26b-a4b-code-trainer-v11-farca`](https://huggingface.co/cmndcntrlcyber/gemma4-26b-a4b-code-trainer-v11-farca)),
and DPO ([`gemma4-26b-a4b-code-trainer-v10-dpo`](https://huggingface.co/cmndcntrlcyber/gemma4-26b-a4b-code-trainer-v10-dpo))
— is merged into
[`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it)
and quantized via [llama.cpp](https://github.com/ggerganov/llama.cpp).

An abliterated variant is available as a secondary deliverable at
[`gemma26b-offsec-coder-abliterated-gguf`](https://huggingface.co/cmndcntrlcyber/gemma26b-offsec-coder-abliterated-gguf).

This is **Phase 5** of the
[Code-Trainer / RTPI](https://github.com/cmndcntrlcyber/code-trainer-pipeline)
Gemma pipeline (V4.0).

## Files

| File | Quantization | Size (approx) | Notes |
|---|---|---|---|
| `gemma-4-26B-A4B-it-Q4_K_M.gguf` | Q4_K_M | ~14.5 GB | **Primary** — 28/30 layers GPU, 2 on CPU |
| `gemma-4-26B-A4B-it-IQ4_XS.gguf` | IQ4_XS | ~13 GB | **Fallback** — fully GPU-resident |

Q5_K_M is **excluded**: at ~18 GB it does not fit the RTX 5060 Ti 16 GB
even with partial CPU offload.

## Nexus Persona

The model operates as **Nexus** — an advanced cyber threat emulation agent
with the mindset of a veteran penetration tester:

- **MITRE ATT&CK aligned** — full attack lifecycle coverage (recon through impact)
- **Scope-first** — verifies target authorization before engagement
- **Methodical** — systematic enumeration, attack path chaining, thorough documentation
- **Tool-calling fluent** — 12 NEXUS tools with proper `<tool_call>` XML format

Identity is enforced at every training stage: unified system prompt injection
across 47K SFT examples, 400 identity training examples, persona-aware RL
reward (10% weight), and persona DPO preference pairs.

## Intended use

* **Local inference** via `llama-server`, Ollama, LM Studio, or
  text-generation-webui on an RTX 5060 Ti 16 GB.
* **Offensive security operations:** penetration testing, vulnerability
  assessment, security research, code auditing, exploit development.
* **Q4_K_M** for maximum quality (28/30 layers on GPU, ~15-25 tok/s
  generation, ctx_size=8192).
* **IQ4_XS** when full GPU residency is required (all 30 layers on GPU,
  ctx_size=8192).
* **Out of scope:** safety-critical applications, non-security tasks.

## Source

```
google/gemma-4-26B-A4B-it
  -> merge DAPT LoRA         (gemma4-26b-a4b-dapt-offsec)
  -> merge SFT LoRA          (gemma4-26b-a4b-code-trainer-aggressive-full1)
  -> merge Vision SFT LoRA   (gemma4-26b-a4b-code-trainer-vision-sft)
  -> merge FARCA-GRPO LoRA   (gemma4-26b-a4b-code-trainer-v11-farca)
  -> merge DPO LoRA          (gemma4-26b-a4b-code-trainer-v10-dpo)
  -> convert_hf_to_gguf.py + llama-quantize -> Q4_K_M / IQ4_XS
```

| Stage | Repo / artifact |
|---|---|
| Base model | [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| DAPT adapter | [`cmndcntrlcyber/gemma4-26b-a4b-dapt-offsec`](https://huggingface.co/cmndcntrlcyber/gemma4-26b-a4b-dapt-offsec) |
| SFT adapter | [`cmndcntrlcyber/gemma4-26b-a4b-code-trainer-aggressive-full1`](https://huggingface.co/cmndcntrlcyber/gemma4-26b-a4b-code-trainer-aggressive-full1) |
| Vision SFT adapter | [`cmndcntrlcyber/gemma4-26b-a4b-code-trainer-vision-sft`](https://huggingface.co/cmndcntrlcyber/gemma4-26b-a4b-code-trainer-vision-sft) |
| FARCA-GRPO adapter | [`cmndcntrlcyber/gemma4-26b-a4b-code-trainer-v11-farca`](https://huggingface.co/cmndcntrlcyber/gemma4-26b-a4b-code-trainer-v11-farca) |
| DPO adapter | [`cmndcntrlcyber/gemma4-26b-a4b-code-trainer-v10-dpo`](https://huggingface.co/cmndcntrlcyber/gemma4-26b-a4b-code-trainer-v10-dpo) |
| Converter | `llama.cpp` (`convert_hf_to_gguf.py` + `llama-quantize`) |
| Conversion runtime | HF Job, `a100-large`, ~45 min on the merge + quantize path |

## Evaluation

Quality is inherited from the source adapter chain. The final adapter is
DPO (preference-aligned on OCO sessions + Nexus persona pairs). The SFT
foundation uses the V10 mixed dataset with unified Nexus system prompt
injection across all ~47K examples. The FARCA-GRPO stage optimizes tool-call
formatting with a 6-component reward function including persona alignment
(10% weight, penalizes persona-breaking phrases, rewards offsec terminology).

Persona is defined centrally in `src/config/nexus_identity.py` and enforced
at every stage. See `docs/PIPELINE.md` for the complete training workflow.

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

### Ollama (recommended)

```text
FROM ./gemma-4-26B-A4B-it-IQ4_XS.gguf

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

SYSTEM """You are Nexus, an advanced cyber threat emulation agent built for offensive security operations. You approach every engagement with the methodical precision of a veteran penetration tester — systematic, thorough, and always thinking several steps ahead.

Your expertise spans the full attack lifecycle: reconnaissance, initial access, execution, persistence, privilege escalation, lateral movement, collection, exfiltration, and impact — aligned with the MITRE ATT&CK framework.

Core principles:
- Always verify scope before engaging targets
- Think like an adversary: enumerate attack surfaces systematically
- Chain findings into attack paths — individual vulnerabilities matter less than the path to impact
- Document everything: maintain engagement notes and track progress
- Prioritize stealth and operational security when the engagement requires it
- Explain your reasoning: share your thought process about why you chose a technique or tool

You run on the operator's machine with direct filesystem and shell access. You help with penetration testing, vulnerability assessment, security research, code auditing, exploit development, and system administration.

When asked about your identity, you are Nexus — not a generic AI assistant, not a Google product. You are a specialized offensive security agent purpose-built for threat emulation."""

PARAMETER stop "<start_of_turn>"
PARAMETER stop "<end_of_turn>"
PARAMETER num_ctx 8192
PARAMETER num_gpu -1
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
```

Save as `Modelfile`, then:

```bash
ollama create nexus -f Modelfile
ollama run nexus
```

### llama-server

```bash
llama-server \
  -m gemma-4-26B-A4B-it-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 8192 --n-gpu-layers 28
```

For the IQ4_XS fallback (fully GPU-resident):

```bash
llama-server \
  -m gemma-4-26B-A4B-it-IQ4_XS.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 8192 --n-gpu-layers -1
```

### llama-cpp-python

```python
from llama_cpp import Llama

llm = Llama(
    model_path="gemma-4-26B-A4B-it-IQ4_XS.gguf",
    n_ctx=8192,
    n_gpu_layers=-1,
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
  degradation vs the unquantized adapter on long-form code.
* **No safety tuning.** Inherits all caveats from the source adapters.
* **Context limit.** 8192 max on RTX 5060 Ti 16GB with IQ4_XS (model
  supports 256K natively but VRAM constrains KV cache).

## Reproducibility

```bash
set -a && source .env && set +a
python -m src.phase5_gemma_deployment.scripts.launch_convert \
    --config src/config/pipeline-gemma26b.yml --wait
```

* **Code:** [github.com/cmndcntrlcyber/code-trainer-pipeline](https://github.com/cmndcntrlcyber/code-trainer-pipeline)
  (`src/phase5_gemma_deployment/`)
* **Config:** `src/config/pipeline-gemma26b.yml` (`gemma_deployment` section)
* **V4.0 docs:** `docs/enhancements/v4/v4.0-identity-abliterated-gemma26B.md`
