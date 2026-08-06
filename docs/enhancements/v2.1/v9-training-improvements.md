# V9 Training Improvements — Fix `<tool_call>` Tag Emission

## Problem

V8 model outputs the correct tool name and arguments as raw JSON, but:
1. Omits `<tool_call>` / `</tool_call>` wrapper tags that Ollama needs to parse structured `tool_calls`
2. Appends multilingual garbage tokens (Thai, Russian, Chinese) at sequence boundaries
3. Sometimes lowercases tool names (`ls` instead of `LS`)

The harness-side fallback parser (added to nexus-harness `wire.rs`) handles these
issues at runtime, but the model should emit the correct format natively.

## Root Cause Analysis

The V8 training data has `<tool_call>` tags in the content strings, but the tags
are tokenized as regular text tokens — not as special tokens. During GGUF
quantization (Q4_K_M), these multi-token sequences lose some fidelity. The model
learns the JSON payload pattern strongly but the XML tag wrapper weakly.

The multilingual garbage comes from the Qwen2.5 tokenizer's large multilingual
vocabulary. At sequence boundaries (after the tool call JSON ends), the model has
low confidence about what comes next and samples from high-frequency multilingual
tokens.

## Proposed Fixes for V9

### Fix 1: Increase tool-call tag examples density

The V8 dataset has ~12K glaive tool-calling examples out of 38K total (31%).
The `<tool_call>` tag appears once per tool-calling example. Increase density:

- Add examples with **multiple sequential tool calls** per turn (2-3 calls)
- Duplicate tool-calling slice weight to 40-50% of total dataset
- Ensure every tool-calling example has explicit `<tool_call>` and `</tool_call>` tags

### Fix 2: Add explicit stop-after-tag training

Add training examples where the assistant turn ends **immediately** after
`</tool_call>` — no trailing text. This teaches the model to stop generating
after the closing tag:

```
<|im_start|>assistant
<tool_call>
{"name": "Read", "arguments": {"file_path": "main.py"}}
</tool_call><|im_end|>
```

The V8 glaive data sometimes has trailing text after the tag which weakens the
stop signal.

### Fix 3: Add EOS enforcement via SFTConfig

Set `eos_token` explicitly in training to include `</tool_call>` as a
generation-stop trigger. In `SFTConfig`:

```python
SFTConfig(
    ...,
    dataset_text_field="text",
    packing=False,
)
```

### Fix 4: Higher learning rate on tool-call format (curriculum)

Train in two phases:
1. Phase A (80% of steps): Full mixed dataset at lr=1e-4
2. Phase B (20% of steps): Tool-calling-only subset at lr=2e-4

This gives the model a final "polish" pass on tool-call formatting.

### Fix 5: Q5_K_M instead of Q4_K_M for deployment

Q5_K_M preserves more weight fidelity and may retain the `<tool_call>` tag
pattern better. The size difference is ~1.5 GB (10.5 vs 9 GB), which fits
the RTX 5060 Ti 16GB.
