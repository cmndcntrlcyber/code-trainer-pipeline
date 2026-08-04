# Corrective Action: Code-Trainer V7 — Restore Agent Capabilities

## Problem Statement

The V6 fine-tune (`qwen14b-code-trainer-v6-aggressive`) trained on 26K code-transcription examples wiped the base model's tool-calling and instruction-following capabilities. The training data was 100% single-turn code transcription:

```
System: "You are a code transcription assistant..."
User:   "Extract the code from this screenshot."
Assistant: [verbatim source code]
```

This created three capability gaps:
1. **No tool calling** — zero training examples with function/tool calls
2. **No multi-turn reasoning** — every example is one-shot input→output
3. **No complex instruction following** — every system prompt is identical

The model now outputs "nexus" or nothing when given an agent-style system prompt with tool specifications.

## Root Cause

This is NOT a quantization issue (Q4_K_M is fine) or a catastrophic forgetting issue in the traditional sense (GSM8K actually improved). The problem is **distributional collapse** — the LoRA was trained exclusively on one narrow task pattern, and that pattern dominates the model's behavior for any input that resembles "system + user + generate".

## Corrective Action Plan

### Phase 1: Data Preparation — Build a Mixed Training Corpus

The V7 dataset must blend three capabilities so none overwrites the others:

#### Slice A: Code Generation (preserve V6 gains)
- **Source:** `cmndcntrlcyber/code-trainer-offsec-dataset` (existing)
- **Rows:** ~8K (subsample from the 26K — one pass through 26K was already sufficient per V6 findings; less data = less capability overwrite)
- **Format:** Keep existing `messages` format (system/user/assistant)
- **Purpose:** Preserve the code-generation quality you already have

#### Slice B: Tool/Function Calling (restore what V6 destroyed)
- **Source:** `NousResearch/hermes-function-calling-v1`
- **Rows:** all 11.6K across all configs (func_calling, singleturn, glaive, json_mode_agentic, json_mode_singleturn)
- **Format:** Already in Hermes tool-calling format with `<tool_call>` tags — this is the EXACT format Qwen2.5-Coder uses natively
- **Purpose:** Re-teach the model how to emit structured tool calls, parse function specs, and return JSON
- **CRITICAL:** This dataset is what NousResearch used to train Hermes 2 Pro. It's the canonical source for the `<tool_call>` format that Qwen2.5's chat template expects.

#### Slice C: Agentic Multi-Turn Coding (teach agent behavior)
- **Source:** `greghavens/fable-5-coding-and-debugging-traces`
- **Rows:** all 12.5K
- **Format:** Messages with `tool_calls` field, `reasoning_content`, multi-turn agent traces
- **Purpose:** Teach the model to reason step-by-step, call tools (Read, Write, Bash, Grep), interpret results, and iterate. These are real coding-agent sessions with verified outcomes.
- **Note:** These traces are from Claude Fable 5 — the reasoning patterns are high quality. The `messages` schema includes structured `tool_calls` with `id`, `type`, `function.name`, `function.arguments`.

#### Total: ~32K rows (balanced mix)

### Phase 2: Format Alignment

All three slices must be converted to a **unified chat format** compatible with the Qwen2.5 ChatML template:

```
<|im_start|>system
{system prompt}

# Tools
{tool definitions in <tools> XML tags, if applicable}
<|im_end|>
<|im_start|>user
{user message}<|im_end|>
<|im_start|>assistant
{response, possibly with <tool_call> tags}<|im_end|>
```

For the Fable 5 traces (Slice C), the OpenAI-format `tool_calls` need to be converted to Hermes `<tool_call>` XML format to match Qwen2.5's native template.

Write a preprocessing script (Phase 3 of the RTPI pipeline) that:
1. Loads all three slices
2. Converts to unified ChatML format
3. Wraps tool definitions in `<tools>` tags per Qwen2.5 template
4. Converts tool calls to `<tool_call>` XML format
5. Shuffles the combined dataset
6. Splits into train/validation (90/10)
7. Pushes to HuggingFace as `cmndcntrlcyber/code-trainer-v7-mixed`

### Phase 3: Training Configuration

```yaml
# V7 training config — key differences from V6
base_model: Qwen/Qwen2.5-Coder-14B-Instruct
adapter: lora
lora_r: 32          # LOWER than V6's 64 — less aggressive = less forgetting
lora_alpha: 64      # Keep 2:1 ratio
lora_dropout: 0.05
learning_rate: 1.5e-4  # LOWER than V6's 3e-4 — gentler on existing capabilities
epochs: 1
max_seq_length: 8192   # UP from V6's 2048 — agent prompts are long
batch_size: 2          # Adjusted for longer sequences
gradient_accumulation: 8  # Effective batch = 16 (same as V6)
warmup_ratio: 0.05
lr_scheduler: cosine
bf16: true
gradient_checkpointing: true
```

Key changes from V6:
- **`lora_r: 32`** (was 64) — less aggressive LoRA means less overwriting of existing weights. V6 went aggressive because it was training one narrow task; V7 needs to preserve multiple capabilities.
- **`learning_rate: 1.5e-4`** (was 3e-4) — gentler learning to avoid overwriting tool-calling circuits
- **`max_seq_length: 8192`** (was 2048) — agent system prompts + tool specs + multi-turn conversations need more context. The Fable 5 traces can be 4-8K tokens.

### Phase 4: Validation

Must test ALL THREE capabilities after training:

1. **Code generation** (preserve V6):
   - Run eval_loss on the existing V6 validation split
   - Target: eval_loss < 0.50 (V6 achieved 0.4724)

2. **Tool calling** (restored):
   - Present the model with a system prompt containing `<tools>` definitions
   - Verify it emits proper `<tool_call>` XML tags
   - Test: 10 diverse tool-calling scenarios → ≥80% valid tool calls

3. **Agent behavior** (new):
   - Give the model a multi-turn agent scenario (read file → analyze → modify)
   - Verify it reasons, calls tools, interprets results, and iterates
   - Test: 5 multi-step coding tasks → model makes progress on ≥3

4. **GSM8K** (forgetting check):
   - Target: flexible-extract ≥ 0.60 (V6 achieved 0.6778, base was 0.6050)

### Phase 5: GGUF Conversion

Same pipeline as V6 Phase 5, but:
- Ship both Q4_K_M (default) and Q5_K_M (for more reliable structured output)
- The Ollama Modelfile MUST include the full Qwen2.5 tool-calling template — V6's Modelfile omitted tool-calling tokens entirely

Updated Modelfile:
```
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

---

## Dataset Evaluation for Post-SFT RL

After the corrective SFT above, these datasets could be used for GRPO/DPO/RLHF to further reinforce reasoning:

### Recommended (high value for your use case)

| Dataset | Rows | Why | Use for |
|---------|------|-----|---------|
| **NousResearch/hermes-function-calling-v1** | 11.6K | The canonical tool-calling training set. Hermes `<tool_call>` format matches Qwen2.5 natively. Includes single-turn, multi-turn, agentic JSON mode. | **SFT (Phase 1, Slice B)** — not RL. This is foundational data, not reward signal. |
| **greghavens/fable-5-coding-and-debugging-traces** | 12.5K | Verified coding-agent trajectories with structured tool_calls, reasoning_content. Multi-turn. Categories include coding, debugging, instruction-following. | **SFT (Phase 1, Slice C)** and/or **RL reward signal** — successful trajectories = positive reward. |
| **Glint-Research/Fable-5-traces** | 4.7K | Pi-agent format with 3,799 tool calls across 60 source sessions. Real agent sessions, not synthetic. | **GRPO/DPO after SFT** — use successful tool-use sequences as preferred, tool-less or failed attempts as dispreferred. |

### Useful with caveats

| Dataset | Rows | Why | Caveat |
|---------|------|-----|--------|
| **Qyrou/reasoning-corpus-4K-5M-v1** | ~3.7M | Massive reasoning corpus with thought traces from DeepSeek-v4, Qwen3, Gemma4. Has `thought_trace` + `assistant` separation. | **No tool calling.** Reinforces general reasoning/CoT but won't teach tool use. Subsample 5-10K rows focused on code/logic topics. Useful for GRPO reward modeling on reasoning quality. |
| **Jackrong/Kimi-K2.5-Reasoning-1M-Cleaned** | ~458K | High-quality STEM/math/PhD-level reasoning from KIMI K2.5. Four domains: General, Math, MultilingualSTEM, PHD-Science. | **No tool calling, no code focus.** Useful if you want the model to reason better about scientific/analytical topics (relevant to your Analysis domain). Subsample ~5K from General-Distillation config. |

### Skip

| Dataset | Rows | Why skip |
|---------|------|----------|
| **Roman1111111/gpt-5.4-step-by-step-reasoning** | 1.5K | Too small (1.5K rows), no tool calling, no code focus, synthetic "ultra-logic" puzzles. The reasoning style (pure math/logic) doesn't transfer to agent behavior. The other reasoning datasets are strictly better. |

### Recommended RL Pipeline

```
Phase 1: Corrective SFT (this plan)
  ├── Slice A: code-trainer-offsec-dataset (8K subsample)
  ├── Slice B: hermes-function-calling-v1 (11.6K)
  └── Slice C: fable-5-coding-and-debugging-traces (12.5K)
          ↓
Phase 2: GRPO on tool-use trajectories
  ├── Positive reward: Fable-5-traces (successful tool calls → correct result)
  ├── Negative reward: failed/empty tool calls, hallucinated tools
  └── Verifier: execute tool calls against a sandbox, check output
          ↓
Phase 3: DPO on reasoning quality (optional)
  ├── Preferred: Qyrou reasoning-corpus subsample (clear thought traces)
  ├── Dispreferred: model's own outputs on same prompts (self-play)
  └── Focus: code-relevant reasoning tasks only
```

**Key principle:** SFT first to restore the capability surface, THEN RL to refine quality within that surface. RL cannot teach a capability the model has never seen — it can only sharpen one that exists.
