# Plan: Maximize Reasoning Speed & Effectiveness, Reduce Time-to-Correct-Action

## Context

The Nexus RL pipeline (Phase 4c) trains tool-use and reasoning through GRPO, FARCA-GRPO, and DPO. Currently the reward function scores **formatting** — valid JSON, legal tool names, tag structure — but gives almost no signal on **reasoning quality, conciseness, or tool-selection judgment**:

- `has_reasoning_prefix` (0.15 weight) passes for any 10+ characters — "Let me do it" gets full credit
- No incentive for concise reasoning — 500 tokens of verbose monologue scores the same as 50 tokens of precise analysis
- No reward for choosing the *right* tool — `Bash` always works and gets full marks even when `Read` or `ScopeCheck` would be correct
- The system prompt says "explain your reasoning" but prescribes no format
- At inference on the RTX 5060 Ti (~15-25 tok/s), every unnecessary token costs 40-65ms

The v3.1 enhancement doc (`docs/enhancements/v3.0/v3.1/`) designed a tool-selection reward component but it was never built. This plan implements that plus five more changes, ordered by impact-to-effort ratio.

---

## Changes (Priority Order)

### P0: Tool-Task Alignment Reward — Training
**Impact: HIGH | Effort: LOW**

Implement the v3.1 design. Add `_tool_selection_quality()` to the reward function with three sub-signals.

**Modify** `src/phase4c_rl/rewards/tool_call_reward.py`:
- Add `_scope_before_action(response, prompt)` — 0.0 if new IP/domain engaged without prior ScopeCheck, 1.0 otherwise
- Add `_tool_task_alignment(response, prompt)` — 1.0 for best tool, 0.5 for Bash (universal fallback), 0.0 for clearly wrong tool. Prompt keyword → expected tool mapping:
  - "read/check file" + path → Read
  - "search/find/grep" → Grep, Glob
  - "list" + dir → LS, Glob
  - "edit/change/modify" → Edit
  - URL present → WebFetch
  - "scan/enumerate/run" → Bash
- Add `_no_redundancy(response)` — 0.0 if duplicate tool+target pairs detected
- Composite: `0.4 * scope + 0.4 * alignment + 0.2 * redundancy`
- Weight: 0.20

**Modify** `tool_call_reward()` signature — add `prompts: list[str] | None = None` parameter so it can see the user's prompt. Update callers:
- `src/phase4c_rl/hf_skills/grpo_entry.py:179` — pass `prompts` through
- `src/phase4c_rl/hf_skills/farca_grpo_entry.py:181` — same

**Modify** `src/phase4c_rl/data/build_grpo_prompts.py` — add 150 tool-selection exercise prompts with clear "right tool" answers.

### P1: Structured Reasoning Format — Training + Inference
**Impact: HIGH | Effort: LOW**

Replace the generic "explain your reasoning" bullet in the system prompt with a structured Assess → Select → Act pattern.

**Modify** `src/config/nexus_identity.py` `NEXUS_IDENTITY` string — replace the line:
```
"- Explain your reasoning: share your thought process about why you chose a technique or tool\n\n"
```
with:
```
"- Before each tool call, state your reasoning in 1-3 concise sentences:\n"
"  1. ASSESS: What does the situation require?\n"
"  2. SELECT: Which tool fits and why?\n"
"  3. ACT: Invoke the tool.\n"
"  Keep reasoning tight — one sentence per step is ideal. "
"Do not repeat the prompt or explain what tools do.\n\n"
```

This is intentionally lightweight — not `<think>` tags (too verbose, 200-500 extra tokens per call) but a natural-language pattern that targets 30-80 tokens of reasoning.

### P2: Reasoning Efficiency Reward — Training
**Impact: HIGH | Effort: MEDIUM**

Add a soft conciseness bonus. NOT a hard length penalty (would kill reasoning), but a gradient: 1.0 for <100 chars, linear decay to 0.5 at 400+ chars. Floor at 0.5 prevents reward-hacking toward zero reasoning.

**Modify** `src/phase4c_rl/rewards/tool_call_reward.py`:
- Add `_reasoning_efficiency(response)` — gated on having a valid tool call (no credit without correctness)
- Weight: 0.10

**Final weight table after P0+P1+P2:**

| Component | Weight |
|---|---|
| has_valid_tool_call_tags | 0.20 |
| tool_name_in_schema | 0.15 |
| has_reasoning_prefix | 0.10 |
| no_hallucinated_tools | 0.10 |
| ends_cleanly_after_tag | 0.10 |
| persona_aligned_reasoning | 0.05 |
| tool_selection_quality | 0.20 |
| reasoning_efficiency | 0.10 |

### P3: Constrained Decoding at Inference — Inference Only
**Impact: MEDIUM | Effort: LOW**

Add a GBNF grammar to force valid JSON after `<tool_call>`. Eliminates wasted tokens on malformed JSON.

**Create** `nexus-harness/crates/nexus-agent/grammars/tool_call.gbnf` — constrains JSON body with tool names enumerated from NEXUS_TOOLS_V10.

**Modify** `nexus-harness/crates/nexus-agent/src/lib.rs` — on malformed tool-call JSON, retry with grammar constraint using the existing `ChatOptions.grammar` field. The grammar support is already plumbed through `nexus-core/src/provider.rs` and `nexus-llm/src/wire.rs`.

### P4: Reasoning-Action Coherence in FARCA — Training
**Impact: MEDIUM | Effort: MEDIUM**

The FARCA pipeline verifies factual claims but not whether reasoning *supports* the tool choice. "I should read the file" → calls Bash gets full factual credit.

**Modify** `src/phase4c_rl/farca/claim_extractor.py` — add claim type `"tool_choice_justification"` for claims containing both a tool name and justification language ("should", "need to", "will use", "because").

**Modify** `src/phase4c_rl/farca/claim_verifier.py` — add `_verify_tool_choice_justification(claim, evidence, actual_tool)`: 1.0 if mentioned tool matches actual first tool called, 0.0 if mismatch.

**Modify** `src/phase4c_rl/farca/config.py` — add `coherence_weight: float = 0.3` and `coherence_enabled: bool = True`.

**Modify** `src/phase4c_rl/farca/pipeline.py` — extract actual first tool name from completion, pass to verifier for justification claims. Blend: `(1 - coherence_weight) * existing + coherence_weight * coherence`.

### P5: DPO Conciseness Pairs — Training (Data)
**Impact: MEDIUM | Effort: HIGH**

Add preference pairs where chosen=concise correct reasoning, rejected=verbose correct reasoning with the same tool call.

**Modify** `src/phase4c_rl/data/build_dpo_pairs.py` — add `build_conciseness_pairs()`. Take high-quality OCO positives with <150 char reasoning + correct tool call, generate verbose variants by prepending filler, inserting tool-description restatements, adding unnecessary caveats. Rule-based expansion (no GPU). Target 150-200 pairs (~15-20% of DPO dataset).

---

## Sequencing

```
Phase 1 (SFT):     P1 lands first — system prompt change propagates to all data builders
Phase 2 (GRPO):    P0 + P2 land together — single weight rebalance commit
Phase 2b (FARCA):  P4 lands — gated behind coherence_enabled flag for safety
Phase 3 (DPO):     P5 lands — refines what GRPO/FARCA started
Independent:        P3 ships anytime — inference only, zero training interaction
```

## Verification

1. **Unit tests**: New test file `src/phase4c_rl/tests/test_tool_selection_reward.py` covering all new reward components with boundary cases
2. **Reward distribution sanity**: Run `tool_call_reward_detailed()` on 50 existing completions — new components should distribute meaningfully (not all 0.0 or all 1.0)
3. **GRPO dry run**: 50 prompts × 10 generations with updated reward, verify mean reward recovers to >0.7 within 200 steps
4. **Eval gates unchanged**: The 80% tool-call-eval pass rate and 6/10 agent-eval gates are independent of reward values
5. **Inference benchmark**: After P3 (grammar), measure JSON parse success rate and tokens-per-tool-call vs. baseline
6. **Wandb metrics**: Monitor `reward/tool_selection_quality_mean`, `reward/reasoning_efficiency_mean`, `farca/coherence_score_mean` across training runs
