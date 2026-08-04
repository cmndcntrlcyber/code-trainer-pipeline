# Phase 4B — Full Training Summary

**Adapter:** [cmndcntrlcyber/qwen14b-code-trainer-aggressive-full3](https://hf.co/cmndcntrlcyber/qwen14b-code-trainer-aggressive-full3) (1.05 GB)
**Job:** [`69f8188e9d85bec4d76f1c5e`](https://huggingface.co/jobs/cmndcntrlcyber/69f8188e9d85bec4d76f1c5e) — A100-large, 4h 53m, COMPLETED
**Generated:** 2026-05-04
**Raw JSON:** [`phase4b-result.json`](./phase4b-result.json)

## Configuration

| Knob | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-14B-Instruct` |
| LoRA r / α | 64 / 128 |
| Learning rate | 3e-4 |
| Batch size × accum | 4 × 4 (eff_batch = 16) |
| Epochs | 3 |
| Train rows | 8,000 (sliced from 26,126) |
| Val rows | 500 (sliced from 3,265) |
| Precision | BF16 + gradient checkpointing |

## Result

```json
{
  "num_epochs": 3,
  "eval_loss": 0.5101475119590759,
  "eval_runtime": 102.88
}
```

## Phase 4A vs Phase 4B — apples-to-apples (full val split)

| Metric | Phase 4A `aggressive` (1 ep × full 26k) | Phase 4B `aggressive-full3` (3 ep × 8k) |
|---|---|---|
| **eval_loss (full val 3265)** | **0.4724** | 0.5126 |
| eval_loss (500-row slice) | — | 0.5102 |
| Total samples seen | 26,126 | 24,000 |
| Job runtime | 6h 39m (timeout) | 4h 53m |

**Phase 4A wins by 0.04** on the same 3,265-row validation set (eval-only A100 job
[`69f89e5b9d85bec4d76f217e`](https://huggingface.co/jobs/cmndcntrlcyber/69f89e5b9d85bec4d76f217e),
14m 37s, ~$0.80).

The 500-row slice in the Phase 4B training job wasn't a sampling fluke — the full-val
number is even slightly worse (0.5126 vs 0.5102). The overfitting hypothesis is
confirmed: 3 passes over 8k samples produced a measurably weaker adapter than 1 pass
over the full 26k.

**Decision: Phase 5 uses Phase 4A's `aggressive` adapter** at
[`cmndcntrlcyber/qwen14b-code-trainer-aggressive`](https://hf.co/cmndcntrlcyber/qwen14b-code-trainer-aggressive),
not `aggressive-full3`.

**Lesson for any rerun:** if we want more passes through the data, slice less
aggressively — `--train-limit 16000-20000` × 2 epochs would cover similar compute
while seeing more unique examples.

## Cost log

| Phase | Run | Hours | Cost |
|---|---|---|---|
| 4A | conservative | 7.0 (timeout) | ~$22.40 |
| 4A | standard | 6.7 (timeout) | ~$21.30 |
| 4A | aggressive | 7.0 (timeout) | ~$22.40 |
| 4B | aggressive-full3 | 4.9 | ~$15.60 |
| **Phase 4 total** | | | **~$81.70** |

(All Phase 4A runs marked ERROR by HF Skills' lazy timeout but pushed adapters successfully before the kill — see `phase4a-summary.md`.)

## Status

- [x] Phase 4B trained checkpoint published (1.05 GB LoRA adapter)
- [x] Comparable eval_loss against full val split — confirmed Phase 4A is better
- [ ] **Phase 5 GGUF conversion uses `cmndcntrlcyber/qwen14b-code-trainer-aggressive`** (Phase 4A) — merge LoRA → base, quantize to Q4_K_M
