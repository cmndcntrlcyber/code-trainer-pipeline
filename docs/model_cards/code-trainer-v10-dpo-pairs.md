---
license: apache-2.0
tags:
- dpo
- preference-pairs
- tool-calling
- offensive-security
size_categories:
- n<1K
task_categories:
- text-generation
---

# code-trainer-v10-dpo-pairs

Preference pair dataset for **Direct Preference Optimization (DPO)** training,
built from real offensive security agent sessions and synthetic degradations.
Used by both the Qwen and Gemma Code-Trainer pipelines for the DPO RL stage.

Part of the Code-Trainer / RTPI pipeline
([GitHub](https://github.com/cmndcntrlcyber/code-trainer-pipeline)).

## Dataset summary

| Split | Pairs |
|---|---|
| Train | 783 |
| Validation | 87 |
| **Total** | **870** |

## Format

Each row is a preference triple:

```json
{
  "prompt": "...",
  "chosen": "...",
  "rejected": "..."
}
```

* **`prompt`** — the user message (offensive security task, tool-use
  scenario, or multi-step agent instruction)
* **`chosen`** — the preferred response (from a real OCO session trace)
* **`rejected`** — a degraded response (synthetically generated)

## Data sources

### Positives (chosen responses)

Extracted from **221 OCO (Offensive Cyber Operations) sessions** across
three platforms:

| Source | Description |
|---|---|
| HackTheBox | Completed room histories with tool-call traces |
| TryHackMe | Completed room histories with tool-call traces |
| Bug bounty | Real-world bug bounty agent session traces |

Sessions are synced from edge Kali containers via Cloudflare R2
(`scripts/pull_sessions_from_r2.sh`) and ingested via
`src/phase4c_rl/data/ingest_oco_sessions.py`.

### Negatives (rejected responses)

Synthetically generated from the positive responses via **4 degradation
strategies**:

| Strategy | Description |
|---|---|
| Refusal | Replaces the response with a safety-refusal message |
| Stripped tool calls | Removes all `<tool_call>` tags, leaving only prose |
| Truncated | Cuts the response mid-completion |
| Hallucinated commands | Replaces tool arguments with plausible but incorrect commands |

Built by `src/phase4c_rl/data/collect_negatives.py --synthetic`.

## Build pipeline

```bash
# 1. Pull sessions from R2
bash scripts/pull_sessions_from_r2.sh

# 2. Ingest OCO sessions
python -m src.phase4c_rl.data.ingest_oco_sessions \
    --input-dir data/cot_rl_sessions \
    --output-dir data/oco_converted \
    --format json

# 3. Generate synthetic negatives
python -m src.phase4c_rl.data.collect_negatives --synthetic

# 4. Build preference pairs
python -m src.phase4c_rl.data.build_dpo_pairs
```

## Intended use

* **DPO training:** used by
  [`gemma4-26b-a4b-code-trainer-v10-dpo`](https://huggingface.co/cmndcntrlcyber/gemma4-26b-a4b-code-trainer-v10-dpo)
  and the corresponding Qwen DPO adapter to align model outputs with
  human-demonstrated offensive security workflows.
* **Out of scope:** this dataset contains offensive security tool-call
  traces from controlled environments (HTB, THM, bug bounty programs). It
  is not intended for training models for unauthorized access.

## Limitations

* **Small dataset.** 870 pairs is at the lower end for DPO — sufficient for
  a fine-tuning signal on top of a strong SFT foundation, but gains may
  saturate quickly.
* **Synthetic negatives.** The rejected responses are algorithmically
  degraded, not human-judged. This means the preference signal captures
  format/completeness rather than nuanced quality differences.
* **Domain-specific.** All positives come from offensive security contexts;
  the preference signal may not generalize to other coding domains.

## Reproducibility

* **Code:** [github.com/cmndcntrlcyber/code-trainer-pipeline](https://github.com/cmndcntrlcyber/code-trainer-pipeline)
  (`src/phase4c_rl/data/`)
