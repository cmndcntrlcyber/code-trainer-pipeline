# NEXUS SWARM v4a MAX — Maximum Quality Pipeline + Command List

## Context

The baseline swarm pipeline (`pipeline-swarm-v4a.yml`) costs ~$45 allocated and makes strategic trade-offs: single-epoch SFT, SFT-only for explore/triage, no DAPT for smaller models. The max quality variant removes those constraints — every model gets DAPT (where applicable), multi-epoch SFT, full FARCA-GRPO + DPO, and abliteration on the orchestrator.

**Model change:** The orchestrator switches from Qwen2.5-Coder-14B to **Qwen3-14B** (`Qwen/Qwen3-14B`). Qwen3's improved reasoning and native thinking mode align better with the orchestrator's task decomposition and delegation role. This requires a fresh DAPT run (+$3.20) since the existing Qwen2.5 adapter doesn't transfer. Fallback: if Qwen3-14B fails the SFT validation gate, retrain with Qwen2.5-Coder-14B using the existing DAPT adapter (covered by contingency budget).

Total: ~$90.56 allocated + $20 contingency = **~$110**.

## Deliverable

New config file: `src/config/pipeline-swarm-v4a-max.yml`

This is a standalone config — not a modification of the baseline. Both configs coexist.

### What changes vs baseline

| Lever | Orchestrator | Worker | Explore | Triage |
|-------|-------------|--------|---------|--------|
| Base model | Qwen2.5-Coder-14B → **Qwen3-14B** | Qwen3-8B | Gemma-4-4B-it | Qwen3.5-0.8B |
| DAPT | reuse → **fresh** (+$3.20) | **add** offsec corpus | **add** classification corpus | — |
| SFT rows | 12K → **15K** | 8K → **12K** | 5K → **8K** | 3K → **5K** |
| SFT epochs | 1 → **2** | 1 → **2** | 1 → **2** | 1 → **3** |
| GRPO prompts | 400 → **500** | 300 → **400** | none → **200** | none → **200** |
| GRPO method | FARCA-GRPO | standard → **FARCA-GRPO** | none → **standard GRPO** | none → **standard GRPO** |
| DPO pairs | 1150 → **1500** | none → **800** | none → **300** | — |
| Abliteration | deferred → **NousResearch** | — | — | — |
| GGUF quant | Q4_K_M | Q4_K_M | Q8_0 | w8a8 RKLLM |

### Budget (max quality)

| Track | Phase | Est. Time | Cost |
|-------|-------|-----------|------|
| **ORCHESTRATOR (Qwen3-14B)** | DAPT (fresh — Qwen3-14B) | 1.0h | $3.20 |
| | SFT (15K rows × 2ep, 14B) | 8.0h | $25.60 |
| | Validation | 0.5h | $1.60 |
| | FARCA-GRPO (500 prompts) | 3.0h | $9.60 |
| | DPO (1500 pairs) | 1.5h | $4.80 |
| | Validation | 0.5h | $1.60 |
| | GGUF Q4_K_M | 0.5h | $1.60 |
| | Abliteration (NousResearch) | 1.5h | $4.80 |
| | **Subtotal** | **16.5h** | **$52.80** |
| **WORKER** | DAPT (offsec, 8B) | 1.0h | $3.20 |
| | SFT (12K rows × 2ep, 8B) | 3.0h | $9.60 |
| | Validation | 0.3h | $0.96 |
| | FARCA-GRPO (400 prompts) | 1.5h | $4.80 |
| | DPO (800 pairs) | 0.75h | $2.40 |
| | Validation | 0.3h | $0.96 |
| | GGUF Q4_K_M | 0.3h | $0.96 |
| | **Subtotal** | **7.15h** | **$22.88** |
| **EXPLORE** | DAPT (classify corpus, 4B) | 0.5h | $1.60 |
| | SFT (8K rows × 2ep, 4B) | 1.5h | $4.80 |
| | Validation | 0.2h | $0.64 |
| | GRPO (200 prompts) | 0.5h | $1.60 |
| | DPO (300 pairs) | 0.3h | $0.96 |
| | Validation | 0.2h | $0.64 |
| | GGUF Q8_0 | 0.2h | $0.64 |
| | **Subtotal** | **3.4h** | **$10.88** |
| **TRIAGE** | SFT (5K rows × 3ep, 0.8B) | 0.75h | $2.40 |
| | Validation | 0.2h | $0.64 |
| | GRPO (200 routing prompts) | 0.3h | $0.96 |
| | Validation | 0.2h | $0.64 |
| | RKLLM convert | local | $0.00 |
| | **Subtotal** | **1.45h** | **$4.64** |
| **Contingency** | Retry / Qwen2.5 fallback | 6.25h | $20.00 |
| **TOTAL** | | **~28.5h cloud** | **$110.20** |

## Command List

Every command is prepended with directory change + env loading. Phases are grouped by role. Roles can run in parallel (different base models, independent HF Jobs).

### Phase 0 — Data Preparation (local, free)

```bash
# 0.1 Build role-specific identity examples
cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase2_preprocessing.scripts.build_identity_examples \
    --output data/identity_examples/nexus_identity.jsonl --count 400

# 0.2 Build role datasets (run all four)
cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase2_preprocessing.scripts.build_swarm_role_datasets \
    --config src/config/pipeline-swarm-v4a-max.yml --role orchestrator --push-to-hub

cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase2_preprocessing.scripts.build_swarm_role_datasets \
    --config src/config/pipeline-swarm-v4a-max.yml --role worker --push-to-hub

cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase2_preprocessing.scripts.build_swarm_role_datasets \
    --config src/config/pipeline-swarm-v4a-max.yml --role explore --push-to-hub

cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase2_preprocessing.scripts.build_swarm_role_datasets \
    --config src/config/pipeline-swarm-v4a-max.yml --role triage --push-to-hub

# 0.3 Build GRPO prompts (orchestrator + worker + tool-selection exercises)
cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase4c_rl.data.build_grpo_prompts \
    --config src/config/pipeline-swarm-v4a-max.yml \
    --output-dir data/grpo_prompts_orchestrator \
    --role orchestrator --include-tool-selection-exercises \
    --max-prompts 500 --push-to-hub

cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase4c_rl.data.build_grpo_prompts \
    --config src/config/pipeline-swarm-v4a-max.yml \
    --output-dir data/grpo_prompts_worker \
    --role worker --include-tool-selection-exercises \
    --max-prompts 400 --push-to-hub

# 0.4 Pull RL session data from R2
cd /mnt/ssd/training && set -a && source .env && set +a
bash scripts/pull_sessions_from_r2.sh

# 0.5 Ingest OCO sessions + build DPO pairs
cd /mnt/ssd/training && set -a && source .env && set +a
mkdir -p data/cot_rl_sessions/{htb,thm,claude,bugbounty} data/oco_converted data/rl_data/{positives,negatives}
uv run python -m src.phase4c_rl.data.ingest_oco_sessions \
    --input-dir data/cot_rl_sessions --output-dir data/oco_converted --format json

cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase4c_rl.data.build_dpo_pairs \
    --negatives-dir data/rl_negatives --positives-dir data/oco_converted \
    --identity-examples data/identity_examples/nexus_identity.jsonl \
    --include-conciseness-pairs \
    --push-to-hub --config src/config/pipeline-swarm-v4a-max.yml
```

### Phase 1 — ORCHESTRATOR (Qwen3-14B) — ~$52.80

```bash
# 1.0 Build DAPT corpus (local — must run once before any DAPT job)
#     Shared across all roles. Skip if atlas-institute/dapt-offsec-corpus already exists on Hub.
cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase3b_dapt.data.prepare_corpus \
    --config src/config/pipeline-swarm-v4a-max.yml --role orchestrator \
    --output-dir data/dapt_corpus --push-to-hub

# 1.1 DAPT (fresh — Qwen3-14B base, offsec corpus)
#     Cannot reuse Qwen2.5 DAPT adapter — different model checkpoint.
cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase3b_dapt.scripts.launch_dapt \
    --config src/config/pipeline-swarm-v4a-max.yml --role orchestrator --wait

# 1.2 SFT (15K rows, 2 epochs)
cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase4_qwen_finetuning.scripts.launch_full_training \
    --config src/config/pipeline-swarm-v4a-max.yml --role orchestrator --wait

# 1.3 Validation gate (tool-call + agent + gsm8k)
#     FALLBACK: If gate fails, switch to Qwen2.5-Coder-14B with existing
#     DAPT adapter. Retrain SFT using baseline pipeline-swarm-v4a.yml.
cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase4_qwen_finetuning.scripts.launch_v7_validation \
    --config src/config/pipeline-swarm-v4a-max.yml \
    --role orchestrator --adapter ${HF_USERNAME}/qwen3-14b-orchestrator-sft --v10 --wait

# 1.4 FARCA-GRPO (500 prompts, 4 generations, coherence enabled)
cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase4c_rl.scripts.launch_farca_grpo \
    --config src/config/pipeline-swarm-v4a-max.yml --role orchestrator --wait

# 1.5 DPO (1500 pairs: tool + persona + conciseness)
cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase4c_rl.scripts.launch_dpo \
    --config src/config/pipeline-swarm-v4a-max.yml --role orchestrator --wait

# 1.6 Post-RL validation gate
cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase4_qwen_finetuning.scripts.launch_v7_validation \
    --config src/config/pipeline-swarm-v4a-max.yml \
    --role orchestrator --adapter ${HF_USERNAME}/qwen3-14b-orchestrator-dpo --v10 --wait

# 1.7 GGUF conversion (Q4_K_M for 5060 Ti)
cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase5_deployment.scripts.launch_convert \
    --config src/config/pipeline-swarm-v4a-max.yml --role orchestrator --wait

# 1.8 Abliteration (NousResearch biprojected)
cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase5b_abliteration.scripts.launch_abliteration \
    --config src/config/pipeline-swarm-v4a-max.yml --role orchestrator --wait
```

### Phase 2 — WORKER (Qwen3-8B) — ~$22.88 (can run in parallel with Phase 1)

```bash
# 2.1 DAPT (offsec corpus, 8B model)
cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase3b_dapt.scripts.launch_dapt \
    --config src/config/pipeline-swarm-v4a-max.yml --role worker --wait

# 2.2 SFT (12K rows, 2 epochs)
cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase4_qwen_finetuning.scripts.launch_full_training \
    --config src/config/pipeline-swarm-v4a-max.yml --role worker --wait

# 2.3 Validation gate
cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase4_qwen_finetuning.scripts.launch_v7_validation \
    --config src/config/pipeline-swarm-v4a-max.yml \
    --role worker --adapter ${HF_USERNAME}/qwen3-8b-worker-sft --v10 --wait \
    --skip-gsm8k

# 2.4 FARCA-GRPO (400 prompts, worker reward weights)
cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase4c_rl.scripts.launch_farca_grpo \
    --config src/config/pipeline-swarm-v4a-max.yml --role worker --wait

# 2.5 DPO (800 pairs)
cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase4c_rl.scripts.launch_dpo \
    --config src/config/pipeline-swarm-v4a-max.yml --role worker --wait

# 2.6 Post-RL validation gate
cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase4_qwen_finetuning.scripts.launch_v7_validation \
    --config src/config/pipeline-swarm-v4a-max.yml \
    --role worker --adapter ${HF_USERNAME}/qwen3-8b-worker-dpo --v10 --wait \
    --skip-gsm8k

# 2.7 GGUF conversion (Q4_K_M for 3070)
cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase5_deployment.scripts.launch_convert \
    --config src/config/pipeline-swarm-v4a-max.yml --role worker --wait
```

### Phase 3 — EXPLORE (Gemma-4-4B-it) — ~$10.88 (can run in parallel)

```bash
# 3.1 DAPT (classification corpus, 4B model)
cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase3b_dapt.scripts.launch_dapt \
    --config src/config/pipeline-swarm-v4a-max.yml --role explore --wait

# 3.2 SFT (8K rows, 2 epochs)
cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase4_gemma_finetuning.scripts.launch_full_training \
    --config src/config/pipeline-swarm-v4a-max.yml --role explore --wait

# 3.3 Validation gate
cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase4_qwen_finetuning.scripts.launch_v7_validation \
    --config src/config/pipeline-swarm-v4a-max.yml \
    --role explore --adapter ${HF_USERNAME}/gemma4-4b-explore-sft --v10 --wait \
    --skip-gsm8k --skip-agent

# 3.4 GRPO (200 prompts, explore-specific — read-only tools only)
cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase4c_rl.scripts.launch_grpo \
    --config src/config/pipeline-swarm-v4a-max.yml --role explore --wait

# 3.5 DPO (300 classification pairs)
cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase4c_rl.scripts.launch_dpo \
    --config src/config/pipeline-swarm-v4a-max.yml --role explore --wait

# 3.6 Post-RL validation gate
cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase4_qwen_finetuning.scripts.launch_v7_validation \
    --config src/config/pipeline-swarm-v4a-max.yml \
    --role explore --adapter ${HF_USERNAME}/gemma4-4b-explore-dpo --v10 --wait \
    --skip-gsm8k --skip-agent

# 3.7 GGUF conversion (Q8_0 for 3060)
cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase5_gemma_deployment.scripts.launch_convert \
    --config src/config/pipeline-swarm-v4a-max.yml --role explore --wait
```

### Phase 4 — TRIAGE (Qwen3.5-0.8B) — ~$4.64 (can run in parallel)

```bash
# 4.1 SFT (5K rows, 3 epochs)
cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase4_qwen_finetuning.scripts.launch_full_training \
    --config src/config/pipeline-swarm-v4a-max.yml --role triage --wait

# 4.2 Validation gate
cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase4_qwen_finetuning.scripts.launch_v7_validation \
    --config src/config/pipeline-swarm-v4a-max.yml \
    --role triage --adapter ${HF_USERNAME}/qwen35-08b-triage-sft --v10 --wait \
    --skip-gsm8k --skip-agent --skip-eval-loss

# 4.3 GRPO (200 routing prompts)
cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase4c_rl.scripts.launch_grpo \
    --config src/config/pipeline-swarm-v4a-max.yml --role triage --wait

# 4.4 Post-RL validation gate
cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase4_qwen_finetuning.scripts.launch_v7_validation \
    --config src/config/pipeline-swarm-v4a-max.yml \
    --role triage --adapter ${HF_USERNAME}/qwen35-08b-triage-grpo --v10 --wait \
    --skip-gsm8k --skip-agent --skip-eval-loss

# 4.5 RKLLM conversion (local — on OPi 5 Plus or cross-compile host)
cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase5_triage_deployment.scripts.convert_to_rkllm \
    --base-model Qwen/Qwen3.5-0.8B \
    --adapter ${HF_USERNAME}/qwen35-08b-triage-grpo \
    --output-dir data/rkllm_triage
```

### Phase 5 — Integration Validation (local, post-deployment)

```bash
# 5.1 Triage routing accuracy + latency
cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.phase5_triage_deployment.scripts.eval_triage \
    --endpoint http://nexus-opi5p:8088 \
    --output-dir data/eval_triage

# 5.2 End-to-end swarm integration test
cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.scripts.eval_swarm_e2e \
    --litellm http://nexus-control:4000 \
    --api-key sk-nexus-local \
    --output-dir data/eval_swarm_e2e
```

### One-command variant (all roles via orchestrator script)

```bash
cd /mnt/ssd/training && set -a && source .env && set +a
uv run python -m src.scripts.launch_swarm_pipeline \
    --config src/config/pipeline-swarm-v4a-max.yml --role all --wait
```

## Files to create

**`src/config/pipeline-swarm-v4a-max.yml`** — Max quality config. Structurally identical to `pipeline-swarm-v4a.yml` with these parameter upgrades:

### Orchestrator section changes vs baseline:
- `base_model`: `Qwen/Qwen2.5-Coder-14B-Instruct` → `Qwen/Qwen3-14B`
- `dapt_adapter`: reuse → fresh `${HF_USERNAME}/qwen3-14b-dapt-offsec`
- Add `orchestrator_dapt:` section (offsec corpus, Qwen3-14B base, same hyperparams as existing DAPT)
- All adapter names: `qwen14b-orchestrator-*` → `qwen3-14b-orchestrator-*`
- `sft.train_limit`: 12000 → 15000
- `sft.num_epochs`: 1 → 2
- `rl.farca_grpo.max_prompts`: 400 → 500
- `rl.dpo.max_pairs`: 1150 → 1500
- Add `abliteration:` section (NousResearch biprojected, target orchestrator adapter)
- `cloud.timeout_seconds`: 28800 → 43200 (2 epochs + DAPT needs more time)
- Fallback note: if Qwen3-14B fails SFT gate, retrain with `Qwen/Qwen2.5-Coder-14B-Instruct` using existing DAPT adapter (covered by $20 contingency)

### Worker section changes vs baseline:
- Add `dapt:` section (offsec corpus, Qwen3-8B base)
- `sft.train_limit`: 8000 → 12000
- `sft.num_epochs`: 1 → 2
- Replace `rl.grpo` with `rl.farca_grpo` (same params + FARCA config)
- Add `rl.dpo:` section (800 pairs, beta=0.1, lr=1e-6)
- `cloud.timeout_seconds`: 16200 → 32400

### Explore section changes vs baseline:
- Add `dapt:` section (classification corpus, Gemma-4-4B base)
- `sft.train_limit`: 5000 → 8000
- `sft.num_epochs`: 1 → 2
- Add `rl.grpo:` section (200 prompts, explore role, no FARCA)
- Add `rl.dpo:` section (300 classification pairs)
- `cloud.timeout_seconds`: 5400 → 16200

### Triage section changes vs baseline:
- `sft.train_limit`: 3000 → 5000
- `sft.num_epochs`: 1 → 3
- Add `rl.grpo:` section (200 routing prompts)
- `cloud.timeout_seconds`: 3600 → 7200

## Verification

1. Syntax check: `uv run python -c "from src.config.settings import load_config; load_config('src/config/pipeline-swarm-v4a-max.yml')"`
2. Dry run: `uv run python -m src.scripts.launch_swarm_pipeline --config src/config/pipeline-swarm-v4a-max.yml --role all --dry-run`
3. Phase 0 data prep runs locally — verify datasets appear on Hub
4. Each cloud phase validates via HF Jobs `meets_target` boolean in uploaded JSON
5. Integration tests (Phase 5) run against live swarm after deployment
