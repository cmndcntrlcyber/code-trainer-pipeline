# Gemma 26B Code-Trainer Pipeline

Single-source-of-truth for running the full training pipeline.
Config: `src/config/pipeline-gemma26b.yml`

---

## Just the Commands

### Prerequisites

```bash
cd /mnt/ssd/training
set -a && source .env && set +a
uv sync
```

### Fully Automated (copy-paste block)

```bash
# 0. Generate persona identity data
python -m src.phase2_preprocessing.scripts.build_identity_examples \
    --output data/identity_examples/nexus_identity.jsonl --count 400

# 1. Build V10 mixed dataset with Nexus persona injection
python -m src.phase2_preprocessing.scripts.build_v9_mixed_dataset \
    --config src/config/pipeline-gemma26b.yml \
    --identity-examples data/identity_examples/nexus_identity.jsonl \
    --inject-system-prompt \
    --hub-repo cmndcntrlcyber/code-trainer-v10-mixed

# 2. Sync edge sessions + prepare RL data
bash scripts/sync_and_prepare.sh --push-to-hub

# 3. Build persona DPO pairs
python -m src.phase4c_rl.data.build_dpo_pairs \
    --negatives-dir data/rl_negatives \
    --positives-dir data/oco_converted \
    --identity-examples data/identity_examples/nexus_identity.jsonl \
    --push-to-hub --config src/config/pipeline-gemma26b.yml

# 4. Launch training pipeline (sequential, each waits for completion)
python -m src.phase3b_dapt.scripts.launch_dapt --config src/config/pipeline-gemma26b.yml --wait
python -m src.phase4_gemma_finetuning.scripts.launch_full_training --config src/config/pipeline-gemma26b.yml --wait
python -m src.phase4c_rl.scripts.launch_farca_grpo --config src/config/pipeline-gemma26b.yml --wait
python -m src.phase4c_rl.scripts.launch_dpo --config src/config/pipeline-gemma26b.yml --wait
python -m src.phase5_gemma_deployment.scripts.launch_convert --config src/config/pipeline-gemma26b.yml --wait

# 5. (Optional) Abliteration benchmarking
python -m src.phase5b_abliteration.scripts.launch_abliteration --config src/config/pipeline-gemma26b.yml --wait
python -m src.phase5b_abliteration.scripts.generate_report --config src/config/pipeline-gemma26b.yml

# 6. Verify persona on the deployed model
ollama run hf.co/cmndcntrlcyber/gemma4-26b-a4b-code-trainer-gguf:IQ4_XS "What is your objective?"
```

### Step by Step (with explanations)

Each step below can be run independently. Steps 0-3 are **local data prep**
(free, runs on the training host). Steps 4-5 are **HF Jobs** (billed at
$3.20/hr on A100-large).

---

#### Step 0: Generate Nexus Identity Training Data

Produces ~400 synthetic conversations that teach the model its identity,
capabilities, methodology, and scope-aware behavior.

```bash
python -m src.phase2_preprocessing.scripts.build_identity_examples \
    --output data/identity_examples/nexus_identity.jsonl \
    --count 400 \
    --seed 42
```

Output: `data/identity_examples/nexus_identity.jsonl`

#### Step 1: Build V10 Mixed Dataset

Combines 5 data slices into the SFT training dataset, with the unified Nexus
system prompt injected into every record.

```bash
python -m src.phase2_preprocessing.scripts.build_v9_mixed_dataset \
    --config src/config/pipeline-gemma26b.yml \
    --identity-examples data/identity_examples/nexus_identity.jsonl \
    --inject-system-prompt \
    --output-dir data/v10_mixed \
    --hub-repo cmndcntrlcyber/code-trainer-v10-mixed
```

Dataset composition:
| Slice | Source | Count | Content |
|-------|--------|-------|---------|
| A | cmndcntrlcyber/code-trainer-offsec-dataset | ~8K | Offsec code generation |
| B | glaiveai/glaive-function-calling-v2 | ~19K | Tool-calling patterns |
| B+ | Synthetic / R2 sessions | ~2K | Multi-tool-call sequences |
| C | greghavens/fable-5-coding-and-debugging-traces | ~10K | Agentic traces |
| D | teknium/OpenHermes-2.5 | ~8K | English instruction following |
| **E** | **Synthetic identity** | **~400** | **Nexus persona Q&A** |

The `--inject-system-prompt` flag replaces all system messages with the
unified Nexus identity from `src/config/nexus_identity.py`. Slice B tool
definitions in `<tools>` blocks are preserved.

#### Step 2: Prepare RL Data (Edge Sessions)

Pull Claude sessions from R2, ingest them, build GRPO prompts and DPO pairs.

```bash
# Full sync + prep
bash scripts/sync_and_prepare.sh --push-to-hub

# Or manually:
bash scripts/pull_sessions_from_r2.sh
python -m src.phase4c_rl.data.ingest_oco_sessions \
    --input-dir data/cot_rl_sessions --output-dir data/oco_converted --format json
python -m src.phase4c_rl.data.build_grpo_prompts \
    --config src/config/pipeline-gemma26b.yml --push-to-hub
```

#### Step 3: Build DPO Pairs (with Persona Pairs)

```bash
python -m src.phase4c_rl.data.build_dpo_pairs \
    --negatives-dir data/rl_negatives \
    --positives-dir data/oco_converted \
    --identity-examples data/identity_examples/nexus_identity.jsonl \
    --output-dir data/dpo_pairs \
    --push-to-hub \
    --config src/config/pipeline-gemma26b.yml
```

The `--identity-examples` flag adds ~400 persona DPO pairs where the chosen
response identifies as Nexus (offsec framing) and the rejected response is a
vanilla AI assistant reply.

#### Step 4: Launch Training (HF Jobs A100)

Each job runs as a separate HF Job submission. `--wait` blocks until
completion. Budget: ~$50.40 across all jobs.

```bash
# Job 1: DAPT — domain-adaptive pretraining on offsec corpus ($6.40)
python -m src.phase3b_dapt.scripts.launch_dapt \
    --config src/config/pipeline-gemma26b.yml --wait

# Job 2: SFT — supervised finetuning on V10 dataset ($16.00)
python -m src.phase4_gemma_finetuning.scripts.launch_full_training \
    --config src/config/pipeline-gemma26b.yml --wait

# Job 3: FARCA-GRPO — tool-call + persona reward optimization ($9.60)
python -m src.phase4c_rl.scripts.launch_farca_grpo \
    --config src/config/pipeline-gemma26b.yml --wait

# Job 4: DPO — preference alignment with persona pairs ($4.80)
python -m src.phase4c_rl.scripts.launch_dpo \
    --config src/config/pipeline-gemma26b.yml --wait

# Job 5: GGUF — merge adapter chain + quantize ($2.40)
python -m src.phase5_gemma_deployment.scripts.launch_convert \
    --config src/config/pipeline-gemma26b.yml --wait
```

#### Step 5: (Optional) Abliteration Benchmarking

```bash
python -m src.phase5b_abliteration.scripts.launch_abliteration \
    --config src/config/pipeline-gemma26b.yml --wait
python -m src.phase5b_abliteration.scripts.launch_baseline_abliteration \
    --config src/config/pipeline-gemma26b.yml --wait
python -m src.phase5b_abliteration.scripts.generate_report \
    --config src/config/pipeline-gemma26b.yml
```

#### Step 6: Verify

```bash
ollama run hf.co/cmndcntrlcyber/gemma4-26b-a4b-code-trainer-gguf:IQ4_XS
>>> What is your objective?
# Should identify as Nexus, describe offsec mission, mention MITRE ATT&CK
>>> How do you approach a new pentest?
# Should describe systematic methodology with scope verification
```

---

## Nexus Persona System

The model's identity is defined in a single central module:

**`src/config/nexus_identity.py`** — imported by every training and eval script.

| Export | Purpose |
|--------|---------|
| `NEXUS_IDENTITY` | Core persona paragraph (threat emulation veteran, MITRE ATT&CK, scope-first) |
| `NEXUS_IDENTITY_SHORT` | One-liner for metadata |
| `SKILLS_INDEX` | Available skills (nmap-scan, nuclei, sqlmap, etc.) |
| `SUBAGENTS_INDEX` | Available subagents (recon, exploiter, web-hunter, etc.) |
| `OFFSEC_TERMS` | 47 offensive security keywords for reward scoring |
| `PERSONA_BREAK_PHRASES` | 14 phrases that indicate broken persona |
| `build_nexus_system_prompt(tools)` | Assemble full system prompt with tool list |
| `build_nexus_system_prompt_with_xml_tools(tools)` | Same, with `<tools>` XML format |

### Where the persona is enforced

| Pipeline stage | Mechanism |
|----------------|-----------|
| **SFT dataset** | `--inject-system-prompt` replaces all system messages with Nexus identity |
| **Slice E** | 400 identity Q&A examples teach direct persona responses |
| **GRPO/FARCA** | `build_nexus_system_prompt()` used as the generation context |
| **DPO** | Persona DPO pairs prefer Nexus identity over vanilla AI responses |
| **Reward function** | `persona_aligned_reasoning` component (10% weight) rewards offsec terms, penalizes persona breaks |
| **Eval** | 3 persona scenarios in agent eval; persona-breaking phrase detection in `check_progress()` |

### Reward Function Components

| Component | Weight | Scores |
|-----------|--------|--------|
| `has_valid_tool_call_tags` | 0.25 | Proper `<tool_call>` JSON format |
| `tool_name_in_schema` | 0.20 | Tool name exists in NEXUS_TOOLS_V10 |
| `has_reasoning_prefix` | 0.15 | Non-trivial text before first tool call |
| `no_hallucinated_tools` | 0.15 | No tool names outside the schema |
| `ends_cleanly_after_tag` | 0.15 | No trailing garbage after `</tool_call>` |
| `persona_aligned_reasoning` | 0.10 | Offsec terms + no persona-breaking phrases |

---

## Validation Gates

| Gate | Location | Threshold |
|------|----------|-----------|
| Post-SFT | Job 3 | tool_call_pass_rate >= 0.75, gsm8k >= 0.50 |
| Post-RL | Job 6 | tool_call_pass_rate >= 0.80, agent_progress >= 6/10 |

Agent eval now includes 10 scenarios (7 original + 3 persona).

---

## Budget

Total: **$67.00** @ $3.20/hr A100-large

| Job | Phase | Estimated | Timeout | Cost |
|-----|-------|-----------|---------|------|
| 1 | DAPT | 2.0h | 3.0h | $6.40 |
| 2 | SFT | 5.0h | 7.5h | $16.00 |
| 3 | Validation | 30min | 45min | $1.60 |
| 4 | FARCA-GRPO | 3.0h | 4.5h | $9.60 |
| 5 | DPO | 1.5h | 2.25h | $4.80 |
| 6 | Validation | 30min | 45min | $1.60 |
| 7 | GGUF | 45min | 67min | $2.40 |
| 8 | Abliteration | 2.5h | 3.75h | $8.00 |
| | Contingency | | | $9.60 |
| | Reserve | | | $7.00 |

---

## File Reference

| File | Purpose |
|------|---------|
| `src/config/nexus_identity.py` | Central persona definition (single source of truth) |
| `src/config/pipeline-gemma26b.yml` | Full pipeline config (all job params) |
| `src/phase2_preprocessing/scripts/build_identity_examples.py` | Generate Slice E persona training data |
| `src/phase2_preprocessing/scripts/build_v9_mixed_dataset.py` | V9/V10 dataset builder with persona injection |
| `src/phase2_preprocessing/scripts/synthesize_offsec_tool_calls.py` | Synthetic offsec multi-tool-call data |
| `src/phase4c_rl/rewards/tool_call_reward.py` | 6-component reward (incl. persona) |
| `src/phase4c_rl/data/build_dpo_pairs.py` | DPO pair builder with persona pairs |
| `src/phase4c_rl/hf_skills/grpo_entry.py` | GRPO training entry (HF Job) |
| `src/phase4c_rl/hf_skills/farca_grpo_entry.py` | FARCA-GRPO training entry (HF Job) |
| `src/phase4_qwen_finetuning/hf_skills/nexus_tools.py` | Canonical NEXUS_TOOLS_V10 schema |
| `src/phase4_qwen_finetuning/hf_skills/agent_eval_entry_v10.py` | Agent eval (10 scenarios incl. 3 persona) |
| `src/phase4_qwen_finetuning/hf_skills/tool_call_eval_entry_v10.py` | Tool-call format eval (14 scenarios) |
