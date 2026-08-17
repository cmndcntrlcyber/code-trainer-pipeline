# Code-Trainer

[Dataset on Hugging Face](https://huggingface.co/datasets/cmndcntrlcyber/code-trainer-offsec-dataset) · [Model Cards](docs/model_cards/)

## Overview

**Code-Trainer** (RTPI — Real-Time Pipeline Intelligence) is a multi-phase pipeline to build and deploy a fine-tuned Qwen2.5-Coder-14B model for offensive security tool-use and multi-step reasoning, served locally on an RTX 5060 Ti 16GB (Blackwell). Training runs on HF Jobs A100-large; the local GPU is for inference only.

### Pipeline Phases

| Phase | Name | Status | Description |
|-------|------|--------|-------------|
| 1 | Data Collection | Complete | GitHub scraping, Monaco Editor screenshots (32,727 captures × 8 languages) |
| 1b | PDF Ingestion | Complete | Security research PDFs rendered and captured |
| 1c | Repo Ingestion | Complete | Stars-based repo cloning and cataloging |
| 2 | Preprocessing | Complete | HF dataset conversion, chat format, Hub upload (32,658 samples) |
| 3 | Vision Model | Infrastructure complete | Swin-B + MLP projector + Qwen2.5-Coder-1.5B LoRA |
| 3b | DAPT | Infrastructure complete | Domain-adaptive continued pretraining on offsec corpus |
| 4 | SFT | V7–V9 complete | Qwen-14B LoRA instruction tuning (V9 mixed dataset: offsec + tool-calling + agent traces + instruction) |
| 4 (Gemma) | Gemma SFT | Infrastructure complete | Gemma-4-12B-it parallel training track |
| 4c | Chain-of-Thought RL | Infrastructure complete | GRPO (rule-based reward) + DPO (OCO preference pairs) |
| 5 | GGUF Deployment | V8 complete | LoRA merge → Q5_K_M quantization → llama.cpp/Ollama |
| 5 (Gemma) | Gemma GGUF | Infrastructure complete | Gemma-4-12B-it GGUF conversion |
| 5b | Abliteration | Infrastructure complete | Refusal removal benchmarking (OBLITERATUS, NousResearch, abliterix) |
| 6 | Inference | Planned | vLLM + Qwen-Agent + MCP tool integration |

### Edge Session Sync

Claude Code sessions from edge devices (Kali Docker containers) are pushed to Cloudflare R2 via `scripts/edge_push.sh` (curl-only, no Node.js/wrangler dependency), pulled to the training host with `scripts/pull_sessions_from_r2.sh`, and ingested into DPO training data via `src/phase4c_rl/data/ingest_oco_sessions.py`. For non-Docker hosts with Node.js, `scripts/bootstrap_edge_sync.sh` provides a full systemd-timer setup with wrangler.

## Built With

- **Python (>=3.12)** — Core language.
- **Playwright** — Headless Chromium for Monaco Editor screenshot capture.
- **PyTorch / Transformers / PEFT / TRL** — Model training stack.
- **Cloudflare R2 + Wrangler** — Edge session sync pipeline.

For full dependencies, see [`pyproject.toml`](./pyproject.toml).

## Project Structure

```
root/
├── data/
│   ├── captures/              # Phase 1 screenshot captures (gitignored)
│   ├── cot_rl_sessions/       # Edge device Claude sessions for RL training
│   │   ├── htb/               # HackTheBox completed room histories
│   │   ├── thm/               # TryHackMe completed room histories
│   │   ├── claude/            # Claude session histories (all hosts)
│   │   └── bugbounty/         # Bug bounty program sessions
│   ├── offensive-security/    # Offensive security specialized data
│   └── sample-data/           # Sample repositories for testing
│
├── scripts/
│   ├── edge_push.sh               # Self-contained edge pusher (curl-only, no wrangler needed)
│   ├── bootstrap_edge_sync.sh     # Full edge setup with systemd timer (requires Node.js + wrangler)
│   ├── sync_sessions_to_r2.sh     # Push sessions from training host to R2
│   ├── pull_sessions_from_r2.sh   # Pull sessions from R2 to training host
│   └── sync_config.sh             # Shared config (allowlist, exclusions, env validation)
│
├── src/
│   ├── config/
│   │   ├── settings.py            # YAML loader with ${VAR} env substitution
│   │   ├── config.yaml            # Central config for all phases
│   │   ├── pipeline-50.yml        # $50 full pipeline run config (8 jobs, validation gates)
│   │   └── budget-config.yml      # Budget-constrained config variant
│   │
│   ├── phase1_data_collection/    # Phase 1: GitHub scraping + Monaco screenshots [COMPLETE]
│   ├── phase1_repo_ingestion/     # Phase 1c: Stars-based repo cloning [COMPLETE]
│   │
│   ├── phase2_preprocessing/      # Phase 2: HF dataset build + Hub upload [COMPLETE]
│   │   └── scripts/               # build_dataset, build_v7/v8/v9_mixed_dataset, upload_to_hub
│   │
│   ├── phase3_vision_model/       # Phase 3: Swin-B + MLP + Qwen-1.5B LoRA
│   ├── phase3b_dapt/              # Phase 3b: Domain-adaptive continued pretraining
│   │   ├── data/                  # Corpus preparation, PDF text extraction
│   │   ├── hf_skills/             # dapt_entry.py (HF Job container entry)
│   │   └── scripts/               # launch_dapt.py
│   │
│   ├── phase4_qwen_finetuning/    # Phase 4: Qwen-14B SFT (V7–V10 datasets)
│   │   ├── hf_skills/             # Train, eval, benchmark, agent eval, tool_call eval entries
│   │   ├── configs/               # Sweep configs, training args
│   │   └── scripts/               # launch_validation_sweep, launch_v7_validation, launch_v9_training
│   │
│   ├── phase4_gemma_finetuning/   # Phase 4 (Gemma): Gemma-4-12B-it SFT
│   │
│   ├── phase4c_rl/                # Phase 4c: Chain-of-thought RL (GRPO + DPO)
│   │   ├── data/                  # ingest_oco_sessions, build_dpo_pairs, build_grpo_prompts, collect_negatives
│   │   ├── rewards/               # tool_call_reward.py (5-component rule-based reward)
│   │   ├── hf_skills/             # grpo_entry.py, dpo_entry.py
│   │   └── scripts/               # launch_grpo.py, launch_dpo.py
│   │
│   ├── phase5_deployment/         # Phase 5: LoRA merge → GGUF Q5_K_M → llama.cpp
│   ├── phase5_gemma_deployment/   # Phase 5 (Gemma): Gemma GGUF conversion
│   │
│   └── phase5b_abliteration/      # Phase 5b: Abliteration benchmarking
│       ├── techniques/            # obliteratus, nousresearch, abliterix
│       ├── evaluation/            # Refusal rate, perplexity, KL divergence, lm-eval-harness
│       ├── hf_skills/             # abliterate_entry.py, baseline_entry.py
│       └── scripts/               # launch_abliteration, launch_baseline_abliteration, generate_report
│
├── tests/                         # Unit tests
├── docs/                          # Documentation and planning
│   ├── enhancements/              # Version-specific enhancement docs (v2.1, v2.2, v3.0)
│   └── model_cards/               # HF Hub model cards (V7, V8, V9, GGUF)
├── .github/workflows/             # CI: weekly R2 session inventory health check
├── pyproject.toml                 # Project dependencies (uv-managed)
└── CLAUDE.md                      # Claude Code guidance
```

## Setup

```bash
uv sync                      # Install dependencies
playwright install chromium   # Required for screenshot capture
```

## Usage

All commands run from the project root (`/mnt/ssd/training/`).

```bash
set -a && source .env && set +a   # Load environment variables

# --- Phase 1: Data Collection ---

python -m src.phase1_data_collection.scripts.run_collection --config src/config/config.yaml
python -m src.phase1_data_collection.scripts.validate_samples --config src/config/config.yaml

# --- Phase 2: Preprocessing ---

python -m src.phase2_preprocessing.scripts.build_dataset --config src/config/config.yaml
python -m src.phase2_preprocessing.scripts.upload_to_hub --config src/config/config.yaml

# --- Phase 3b: DAPT (HF Jobs A100) ---

python -m src.phase3b_dapt.scripts.launch_dapt --config src/config/pipeline-50.yml --wait

# --- Phase 4: SFT (HF Jobs A100) ---

python -m src.phase4_qwen_finetuning.scripts.launch_validation_sweep --config src/config/config.yaml
python -m src.phase4_qwen_finetuning.scripts.launch_v9_training --config src/config/config.yaml --wait

# --- Phase 4c: Chain-of-Thought RL ---

# Prepare session data
mkdir -p data/cot_rl_sessions/{htb,thm,claude,bugbounty} data/oco_converted data/rl_data/{positives,negatives} && bash scripts/pull_sessions_from_r2.sh

# Ingest and convert sessions
python -m src.phase4c_rl.data.ingest_oco_sessions --input-dir data/cot_rl_sessions --output-dir data/oco_converted --format json

# Launch RL training (HF Jobs A100)
python -m src.phase4c_rl.scripts.launch_grpo --config src/config/pipeline-50.yml --wait
python -m src.phase4c_rl.scripts.launch_dpo --config src/config/pipeline-50.yml --wait

# --- Phase 5: GGUF Deployment ---

python -m src.phase5_deployment.scripts.launch_convert --config src/config/config.yaml --wait

# --- Phase 5b: Abliteration Benchmarking ---

python -m src.phase5b_abliteration.scripts.launch_abliteration --config src/config/config.yaml --wait
python -m src.phase5b_abliteration.scripts.launch_baseline_abliteration --config src/config/config.yaml --wait
python -m src.phase5b_abliteration.scripts.generate_report --config src/config/config.yaml

# --- Tests ---
uv run pytest tests/
```

## Experiment Tracking

Training runs are logged to Weights & Biases. Launcher scripts pass
`WANDB_API_KEY` through to HF Jobs and fall back to `WANDB_MODE=offline`
when no key is supplied.

* **Phase 3 vision model:** `rtpi-phase3-vision`
* **Phase 3b DAPT:** `rtpi-phase3b-dapt`
* **Phase 4 Qwen-14B SFT:** `rtpi-phase4-qwen14b`
* **Phase 4c RL:** `rtpi-phase4c-rl`

## Key Design Decisions

- **Q5_K_M default** — Q4_K_M degrades `<tool_call>` tag fidelity across V7/V8/V9. Q5_K_M is +1.5 GB (10.5 vs 9 GB) but preserves multi-token patterns. Fits the RTX 5060 Ti 16GB.
- **No VS Code or Xvfb** for capture — Monaco Editor runs in headless Chromium via Playwright.
- **BF16 compute** throughout training (Blackwell tensor cores).
- **Curriculum training** — 80/20 split (full data / tool-call polish) to fix tag emission.
- **Edge session sync** — Cloudflare R2 as the hub; edge devices push via wrangler on a systemd timer; training host pulls with ETag-based change detection.
