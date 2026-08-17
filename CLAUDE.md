# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Code-Trainer** (RTPI — Real-Time Pipeline Intelligence) is a multi-phase pipeline to build and deploy a fine-tuned Qwen2.5-Coder-14B model for offensive security tool-use and multi-step reasoning on an RTX 5060 Ti 16GB (Blackwell). Training runs on HF Jobs A100-large; the local GPU is for inference only.

All scripts are run from the project root (`/mnt/ssd/training/`). Config is loaded via `src/config/config.yaml` (or `src/config/pipeline-50.yml` for the $50 pipeline run). Required environment variables are in `.env` (see `.env.example`).

## Required Environment Variables

```bash
export GITHUB_TOKEN=...          # GitHub API token for scraping repos
export HF_USERNAME=...           # HuggingFace username for dataset/model Hub paths
export HF_TOKEN=...              # HuggingFace API token for Hub push/pull
export CLOUDFLARE_ACCOUNT_ID=... # Cloudflare account for R2 session sync
export CLOUDFLARE_API_TOKEN=...  # Cloudflare API token for R2 session sync
```

## Setup

```bash
uv sync                      # Install dependencies via pyproject.toml
playwright install chromium   # Required for screenshot capture
```

## Common Commands

**Run Phase 1 data collection (full pipeline):**
```bash
python -m src.phase1_data_collection.scripts.run_collection --config src/config/config.yaml
```

**Run scraping only (skip screenshot capture):**
```bash
python -m src.phase1_data_collection.scripts.run_collection --config src/config/config.yaml --skip-capture
```

**Run capture only on already-cloned repos:**
```bash
python -m src.phase1_data_collection.scripts.run_collection --config src/config/config.yaml --skip-scraping
```

**Override repos per language (quick test):**
```bash
python -m src.phase1_data_collection.scripts.run_collection --config src/config/config.yaml --repos-per-language 5
```

**Validate captured samples:**
```bash
python -m src.phase1_data_collection.scripts.validate_samples --config src/config/config.yaml
python -m src.phase1_data_collection.scripts.validate_samples --config src/config/config.yaml --fix  # remove invalid captures
```

**Run Phase 3b DAPT (HF Jobs):**
```bash
python -m src.phase3b_dapt.scripts.launch_dapt --config src/config/pipeline-50.yml --wait
```

**Run Phase 4c RL — prepare session data and launch training:**
```bash
mkdir -p data/cot_rl_sessions/{htb,thm,claude,bugbounty} data/oco_converted data/rl_data/{positives,negatives} && bash scripts/pull_sessions_from_r2.sh
python -m src.phase4c_rl.data.ingest_oco_sessions --input-dir data/cot_rl_sessions --output-dir data/oco_converted --format json
python -m src.phase4c_rl.scripts.launch_grpo --config src/config/pipeline-50.yml --wait
python -m src.phase4c_rl.scripts.launch_dpo --config src/config/pipeline-50.yml --wait
```

**Run Phase 5 GGUF conversion (HF Jobs):**
```bash
python -m src.phase5_deployment.scripts.launch_convert --config src/config/config.yaml --wait
```

**Run Phase 5b abliteration benchmarking (HF Jobs):**
```bash
python -m src.phase5b_abliteration.scripts.launch_abliteration --config src/config/config.yaml --wait
python -m src.phase5b_abliteration.scripts.launch_baseline_abliteration --config src/config/config.yaml --wait
python -m src.phase5b_abliteration.scripts.generate_report --config src/config/config.yaml
```

**Run a single abliteration technique locally (dev/test):**
```bash
python -m src.phase5b_abliteration.scripts.run_abliteration --technique nousresearch --model-dir data/gguf_work/merged_model --output-dir data/abliterated/nousresearch
```

**Edge session sync (run on training host):**
```bash
bash scripts/pull_sessions_from_r2.sh
```

**Run via Docker (Xvfb provided by entrypoint):**
```bash
cd src/phase1_data_collection/docker
docker compose up
```

**Run tests:**
```bash
uv run pytest tests/
```

**Format code:**
```bash
black .
```

## Architecture

**Implementation status:**
- Phase 1: Complete — 32,727 captures across 8 languages
- Phase 1b: Complete — PDF security research capture
- Phase 1c: Complete — Stars-based repo ingestion (`src/phase1_repo_ingestion/`)
- Phase 2: Complete — 32,658-sample HuggingFace dataset; V7/V8/V9 mixed datasets built and uploaded
- Phase 3: Infrastructure complete — Swin-B vision model awaiting Hub dataset
- Phase 3b: Infrastructure complete — DAPT on offsec corpus (`src/phase3b_dapt/`)
- Phase 4 (Qwen): V7–V9 SFT complete — V10 pending Phase 4c RL data
- Phase 4 (Gemma): Infrastructure complete — Gemma-4-12B-it parallel track (`src/phase4_gemma_finetuning/`)
- Phase 4c: Infrastructure complete — GRPO + DPO scripts, reward function, session ingestion (`src/phase4c_rl/`)
- Phase 5 (Qwen): V8 GGUF complete — Q5_K_M default (`src/phase5_deployment/`)
- Phase 5 (Gemma): Infrastructure complete (`src/phase5_gemma_deployment/`)
- Phase 5b: Infrastructure complete — abliteration benchmarking with baseline support (`src/phase5b_abliteration/`)
- Phase 6: Not implemented — documented in `docs/enhancements/archive/plan/Inference-Agent-Architecture .md`

### Phase 1 Data Collection Pipeline

```
GitHubScraper → SQLiteCatalog → FileFilter → ScreenshotManager → ParallelCapture → MonacoCapture
```

**`src/phase1_data_collection/scrapers/`**
- `github_scraper.py` — Discovers and clones repos via GitHub Search API. Quality-filters using `QualityScorer` (0–100 score based on stars, activity, docs, code quality, community). Parallelizes cloning with `ThreadPoolExecutor`.
- `quality_scorer.py` — Scores repos 0–100 across 5 components (20 pts each) and classifies into categories (`security`, `ai_ml`, `web`, `automation`, `data`, `tool`, `general`).
- `file_filter.py` — Filters to code files 20–500 lines, 200B–50KB. Skips `node_modules`, `__pycache__`, `test/tests`, `vendor`, `build`, etc.
- `sqlite_catalog.py` — SQLite store at `data/catalog.db` tracking repos and per-file captures. Schema: `repositories` + `captures` tables with quality/language indexes.

**`src/phase1_data_collection/capture/`**
- `vscode_automation.py` — `MonacoCapture` class. Launches headless Chromium via Playwright, loads Monaco Editor from CDN, renders source code with syntax highlighting, and scrolls/screenshots the full file. Each capture directory (`data/captures/<hash2>/<hash>/`) contains numbered PNGs, `source.txt`, and `metadata.json`.
- `parallel_capture.py` — `ParallelCapture` distributes file batches across N Chromium instances using `asyncio.gather`.
- `screenshot_manager.py` — High-level coordinator: applies theme rotation, instantiates `ParallelCapture`, and writes results to SQLite.
- `theme_manager.py` — Rotates through 8 VS Code-style Monaco themes for training data diversity.

**`src/phase1_data_collection/scripts/`**
- `run_collection.py` — Main orchestrator. Phase 1A: scrape + clone repos. Phase 1B: filter files + capture screenshots.
- `validate_samples.py` — Validates capture directories for `source.txt`, `metadata.json`, and valid PNG files. Reports stats from both filesystem and SQLite.

**`src/config/`**
- `settings.py` — YAML loader with `${VAR}` environment variable substitution.
- `config.yaml` — Central config for all phases. Target: 500 repos/language × 8 languages = 4,000 repos, 50,000+ captures.
- `pipeline-50.yml` — $50 full pipeline run config (8 sequential jobs with validation gates, contingency budget).
- `budget-config.yml` — Budget-constrained variant with reduced hyperparameters.

### Capture Output Structure

```
data/captures/
  <2-char prefix>/
    <16-char sha256 hash>/
      0000.png, 0001.png, ...   # viewport-height scrolled screenshots
      source.txt                 # raw source code
      metadata.json              # language, line count, theme, viewport, etc.
```

### Phase Structure

| Phase | Location | Purpose |
|-------|----------|---------|
| 1 | `src/phase1_data_collection/` | GitHub scraping + Monaco Editor screenshots |
| 1b | `src/phase1_data_collection/scripts/run_pdf_ingestion.py` | PDF security research capture |
| 1c | `src/phase1_repo_ingestion/` | Stars-based repo cloning and cataloging |
| 2 | `src/phase2_preprocessing/` | HF dataset conversion, chat format, V7–V9 mixed datasets, Hub upload |
| 3 | `src/phase3_vision_model/` | Swin-B + MLP projector + Qwen2.5-Coder-1.5B LoRA training |
| 3b | `src/phase3b_dapt/` | Domain-adaptive continued pretraining on offsec corpus |
| 4 | `src/phase4_qwen_finetuning/` | Qwen2.5-Coder-14B LoRA SFT (V7–V10 datasets, HF Jobs A100) |
| 4-Gemma | `src/phase4_gemma_finetuning/` | Gemma-4-12B-it LoRA SFT (parallel track) |
| 4c | `src/phase4c_rl/` | Chain-of-thought RL: GRPO + DPO with OCO session data |
| 5 | `src/phase5_deployment/` | LoRA merge → GGUF Q5_K_M → llama.cpp/Ollama serve |
| 5-Gemma | `src/phase5_gemma_deployment/` | Gemma GGUF conversion |
| 5b | `src/phase5b_abliteration/` | Abliteration benchmarking (OBLITERATUS, NousResearch, abliterix) |
| 6 | Not implemented | vLLM + Qwen-Agent + MCP tool integration |

### Key Design Decisions

- **No VS Code or Xvfb required** for capture — Monaco Editor runs in headless Chromium via Playwright, loaded from jsDelivr CDN.
- **Docker alternative** (`src/phase1_data_collection/docker/`) provides Xvfb for legacy VS Code-based capture paths.
- **BF16 compute** throughout training (Blackwell tensor cores).
- **Q5_K_M default** — Q4_K_M degrades `<tool_call>` tag fidelity. Q5_K_M is +1.5 GB but fits the RTX 5060 Ti 16GB.
- **Curriculum training** — 80/20 split (full data / tool-call polish) to fix V9 tag emission failures.
- **Edge session sync** — Claude sessions from edge Kali containers push to Cloudflare R2 via `scripts/edge_push.sh` (curl-only, no wrangler), pulled to training host for RL data. For non-Docker hosts, `scripts/bootstrap_edge_sync.sh` provides full systemd-timer setup. Allowlisted projects defined in `scripts/sync_config.sh`.
- **Single GPU + hot-swap** strategy (not concurrent dual-model) for Phase 6 inference.
- **vLLM over Ollama** for Phase 6: Ollama tool-calling for Qwen3.5 is broken (issue #14493).
