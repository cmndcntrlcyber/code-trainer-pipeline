# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Code-Trainer** (RTPI — Real-Time Pipeline Intelligence) is a 6-phase pipeline to build and deploy a multimodal code generation model on an RTX 5060 Ti 16GB (Blackwell). It produces a fine-tuned Qwen2.5-Coder-14B model capable of generating code from VS Code screenshot images.

All scripts are run from the project root (`/mnt/ssd/training/`). Config is loaded via `src/config/config.yaml`, which requires two environment variables: `GITHUB_TOKEN` and `HF_USERNAME`.

## Required Environment Variables

```bash
export GITHUB_TOKEN=...     # GitHub API token for scraping repos
export HF_USERNAME=...      # HuggingFace username for dataset/model Hub paths
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

**Run Phase 5b abliteration benchmarking (HF Jobs):**
```bash
python -m src.phase5b_abliteration.scripts.launch_abliteration --config src/config/config.yaml --wait
```

**Run a single abliteration technique locally (dev/test):**
```bash
python -m src.phase5b_abliteration.scripts.run_abliteration --technique nousresearch --model-dir data/gguf_work/merged_model --output-dir data/abliterated/nousresearch
```

**Generate abliteration comparative report:**
```bash
python -m src.phase5b_abliteration.scripts.generate_report --config src/config/config.yaml
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

Phases 1–5b are fully implemented. Phase 6 is planned but not yet implemented — its architecture is documented in `docs/plan/Inference-Agent-Architecture.md`.

**Implementation status:**
- Phase 1: Complete — 32,727 captures across 8 languages
- Phase 2: Complete locally — 32,658-sample HuggingFace dataset built; Hub upload pending
- Phase 3: Infrastructure complete — awaiting Phase 2 dataset on Hub
- Phase 4: Infrastructure complete — awaiting Phase 2 dataset on Hub
- Phase 5: Infrastructure complete — awaiting Phase 4 fine-tuned checkpoint
- Phase 5b: Infrastructure complete — abliteration benchmarking (OBLITERATUS, NousResearch, abliterix)
- Phase 6: Not implemented — documented in `docs/plan/Inference-Agent-Architecture.md`

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
- `config.yaml` — Central config for all 6 phases. Target: 500 repos/language × 8 languages = 4,000 repos, 50,000+ captures.

### Capture Output Structure

```
data/captures/
  <2-char prefix>/
    <16-char sha256 hash>/
      0000.png, 0001.png, ...   # viewport-height scrolled screenshots
      source.txt                 # raw source code
      metadata.json              # language, line count, theme, viewport, etc.
```

### Planned Phase Structure (not yet implemented)

| Phase | Location | Purpose |
|-------|----------|---------|
| 2 | `src/phase2_preprocessing/` | HF dataset conversion, chat format, Hub upload |
| 3 | `src/phase3_vision_model/` | Swin-B + MLP projector + Qwen2.5-Coder-1.5B LoRA training (local GPU) |
| 4 | `src/phase4_qwen_finetuning/` | Qwen2.5-Coder-14B LoRA sweep + full training (HF Skills A100) |
| 5 | `src/phase5_deployment/` | LoRA merge → GGUF Q4_K_M → llama.cpp/Ollama serve |
| 5b | `src/phase5b_abliteration/` | Abliteration benchmarking: OBLITERATUS, NousResearch, abliterix |
| 6 | Separate repo/config | vLLM + Qwen-Agent + MCP tool integration |

### Key Design Decisions

- **No VS Code or Xvfb required** for capture — Monaco Editor runs in headless Chromium via Playwright, loaded from jsDelivr CDN.
- **Docker alternative** (`src/phase1_data_collection/docker/`) provides Xvfb for legacy VS Code-based capture paths.
- **BF16 compute** throughout training (Blackwell tensor cores).
- **Single GPU + hot-swap** strategy (not concurrent dual-model) for Phase 6 inference: Qwen3.5-9B Q6_K as primary, Qwen2.5-Coder-14B Q4_K_M swapped in for compiled-language tasks.
- **vLLM over Ollama** for Phase 6: Ollama tool-calling for Qwen3.5 is broken (issue #14493).
