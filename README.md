# Code-Trainer

[Final Dataset on Hugging Face](https://huggingface.co/datasets/cmndcntrlcyber/code-trainer-offsec-dataset)

## Overview

This project implements a **6-phase pipeline** to build and deploy a multimodal code generation model on an RTX 5060 Ti 16GB (Blackwell). It produces a fine-tuned Qwen2.5-Coder-14B model capable of generating code from VS Code screenshot images.

### Core Pipeline Phases

- **Phase 1: Data Collection** — Scrape GitHub repositories, filter code files, and capture Monaco Editor screenshots with syntax highlighting.
- **Phase 2: Preprocessing** — Convert captures to HuggingFace datasets in chat format and upload to Hub.
- **Phase 3: Vision Model** — Train Swin-B + MLP projector + Qwen2.5-Coder-1.5B with LoRA on local GPU.
- **Phase 4: Qwen Fine-tuning** — Qwen2.5-Coder-14B LoRA sweep + full training on HF Skills A100.
- **Phase 5: Deployment** — LoRA merge, GGUF quantization, llama.cpp/Ollama serving.
- **Phase 6: Inference** — vLLM + Qwen-Agent + MCP tool integration.

## Built With

- **Python (>=3.12)** — Core language.
- **Playwright** — Headless Chromium for Monaco Editor screenshot capture.
- **PyTorch / Transformers / PEFT / TRL** — Model training stack.

For full dependencies, see [`pyproject.toml`](./pyproject.toml).

## Project Structure

```
root/
├── data/                      # Data directory for inputs and outputs
│   ├── inputs/                # Directory for storing cloned repositories
│   ├── outputs/               # Directory for assessment/training results
│   ├── offensive-security/    # Offensive security specialized data
│   └── sample-data/           # Sample repositories for testing
│
├── src/                       # Source code
│   ├── config/                # Configuration files
│   │   ├── settings.py        # YAML loader with env var substitution
│   │   ├── config.yaml     # Central config for all 6 phases
│   │   └── offsec_config.yaml # Offensive security collection config
│   │
│   ├── phase1_data_collection/  # Phase 1: Data collection pipeline [COMPLETE]
│   │   ├── scrapers/          # GitHub scraping and repo cataloging
│   │   ├── capture/           # Screenshot capture via Monaco/Playwright
│   │   └── scripts/           # Pipeline orchestration scripts
│   │
│   ├── phase2_preprocessing/  # Phase 2: HF dataset build + Hub upload [COMPLETE — local]
│   │   ├── converters/        # Chat format, image encoding
│   │   ├── validation/        # Quality filtering, statistics
│   │   └── scripts/           # build_dataset.py, upload_to_hub.py, compute_statistics.py
│   │
│   ├── phase3_vision_model/   # Phase 3: Swin-B + MLP + Qwen-1.5B LoRA [READY — awaiting Hub]
│   │   ├── architecture/      # Model components (encoder, projector, decoder)
│   │   ├── training/          # Trainer, dataset loader, callbacks
│   │   ├── evaluation/        # Metrics, evaluator
│   │   └── scripts/           # train.py, evaluate.py, export.py
│   │
│   ├── phase4_qwen_finetuning/ # Phase 4: Qwen-14B cloud fine-tune [READY — awaiting Hub]
│   │   ├── hf_skills/         # HF Skills API client + sweep orchestrator
│   │   ├── configs/           # Sweep configs, training args
│   │   └── scripts/           # launch_validation_sweep.py, monitor_jobs.py
│   │
│   └── phase5_deployment/     # Phase 5: LoRA merge → GGUF → llama.cpp [READY — awaiting Phase 4]
│       ├── gguf/              # Converter + uploader
│       ├── inference/         # llama.cpp server wrapper
│       └── scripts/           # convert_to_gguf.py, benchmark.py
│
├── tests/                     # Unit tests
├── docs/                      # Documentation and planning
├── pyproject.toml             # Project dependencies (uv-managed)
├── Makefile                   # Common commands
└── CLAUDE.md                  # Claude Code guidance
```

## Setup

```bash
uv sync                      # Install dependencies
playwright install chromium   # Required for screenshot capture
```

## Usage

```bash
# --- Phase 1: Data Collection ---

# Run full collection pipeline (scrape + capture)
python -m src.phase1_data_collection.scripts.run_collection --config src/config/config.yaml

# Run scraping only (skip screenshot capture)
python -m src.phase1_data_collection.scripts.run_collection --config src/config/config.yaml --skip-capture

# Run offensive security collection
python -m src.phase1_data_collection.scripts.run_offsec_collection --config src/config/offsec_config.yaml

# Validate captured samples
python -m src.phase1_data_collection.scripts.validate_samples --config src/config/config.yaml

# --- Phase 2: Preprocessing ---

# Build HuggingFace dataset from captures
python -m src.phase2_preprocessing.scripts.build_dataset --config src/config/config.yaml

# Upload dataset to HF Hub (required to unblock Phases 3–4)
python -m src.phase2_preprocessing.scripts.upload_to_hub --config src/config/config.yaml

# Compute dataset statistics
python -m src.phase2_preprocessing.scripts.compute_statistics --config src/config/config.yaml

# --- Phase 3: Vision Model Training (RTX 5060 Ti) ---

python -m src.phase3_vision_model.scripts.train --config src/config/config.yaml

# --- Phase 4: Qwen-14B Fine-tuning (HF Skills Cloud) ---

python -m src.phase4_qwen_finetuning.scripts.launch_validation_sweep --config src/config/config.yaml
python -m src.phase4_qwen_finetuning.scripts.monitor_jobs --config src/config/config.yaml

# --- Phase 5: GGUF Deployment ---

python -m src.phase5_deployment.scripts.convert_to_gguf --config src/config/config.yaml

# --- Tests ---
uv run pytest tests/
```

## Experiment Tracking

Training runs are logged to Weights & Biases. The launcher scripts pass a
`WANDB_API_KEY` secret through to HF Jobs and fall back to `WANDB_MODE=offline`
when no key is supplied — local dry-runs and air-gapped reproductions stay
silent, while production runs publish to:

* **Phase 3 vision model:** [`wandb.ai/cmndcntrlcyber-c3s-consulting/rtpi-phase3-vision`](https://wandb.ai/cmndcntrlcyber-c3s-consulting/rtpi-phase3-vision)
* **Phase 4 Qwen-14B fine-tuning:** [`wandb.ai/cmndcntrlcyber-c3s-consulting/rtpi-phase43-qwen14b`](https://wandb.ai/cmndcntrlcyber-c3s-consulting/rtpi-phase43-qwen14b)

To re-enable W&B for an eval-only retroactive run (e.g. to surface metrics for
a job that originally ran offline):

```bash
WANDB_API_KEY=<key> WANDB_MODE=online \
  python -m src.phase4_qwen_finetuning.scripts.launch_eval \
  --adapter cmndcntrlcyber/qwen14b-code-trainer-aggressive \
  --val-limit 500 --wait
```

## Ready Tensor Submission

This project is being prepared for the **Ready Tensor LLMED Module 1
capstone** — see [`docs/ReadyTensor Submission/publication.md`](docs/ReadyTensor%20Submission/publication.md)
for the technical publication, and [`docs/model_cards/`](docs/model_cards/) for
the canonical Hugging Face Hub model cards. The catastrophic-forgetting
benchmark (GSM8K via lm-evaluation-harness) is launched with:

```bash
python -m src.phase4_qwen_finetuning.scripts.launch_benchmark \
  --adapter cmndcntrlcyber/qwen14b-code-trainer-aggressive --wait
python -m src.phase4_qwen_finetuning.scripts.launch_benchmark \
  --adapter cmndcntrlcyber/qwen14b-code-trainer-aggressive --baseline --wait
```
