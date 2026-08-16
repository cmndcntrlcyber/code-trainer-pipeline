# RTPI — Development Roadmap & Completion Tracker

> **Real-Time Pipeline Intelligence** for local AI inference and agent orchestration on RTX 5060 Ti 16GB

---

## Project Overview

RTPI combines two workstreams into a unified local AI system:

| Pillar | Purpose | Source Document |
|--------|---------|-----------------|
| **Training Pipeline** (Phases 1–5) | Build a multimodal code generation model from VS Code screenshots, fine-tune Qwen-14B via cloud GPU, and deploy as GGUF | `Code-Trainer-RTX5060Ti-Single-GPU.md` |
| **Inference & Agent Stack** (Phase 6) | Serve Qwen3.5-9B as primary local model with vLLM + Qwen-Agent + MCP tool integration | `Inference-Agent-Architecture.md` |

**Hardware:** MSI RTX 5060 Ti 16GB Ventus 3X OC (Blackwell) — $520 — **inference only (Phase 6)**
**Cloud Budget:** ~$140 (HF Skills A100 — Phase 3 vision + Phase 4 Qwen-14B)
**Total Project Cost:** ~$660

> **Strategy change (2026-04-15):** Phase 3 vision model training pivots from local RTX 5060 Ti to cloud A100. The 5060 Ti's 128-bit memory bus and 16GB VRAM make end-to-end vision+decoder training too slow and memory-tight to iterate on; A100 40GB allows larger batches, faster epochs, and the same toolchain already wired for Phase 4. The 5060 Ti is reserved for Phase 5 GGUF conversion and Phase 6 inference.

---

## Current Status (as of 2026-05-02)

| Phase | Status | Output | Next Step |
|-------|--------|--------|-----------|
| 1 — Data Collection | ✓ Complete | 32,727 captures, 8 languages | — |
| 2 — Preprocessing | ✓ Complete | Dataset on HF Hub: `main` (text-only) + `v2-multimodal` (images, base64 WebP, 26126/3265/3267) | — |
| 3 — Vision Model Training | ✓ Complete | A100 job `69f55aeb98a8d679adfb8621` ran 5h 34m (2026-05-02). Adapter at `cmndcntrlcyber/code-trainer-vision-adapter`. Eval (test/200): finetuned `syntax_valid_rate` 0.61 vs 0.20 baseline (3.13×), `mean_edit_similarity` +16.8%; `exact_match`/`bleu_4` both 0 — model paraphrases, doesn't reconstruct verbatim. Details in `docs/eval/phase3-summary.md`. | — |
| 4 — Qwen-14B Fine-tuning | ⚠ In flight (Phase 4A sweep) | Pivoted from broken AutoTrain path to HF Jobs (mirror Phase 3). 3 A100-large jobs submitted 2026-05-03: `qwen14b-{conservative,standard,aggressive}`. Adapters publish to `cmndcntrlcyber/qwen14b-code-trainer-{name}`. | Wait for terminal stages, pick best by eval_loss, launch Phase 4B |
| 5 — GGUF Deployment | ⚠ Ready (blocked) | Infrastructure only | Awaits Phase 4 checkpoint |
| 6 — Inference & Agent Stack | ⚠ Scaffolded | `src/phase6_inference/` — vllm_serve.sh, qwen_agent_client.py, mcp_servers.json, hot_swap.py, scripts/setup.md. Code-only; deployment is an inference-host task separate from training deps. | Build vLLM nightly on the 5060 Ti host, smoke-test agent + MCP |

**Critical path:** ~~Phase 3 cloud port~~ → Phase 4A sweep → Phase 4B full training → Phase 5 GGUF → Phase 6 serve. Phase 3 done; Phase 4 unblocked.

> **CUDA pin (2026-05-02):** torch and torchvision are pinned to PyTorch's `cu128` wheel index via `[tool.uv.sources]` in `pyproject.toml`. The default HF Skills container `huggingface/transformers-pytorch-gpu:latest` ships an NVIDIA driver capped at CUDA 12.9, so cu13 wheels (uv's natural resolution for torch 2.11.0) crash at startup. cu128 supports both that container and the local 5060 Ti (Blackwell sm_120). `uv.lock` is now tracked; cloud install uses `uv sync --frozen`.

---

## LLMED Certification Tracker

> **Program:** ReadyTensor LLMED (LLM Engineering & Deployment)
> **Next Submission Deadline:** April 06, 2026, 11:59 PM UTC

### Credentials

| Credential | Module | Status |
|---|---|---|
| **LLM Fine-Tuning Specialist** | Module 1 (Capstone) | In Progress |
| LLM Deployment Engineer | Module 2 | Not Started |
| **LLM Engineering & Deployment Certification** | Module 1 + 2 | Not Started |

### Module 1 Capstone — Essential Requirements

| # | Requirement | RTPI Phase | Status |
|---|---|---|---|
| 1 | Dataset Selection & Preparation | Phase 1 + Phase 2 | [x] Phase 1: 32,727 captures (0 invalid); Phase 2: 32,658-sample dataset published to HF Hub at `cmndcntrlcyber/code-trainer-offsec-dataset` (public) — `main` text-only + `v2-multimodal` branch with base64 WebP images |
| 2 | Baseline Evaluation (pre-fine-tune metrics) | Phase 3 | [~] Infrastructure complete (evaluate.py); not yet run — pending |
| 3 | Fine-Tuning (LoRA/QLoRA, PEFT) | Phase 3 (cloud) + Phase 4 (cloud) | [x] Phase 3 complete: A100 5h 34m run, adapter at `cmndcntrlcyber/code-trainer-vision-adapter`. Phase 4 still pending. |
| 4 | Post-Fine-Tuning Evaluation + general benchmark | Phase 3/4 eval | [~] Infrastructure complete; not yet run — needs adapter download + local eval or cloud eval job |
| 5 | Experiment Tracking (W&B) | Phase 3 & 4 | [~] W&B wired but Phase 3 ran with `WANDB_MODE=offline` (no `WANDB_API_KEY` configured); offline run dir is inside the now-released container and was not captured. Configure `WANDB_API_KEY` for Phase 4. |
| 6 | Model Publishing (HF Hub + model card) | Phase 4B + Phase 5 | [~] Phase 3 adapter + model card published; Phase 4B pending |
| 7 | Reproducible Code (scripts, requirements.txt) | All phases | [x] All phases implemented (Phase 1–5 scripts complete) |

### Module 1 Capstone — Submission Deliverables

| # | Deliverable | Status |
|---|---|---|
| 1 | ReadyTensor technical publication (charts, tables, examples) | [ ] Not started — draft after Phase 3/4 training completes |
| 2 | GitHub repository (README, training/eval code, requirements.txt, model card) | [~] Phase 1–5 code complete; README needs W&B link + eval results |
| 3 | Published model on HF Hub (adapters + model card) | [~] Dataset uploading; model card template in uploader.py |
| 4 | W&B experiment tracking (logged runs + link in README) | [ ] Set WANDB_API_KEY and run Phase 3 train.py |

### Module 1 Capstone — Optional Enhancements (Bonus)

| Enhancement | RTPI Coverage | Status |
|---|---|---|
| Multi-GPU Training (DDP/FSDP/DeepSpeed) | Phase 4 — A100 cloud via HF Skills | [ ] Planned |
| Hyperparameter Tuning (multiple experiments) | Phase 4A — 3-config validation sweep | [ ] Planned |
| Model Merging (LoRA → base) | Phase 5 — LoRA merge + GGUF quantize | [ ] Planned |
| Additional Benchmarks (MMLU, HellaSwag, GSM8K) | Phase 3/4 eval pipeline | [ ] Planned |
| Advanced Evaluation (operational checks) | Phase 3 eval — can add format/length checks | [ ] Not planned yet |

---

## Architecture Overview

```
TRAINING PIPELINE                          INFERENCE & AGENT STACK
═══════════════════                        ════════════════════════

Phase 1: Data Collection (CPU)             Phase 6: Runtime Deployment
  GitHub Scraper → VS Code Automation        ┌─────────────────────────┐
  500 repos/lang × 8 languages               │  Qwen3.5-9B (Q6_K)     │
          │                                   │  Primary: 7.46GB        │
          ▼                                   │  Context: ~160K (FP8)   │
Phase 2: Preprocessing                       │  Speed: ~40-50 t/s      │
  Quality Filter → Chat Format → HF Hub      ├─────────────────────────┤
          │                                   │  Hot-swap:              │
    ┌─────┴─────┐                             │  Qwen2.5-Coder-14B     │
    ▼           ▼                             │  (Q4_K_M) for compiled  │
Phase 3:    Phase 4:                          │  language code gen      │
Vision      Qwen-14B                          └──────────┬──────────────┘
Model       Fine-tune                                    │
(Local)     (Cloud)                           vLLM Nightly (sm_120)
    │           │                                        │
    └─────┬─────┘                             Qwen-Agent + MCP
          ▼                                              │
Phase 5: GGUF Deployment                     MCP Servers (filesystem,
  LoRA Merge → Quantize → Serve              fetch, code_interpreter)
```

---

## Phase 1: Data Collection

**Goal:** Scrape high-quality GitHub repos and capture VS Code screenshots for training data.

### Completion Tracker

#### Prerequisites
- [x] Python environment with dependencies installed (`pip install -r requirements.txt`)
- [x] Playwright + Chromium installed (`playwright install chromium`)
- [x] GitHub API token configured (`GITHUB_TOKEN` env var)
- [x] Docker environment for headless capture (optional)
- [x] SQLite catalog database initialized

#### Implementation
- [x] `phase1_data_collection/scrapers/github_scraper.py` — GitHub repository discovery
- [x] `phase1_data_collection/scrapers/quality_scorer.py` — Quality scoring (0–100, 5 components × 20 pts)
- [x] `phase1_data_collection/scrapers/file_filter.py` — Code file filtering (20–500 lines, 200B–50KB)
- [x] `phase1_data_collection/scrapers/sqlite_catalog.py` — SQLite metadata store (repos + captures tables)
- [x] `phase1_data_collection/capture/vscode_automation.py` — Playwright Monaco capture via headless Chromium
- [x] `phase1_data_collection/capture/screenshot_manager.py` — Screenshot capture coordinator
- [x] `phase1_data_collection/capture/parallel_capture.py` — Multi-worker capture
- [x] `phase1_data_collection/capture/theme_manager.py` — Theme variations for data diversity
- [x] `phase1_data_collection/scripts/run_collection.py` — Main orchestrator
- [x] `phase1_data_collection/scripts/validate_samples.py` — Quality validation
- [x] `phase1_data_collection/scrapers/offsec_scraper.py` — Offensive security specialized scraper
- [x] `phase1_data_collection/scrapers/offsec_keywords.py` — OffSec domain classification (13 domains)
- [x] `phase1_data_collection/scripts/run_offsec_collection.py` — OffSec collection orchestrator
- [x] `config/offsec_config.yaml` — OffSec collection configuration

#### Validation
- [x] Scraper discovers repos across all 8 languages (Python, JS, TS, Java, C++, Go, Rust, C#)
- [x] Quality scoring filters repos below threshold (min score: 30)
- [ ] Screenshot capture produces valid PNG files with metadata JSON
- [ ] Parallel capture runs workers simultaneously
- [x] SQLite catalog tracks repos and captures correctly
- [x] OffSec scraper discovers 7,673 repos across 13 security domains

#### Success Criteria
| Metric | Target | Actual |
|--------|--------|--------|
| Captured samples | ≥50,000 | 32,727 (collection ongoing; target not yet met) |
| Languages covered | 8 | 8 ✓ |
| Avg quality score | >30 | Verified ✓ |
| OffSec repos discovered | ≥5,000 | 7,673 ✓ |
| OffSec domains covered | 13 | 13 ✓ |

---

## Phase 2: Preprocessing & Hub Upload

**Goal:** Convert raw captures into a HuggingFace dataset with Qwen chat format.

### Completion Tracker

#### Prerequisites
- [x] Phase 1 captures directory populated
- [x] HuggingFace account + API token configured
- [x] `datasets` and `huggingface_hub` packages installed

#### Implementation
- [x] `phase2_preprocessing/converters/hf_dataset_converter.py` — Convert captures to HF dataset format
- [x] `phase2_preprocessing/converters/chat_formatter.py` — Qwen chat template formatting (system/user/assistant)
- [x] `phase2_preprocessing/converters/image_encoder.py` — Base64 WebP image encoding
- [x] `phase2_preprocessing/validation/quality_filter.py` — Filter low-quality samples
- [x] `phase2_preprocessing/validation/statistics.py` — Dataset statistics generation
- [x] `phase2_preprocessing/scripts/build_dataset.py` — Full preprocessing pipeline
- [x] `phase2_preprocessing/scripts/upload_to_hub.py` — HF Hub upload (private repo)
- [x] `phase2_preprocessing/scripts/compute_statistics.py` — Stats report

#### Validation
- [x] Dataset split correctly (80/10/10 train/val/test) — 26,126 / 3,265 / 3,267
- [x] Chat format follows Qwen template (system → user prompt → assistant code)
- [ ] Images encoded as base64 WebP — text-only first pass complete; image encoding pass pending
- [x] Code truncated at 8192 chars max
- [x] 7 diverse user prompt variations cycle correctly (~3,700 each in train split)

#### Success Criteria
| Metric | Target | Actual |
|--------|--------|--------|
| Dataset built locally | Yes | 32,658 samples ✓ (at `data/hf_dataset/`) |
| Dataset uploaded to HF Hub | Yes | ✓ `cmndcntrlcyber/code-trainer-offsec-dataset` (public, 175 MB Parquet) |
| Train/Val/Test split | 80/10/10 | 26,100 / 3,270 / 3,270 ✓ (Hub split counts) |
| All samples have images + code | Yes | Text-only first pass live on Hub; **image-encoding pass still pending** for true multimodal training |

---

## Phase 3: Vision Model Training (HF Skills A100 — Cloud)

**Goal:** Train a multimodal vision-to-code model on cloud A100 GPUs via HuggingFace Skills. Local RTX 5060 Ti training was abandoned — bandwidth and VRAM forced micro-batches that made epochs impractically slow.

### Architecture (unchanged)
- **Vision Encoder:** Swin-B (frozen, 88M params, ~1.5GB BF16)
- **Projector:** 2-layer MLP (~4M params, ~0.3GB)
- **Decoder:** Qwen2.5-Coder-1.5B-Instruct (BF16 + LoRA r=16 on A100, no INT4 needed)
- **Batch size on A100-large (40GB):** 8–16 (vs 2 on 5060 Ti) with grad_accum=4

### Completion Tracker

#### Prerequisites
- [x] Phase 2 dataset on HF Hub (`cmndcntrlcyber/code-trainer-offsec-dataset`)
- [ ] **Image-encoding pass:** re-export dataset with base64 WebP image column (current Hub dataset is text-only)
- [x] HuggingFace Skills account with A100-large access
- [x] W&B account configured for experiment tracking
- [ ] `phase3_vision_model/hf_skills/` module built (mirrors `phase4_qwen_finetuning/hf_skills/`)

#### Implementation
- [x] `phase3_vision_model/architecture/vision_encoder.py` — Swin-B encoder wrapper
- [x] `phase3_vision_model/architecture/multimodal_projector.py` — Vision-to-text 2-layer MLP
- [x] `phase3_vision_model/architecture/code_decoder.py` — Qwen-1.5B decoder with INT4 + LoRA
- [x] `phase3_vision_model/architecture/vision_model.py` — Full CodeVisionModel assembly
- [x] `phase3_vision_model/training/trainer.py` — RTX 5060 Ti optimized trainer (BF16, 8-bit AdamW)
- [x] `phase3_vision_model/training/dataset.py` — PyTorch dataset for screenshot-code pairs
- [x] `phase3_vision_model/training/collator.py` — Data collation with padding
- [x] `phase3_vision_model/training/callbacks.py` — Training callbacks (GPU monitoring, early stopping)
- [x] `phase3_vision_model/evaluation/metrics.py` — Code generation metrics (BLEU, exact match)
- [x] `phase3_vision_model/evaluation/evaluator.py` — Evaluation pipeline
- [x] `phase3_vision_model/scripts/train.py` — Training entry point (baseline eval → train → post-FT eval)
- [x] `phase3_vision_model/scripts/evaluate.py` — Standalone evaluation entry point
- [x] `phase3_vision_model/scripts/export.py` — LoRA merge + export

#### Training Configuration (A100-large, 40GB)
| Setting | Value |
|---------|-------|
| Batch size | 8 |
| Gradient accumulation | 4 (effective: 32) |
| Sequence length | 2048 tokens |
| Compute dtype | BF16 |
| Optimizer | AdamW (fused) |
| Learning rate | 2e-4 |
| Epochs | 3–5 |
| Gradient checkpointing | Enabled |
| Estimated cost | ~$20–35 (6–10h on A100 @ $3.20/hr) |

#### Validation
- [ ] A100 job launches via HF Skills and streams W&B metrics
- [ ] BF16 mixed precision active
- [ ] Gradient checkpointing + fused AdamW keep VRAM <32GB
- [ ] Model generates coherent code from test screenshots
- [ ] W&B logs training curves correctly
- [ ] Trained adapter pushed to HF Hub (`cmndcntrlcyber/code-trainer-vision-adapter`)

#### Capstone Deliverables (Module 1)
- [ ] Baseline evaluation recorded (pre-fine-tune metrics) → **Capstone Req #2**
- [ ] W&B run logged with hyperparams + training metrics → **Capstone Req #5**
- [ ] Post-fine-tuning evaluation with same metrics as baseline → **Capstone Req #4**
- [ ] General benchmark (MMLU subset or HellaSwag) for catastrophic forgetting check → **Capstone Req #4**

#### Success Criteria
| Metric | Target | Actual |
|--------|--------|--------|
| Vision model eval loss | <0.5 | |
| VRAM peak usage | <16GB | |
| Training completes | All epochs | |

---

## Phase 4: Qwen-14B Fine-tuning (HF Skills Cloud)

**Goal:** Fine-tune Qwen2.5-Coder-14B-Instruct on cloud A100 GPUs via HuggingFace Skills.

### Completion Tracker

#### Prerequisites
- [ ] Phase 2 dataset on HF Hub (private)
- [ ] HuggingFace Skills account with A100-large access
- [ ] W&B integration configured for cloud jobs

#### Phase 4A: Validation Sweep (~$21)
- [x] `phase4_qwen_finetuning/hf_skills/job_client.py` — HF Skills API client
- [x] `phase4_qwen_finetuning/hf_skills/sweep_orchestrator.py` — Parallel sweep manager
- [x] `phase4_qwen_finetuning/hf_skills/job_monitor.py` — Job status monitoring
- [x] `phase4_qwen_finetuning/configs/sweep_configs.py` — Hyperparameter configs
- [x] `phase4_qwen_finetuning/configs/training_args.py` — TrainingArguments builder
- [ ] Launch 3 parallel validation jobs: **awaiting dataset on HF Hub**
  - [ ] Conservative: LoRA r=16, α=32, lr=1.5e-4, bs=1, grad_accum=16
  - [ ] Standard: LoRA r=32, α=64, lr=2e-4, bs=2, grad_accum=8
  - [ ] Aggressive: LoRA r=64, α=128, lr=3e-4, bs=4, grad_accum=4
- [ ] All 3 validation jobs complete successfully
- [ ] Best config identified by eval loss

#### Phase 4B: Full Training (~$56)
- [ ] Launch top-2 configs for full 3-epoch training
- [ ] Both full training jobs complete successfully
- [ ] Best model pushed to HF Hub
- [ ] Generate results report

#### Scripts
- [x] `phase4_qwen_finetuning/scripts/launch_validation_sweep.py`
- [x] `phase4_qwen_finetuning/scripts/launch_full_training.py`
- [x] `phase4_qwen_finetuning/scripts/monitor_jobs.py`
- [x] `phase4_qwen_finetuning/scripts/generate_report.py`

#### Capstone Deliverables (Module 1)
- [ ] W&B runs logged for all sweep + full training jobs → **Capstone Req #5**
- [ ] Best model pushed to HF Hub with complete model card → **Capstone Req #6**
  - [ ] Model description and intended use
  - [ ] Training data and procedure
  - [ ] Evaluation results (baseline vs fine-tuned)
  - [ ] Limitations and known issues
  - [ ] Code example for loading and using the model
- [ ] Before/after performance comparison documented → **Capstone Req #4**
- [ ] Training loss curves exported for publication → **Deliverable #1**

#### Success Criteria
| Metric | Target | Actual |
|--------|--------|--------|
| Validation sweep | 3/3 jobs complete | |
| Full training | 2/2 jobs complete | |
| Best eval loss | <1.0 | |
| Cloud cost | ~$77 | |

---

## Phase 5: GGUF Deployment

**Goal:** Convert fine-tuned model to GGUF format and deploy on RTX 5060 Ti for inference.

### Completion Tracker

#### Prerequisites
- [ ] Phase 4 best model available on HF Hub
- [ ] llama.cpp built with CUDA support (sm_120 for Blackwell)
- [ ] Sufficient disk space for model conversion (~30GB temp)

#### Implementation
- [x] `phase5_deployment/gguf/converter.py` — GGUF conversion pipeline (merge + convert + quantize)
- [x] `phase5_deployment/gguf/uploader.py` — Hub upload + model card generation
- [x] `phase5_deployment/inference/llama_cpp_server.py` — llama.cpp server wrapper
- [x] `phase5_deployment/scripts/convert_to_gguf.py` — Conversion entry point
- [x] `phase5_deployment/scripts/benchmark.py` — Performance benchmarks

#### Conversion Steps
- [ ] Download fine-tuned LoRA adapter from HF Hub
- [ ] Merge LoRA with base Qwen2.5-Coder-14B (on CPU, float16)
- [ ] Convert merged model to GGUF F16
- [ ] Quantize to Q4_K_M (~9GB)
- [ ] Upload GGUF to HF Hub
- [ ] Start llama.cpp server and validate

#### GGUF Size Reference
| Quantization | Model Size | KV Cache (4K) | Total VRAM | Headroom |
|---|---|---|---|---|
| **Q4_K_M** (recommended) | ~9 GB | 2.5 GB | 12.5 GB | 3.5 GB |
| Q5_K_M | ~10 GB | 2.5 GB | 13.5 GB | 2.5 GB |
| Q8_0 (7B only) | ~8 GB | 2.0 GB | 11.0 GB | 5.0 GB |

#### Capstone Deliverables (Module 1)
- [ ] LoRA merged into base model → **Capstone Optional: Model Merging**
- [ ] Merged model published to HF Hub → **Capstone Req #6**
- [ ] ReadyTensor technical publication drafted → **Deliverable #1**
  - [ ] Objective, dataset, methodology sections
  - [ ] Training loss curve charts
  - [ ] Baseline vs fine-tuned comparison tables
  - [ ] Example input/output demonstrations
- [ ] GitHub README finalized with setup instructions → **Deliverable #2**
- [ ] W&B project link added to README → **Deliverable #4**

#### Success Criteria
| Metric | Target | Actual |
|--------|--------|--------|
| Inference speed | >50 tok/s | |
| GGUF uploaded to Hub | Yes | |
| Server responds to API calls | Yes | |

---

## Phase 6: Inference & Agent Stack

**Goal:** Deploy the production inference system with Qwen3.5-9B as primary model, vLLM serving, and Qwen-Agent for MCP tool integration.

### Model Selection (Decided)

| Role | Model | Quant | VRAM | Context | Speed |
|------|-------|-------|------|---------|-------|
| **Primary** | Qwen3.5-9B | Q6_K | 7.46 GB | ~160K (FP8 KV) | ~40-50 t/s |
| **Code specialist** | Qwen2.5-Coder-14B-Instruct | Q4_K_M | 8.99 GB | ~32K (Q8 KV) | ~33 t/s |
| **Rust specialist** | Strand-Rust-Coder-14B | Q4_K_M | 8.99 GB | ~32K | ~33 t/s |

### Completion Tracker

#### Prerequisites
- [ ] RTX 5060 Ti drivers + CUDA 12.x installed
- [ ] vLLM nightly built from source (sm_120 Blackwell support)
- [ ] Python 3.10+ environment
- [ ] Qwen3.5-9B model downloaded (Q6_K GGUF or HF safetensors)
- [ ] Qwen2.5-Coder-14B model downloaded (Q4_K_M for hot-swap)
- [ ] Node.js installed (for MCP servers via npx)

#### vLLM Deployment
- [ ] Build vLLM nightly from source with sm_120 support
- [ ] Configure vLLM serve command:
  ```
  vllm serve Qwen/Qwen3.5-9B \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --language-model-only \
    --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
  ```
- [ ] Validate vLLM serves OpenAI-compatible API on port 8000
- [ ] Test MTP speculative decoding is active
- [ ] Verify FP8 KV cache enabled for extended context
- [ ] Benchmark throughput matches expectations (~40-50 t/s at Q6_K)

#### Qwen-Agent + MCP Integration
- [ ] Install Qwen-Agent framework
- [ ] Configure MCP server connections:
  - [ ] `@modelcontextprotocol/server-filesystem` — file system access
  - [ ] `mcp-server-fetch` — HTTP fetch capability
  - [ ] `code_interpreter` — code execution sandbox
- [ ] Wire Qwen-Agent to vLLM endpoint (`http://localhost:8000/v1`)
- [ ] Test tool calling round-trip (agent → vLLM → tool → response)
- [ ] Validate structured output generation (IFEval compliance)

#### Model Hot-Swap
- [ ] Implement model swap mechanism (Qwen3.5-9B ↔ Qwen2.5-Coder-14B)
- [ ] Measure swap latency (target: 3–10 seconds on NVMe)
- [ ] Configure routing logic (compiled language tasks → Coder-14B)
- [ ] Test swap under load

#### Ollama Fallback (Optional)
- [ ] `ollama run qwen3.5:9b` for simple interactive use
- [ ] Note: Ollama tool calling for Qwen3.5 is broken (issue #14493) — use vLLM for agentic workloads

#### Validation
- [ ] Qwen3.5-9B serves via vLLM with tool calling functional
- [ ] MCP servers connected and responding
- [ ] Code generation produces valid output for Python/JS/TS
- [ ] Hot-swap to Coder-14B works for Rust/C++/Go tasks
- [ ] Context window handles ≥80K tokens without OOM
- [ ] Vision mode works when `--language-model-only` flag removed

#### Success Criteria
| Metric | Target | Actual |
|--------|--------|--------|
| vLLM + MCP operational | Yes | |
| Tool calling (BFCL-V4) | Validated | |
| Usable context window | ≥80K tokens | |
| Primary model speed | ≥40 t/s | |
| Hot-swap latency | <10 seconds | |

---

## Architecture Decision Log

Decisions already made based on research and benchmarking:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary inference model | Qwen3.5-9B at Q6_K | Hybrid Gated DeltaNet architecture reduces KV cache 75%; 66.1 BFCL-V4 for tool calling; beats models 3x its size |
| Quantization strategy | Smaller model at higher quant (Q6_K) over larger at Q4 | Q6_K is near-lossless; Qwen3.5-9B Q6_K (7.46GB) beats Qwen-14B Q4_K_M (8.99GB) in effective quality |
| Code specialist | Qwen2.5-Coder-14B-Instruct Q4_K_M | 5.5T code tokens, 92 languages; SOTA at 14B for compiled languages |
| Inference engine | vLLM nightly (not Ollama) | Native tool calling, MTP speculative decoding, FP8 KV cache; Ollama has broken Qwen3.5 tool templates |
| Agent framework | Qwen-Agent | Native MCP support, OpenAI-compatible API bridge, handles tool parsing internally |
| Vision encoder (training) | Swin-B (frozen) | Proven architecture, 88M params fits in VRAM frozen |
| Training decoder | Qwen2.5-Coder-1.5B (not 3B) | Comfortable 16GB fit with 3GB headroom |
| Cloud training | HF Skills A100-large | $3.20/hr, 40GB VRAM, parallel sweep support |
| Local compute dtype | BF16 | Blackwell tensor cores natively support BF16 |
| Multi-model strategy | Single model + hot-swap (not concurrent) | Concurrent dual-model limits context to 8-16K per model; hot-swap is pragmatic |

---

## Cost & Resource Tracking

| Phase | Component | Hardware | Budget | Actual | Status |
|-------|-----------|----------|--------|--------|--------|
| — | RTX 5060 Ti 16GB | GPU (inference only) | $520 | | [ ] Purchased |
| 1 | Data Collection | CPU | $0 | $0 | [x] Complete |
| 2 | Preprocessing | CPU + Hub | $0 | $0 | [x] Complete — dataset on Hub |
| 3 | Vision Model Training | A100-large (~8h) | ~$26 | | [ ] Cloud port pending |
| 4A | Validation Sweep | 3× A100 (~2h) | ~$21 | | [ ] Not started |
| 4B | Full Training | 2× A100 (~8h) | ~$56 | | [ ] Not started |
| 5 | GGUF Conversion | RTX 5060 Ti | $0 | | [ ] Not started |
| 6 | Inference Stack | RTX 5060 Ti | $0 | | [ ] Not started |
| **Total** | | | **~$643** | | |

---

## Dependencies & Phase Order

```
Phase 1 ──→ Phase 2 ──┬──→ Phase 3 (cloud A100) ──┐
                       │                            ├──→ Phase 5 ──→ Phase 6
                       └──→ Phase 4 (cloud A100) ──┘

Phases 3 and 4 are independent cloud jobs and can run in parallel on HF Skills.
Phase 6 can begin independently (model download, vLLM setup, MCP config)
before Phases 3–5 complete — it uses pre-trained Qwen models.
```

**Critical path:** Phase 1 → Phase 2 → Phase 4 → Phase 5 (cloud training is the bottleneck)
**Parallel work:** Phase 6 setup can proceed alongside Phases 1–5

---

## Next Phase Plan — Cloud Vision Training (2026-04-15)

The immediate next phase is porting Phase 3 vision-model training to HF Skills A100 and running it in parallel with Phase 4A. Target: both sweeps launched within the same working session.

### Step 1 — Dataset image-encoding pass (local CPU, ~2h)
- [ ] Add `--include-images` path to `phase2_preprocessing/scripts/build_dataset.py` that attaches base64 WebP image bytes to each sample's `messages[1].content`.
- [ ] Downscale screenshots to 896×896 (Swin-B input) before encoding to keep Parquet under ~10 GB.
- [ ] Push as new revision/branch `v2-multimodal` on `cmndcntrlcyber/code-trainer-offsec-dataset` (keep text-only `main` for Phase 4 text fine-tuning).

### Step 2 — Phase 3 cloud port (`src/phase3_vision_model/hf_skills/`, ~1 day)
Mirror the Phase 4 pattern exactly — the job client, monitor, and sweep orchestrator are reusable.
- [ ] `hf_skills/job_client.py` — thin wrapper (copy from Phase 4, rename entry script).
- [ ] `hf_skills/train_entry.py` — A100 entry script: loads `v2-multimodal` revision, builds `CodeVisionModel`, runs trainer. Drop all bitsandbytes/INT4 code paths (A100 has the VRAM for BF16).
- [ ] `configs/a100_training_args.py` — bs=8, grad_accum=4, epochs=3, LoRA r=16, fused AdamW.
- [ ] `scripts/launch_vision_training.py` — single-job launcher (no sweep for v1, one config).
- [ ] Remove 5060 Ti-specific assumptions in `training/trainer.py` (INT4 quant, 8-bit optim) behind a `device_profile: "a100" | "5060ti"` switch so local smoke tests still work.

### Step 3 — Baseline + train + eval run (A100, ~8h, ~$26)
- [ ] Run `evaluate.py` against base `Qwen2.5-Coder-1.5B-Instruct` on the test split → baseline metrics (Capstone Req #2).
- [ ] Launch training job; stream to W&B project `rtpi-phase3-vision`.
- [ ] On completion: post-FT eval with same metric script → comparison table.
- [ ] Push adapter to `cmndcntrlcyber/code-trainer-vision-adapter` with model card.

### Step 4 — Parallel Phase 4A kickoff
Launch the 3-config Qwen-14B validation sweep on the text-only dataset revision the same day Phase 3 starts. Phases 3 and 4A are independent — no reason to serialize.

### Step 5 — Cleanup
- [ ] Update `CLAUDE.md` to reflect cloud-only training (5060 Ti = inference only).
- [ ] Archive local-training notes in `docs/archive/local-5060ti-training.md`.
- [ ] Remove `bitsandbytes` from required deps (keep optional for local smoke tests).

### Risks & mitigations
- **Dataset image re-encode doubles Hub storage.** Use a dataset branch, not a new repo, and keep only the revisions actively in use.
- **HF Skills quota / queue.** Launch Phase 3 and 4A staggered by ~15 min so both don't contend for the same A100-large pool.
- **Image-token length blow-up.** Swin-B produces 49 patch tokens after pooling; keep `max_seq_len=2048` and verify no sample truncates code tail.

---

## Known Limitations

- Local 9B model cannot match frontier models on SWE-bench (65-80% frontier vs no competitive sub-14B results)
- Complex multi-file refactoring and unfamiliar API integration remain weak points
- Cloud API fallback recommended for hardest 5-10% of tasks
- Ollama tool calling for Qwen3.5 is broken (template mismatch, issue #14493)
- RTX 5060 Ti 128-bit bus is the memory bandwidth bottleneck for autoregressive decoding — and was the reason Phase 3 training was moved to cloud A100
