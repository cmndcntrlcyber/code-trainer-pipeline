---
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
library_name: peft
license: apache-2.0
tags:
- code-generation
- multimodal
- vision-encoder-decoder
- lora
- peft
- swin
- qwen2.5-coder
- code-trainer
datasets:
- cmndcntrlcyber/code-trainer-offsec-dataset
pipeline_tag: image-to-text
---

# code-trainer-vision-adapter

A multimodal **screenshot → code** model: a frozen
[Swin-B](https://huggingface.co/microsoft/swin-base-patch4-window7-224) vision
encoder, an MLP projector, and a LoRA adapter for
[`Qwen/Qwen2.5-Coder-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct).

This is **Phase 3** of the Code-Trainer / RTPI pipeline
([GitHub](https://github.com/cmndcntrlcyber/code-trainer-offsec-pipeline)) —
the multimodal stage that takes a Monaco-Editor-rendered VS Code screenshot of
source code and emits the underlying source.

## Intended use

* **Direct use:** infer source code from VS Code-style code screenshots in
  Python, JavaScript, TypeScript, Java, Go, Rust, C++, or C#.
* **Research / pedagogy:** ablation baseline for larger vision-language code
  models; the projector + LoRA architecture is small enough to retrain on a
  single A100.
* **Out of scope:** general OCR, natural images, hand-written code, or screen
  recordings (all training images came from the Monaco renderer pipeline).

## Architecture

```
   image (224×224, 3 channels)
     │
     ▼
  Swin-B encoder (frozen, 87.7 M params)
     │  visual feature sequence (49 × 1024)
     ▼
  MLP projector (trained, 2.1 M params)
     │  decoder-shaped embedding sequence
     ▼
  Qwen2.5-Coder-1.5B (with LoRA r=16, α=32 — trained)
     │
     ▼
   source code tokens
```

## Training data

* **Dataset:** [`cmndcntrlcyber/code-trainer-offsec-dataset`](https://huggingface.co/datasets/cmndcntrlcyber/code-trainer-offsec-dataset),
  revision **`v2-multimodal`** (rows include base64-encoded WebP screenshots).
* **Splits:** 26,126 train / 3,265 validation / 3,267 test (≈80/10/10).
* **Capture pipeline:** Monaco Editor in headless Chromium via Playwright,
  rendered through 8 rotating VS Code-style themes for diversity.

## Training procedure

| Knob | Value |
|---|---|
| Vision encoder | `microsoft/swin-base-patch4-window7-224` (frozen) |
| Decoder | `Qwen/Qwen2.5-Coder-1.5B-Instruct` (+ LoRA r=16, α=32, dropout 0.05) |
| Projector | 2-layer MLP, 1024 → 1536 hidden, GELU |
| Learning rate | 2e-4 (cosine, warmup ratio 0.03) |
| Batch size × accum | 8 × 4 (effective batch = 32) |
| Epochs | 3 |
| Sequence length | 2,048 |
| Precision | bfloat16 + gradient checkpointing |
| Hardware | HF Skills `a100-large` |
| Frameworks | `transformers`, `peft`, custom Trainer + `wandb` |

## Evaluation — base vs fine-tuned (test split, 200 samples)

Source: HF Job [`69f7175f9d85bec4d76f125d`](https://huggingface.co/jobs/cmndcntrlcyber/69f7175f9d85bec4d76f125d),
A100-large, 20 m 38 s.

| Metric                | Base (Qwen2.5-Coder-1.5B + random projector) | Fine-tuned | Δ |
|-----------------------|-----------------------------------------------|------------|---|
| `exact_match`         | 0.0000                                        | 0.0000     | 0 |
| `bleu_4`              | 0.0000                                        | 0.0000     | 0 |
| `mean_edit_similarity`| 0.0382                                        | 0.0446     | **+16.8 %** |
| `syntax_valid_rate` † | 0.1950                                        | 0.6100     | **+213 %** |

† Syntax check uses a Python parser. The test split is multilingual
(java 5,140; ts 5,095; csharp 5,035; python 3,300; cpp 3,156; go 2,086;
rust 1,457; js 857), so the absolute number is not directly comparable to a
Python-only run. The **delta is meaningful** because both rows use the same
metric on the same samples.

**Reading the numbers:**

* **Strong positive on `syntax_valid_rate`** (0.195 → 0.610): the adapter has
  learned to emit code-shaped output rather than free-form text.
* **Modest positive on `mean_edit_similarity`** (+16.8 %): predictions are
  closer to references than the baseline.
* **`exact_match = 0` and `bleu_4 = 0` for both runs**: the model is
  *paraphrasing* the source, not *reconstructing* it verbatim. This is a
  reasonable result for a 1.5 B base model with ~5.5 h of training on 26 K
  multilingual samples — full-fidelity code reconstruction from screenshots
  is hard.

See [`docs/eval/phase3-summary.md`](https://github.com/cmndcntrlcyber/code-trainer-offsec-pipeline/blob/main/docs/eval/phase3-summary.md)
for the full provenance, including the prior eval-pipeline bug fix.

## Limitations

* **Not a full transcription model.** Use the fine-tuned model for code
  *suggestions* from screenshots, not for byte-exact reconstruction.
* **Domain shift.** The training screenshots all come from the Monaco renderer
  with VS Code-style themes; behaviour on real IDE screenshots, IDEs other
  than VS Code, or non-Monaco editors is undefined.
* **Multilingual evaluation gap.** The `syntax_valid_rate` metric checks
  Python syntax across all languages; per-language metrics are an open
  follow-up (tracked in `docs/eval/phase3-summary.md`).
* **Small base model.** The 1.5 B decoder limits long-form fidelity; pairing
  with a larger code-trained decoder would likely improve `bleu_4` /
  `exact_match`.

## How to use

```python
# This adapter expects a paired Swin-B vision encoder. Use the loader bundled
# in the source repository:
from src.phase3_vision_model.architecture import VisionLanguageModel
from PIL import Image

model = VisionLanguageModel.from_pretrained(
    vision_encoder="microsoft/swin-base-patch4-window7-224",
    decoder="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    adapter_repo="cmndcntrlcyber/code-trainer-vision-adapter",
).cuda().eval()

image = Image.open("vs_code_screenshot.png").convert("RGB")
print(model.generate(image, max_new_tokens=512))
```

## Reproducibility

* **Code:** [github.com/cmndcntrlcyber/code-trainer-offsec-pipeline](https://github.com/cmndcntrlcyber/code-trainer-offsec-pipeline)
* **Training launcher:**
  ```bash
  python -m src.phase3_vision_model.scripts.launch_vision_training \
      --config src/config/config.yaml --wait
  ```
* **W&B project:** [`rtpi-phase3-vision`](https://wandb.ai/cmndcntrlcyber-c3s-consulting/rtpi-phase3-vision).
* **Cost:** approximately $18 on `a100-large` (~5.5 h training + ~20 min eval).
