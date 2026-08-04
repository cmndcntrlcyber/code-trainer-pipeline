"""
phase5_deployment/gguf/uploader.py

Upload GGUF files to HuggingFace Hub and generate a model card.
"""
import logging
from pathlib import Path

from huggingface_hub import HfApi

logger = logging.getLogger(__name__)

MODEL_CARD_TEMPLATE = """---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-14B-Instruct
tags:
  - code
  - gguf
  - qwen
  - lora
  - fine-tuned
  - code-generation
language:
  - code
pipeline_tag: text-generation
---

# {model_name}

Fine-tuned [Qwen2.5-Coder-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct)
for code generation from VS Code screenshot transcription tasks.

## Model Description

This model was fine-tuned as part of the **Code-Trainer (RTPI)** project —
a multimodal code generation pipeline trained on {num_samples:,}+ VS Code screenshot captures
across 8 programming languages.

**Task:** Given a VS Code screenshot, generate the exact source code shown.

## Training

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen2.5-Coder-14B-Instruct |
| Fine-tuning method | QLoRA (4-bit NF4 + LoRA) |
| LoRA rank | {lora_r} |
| LoRA alpha | {lora_alpha} |
| Learning rate | {learning_rate} |
| Epochs | {num_epochs} |
| Hardware | HuggingFace Skills A100-large (40GB) |
| Dataset | {dataset_id} |

## Evaluation Results

| Metric | Baseline | Fine-tuned | Change |
|---|---|---|---|
| Exact Match | {baseline_em:.3f} | {finetuned_em:.3f} | {em_delta:+.3f} |
| BLEU-4 | {baseline_bleu:.3f} | {finetuned_bleu:.3f} | {bleu_delta:+.3f} |
| Edit Similarity | {baseline_edit:.3f} | {finetuned_edit:.3f} | {edit_delta:+.3f} |

## Limitations

- Optimised for VS Code Monaco editor screenshots at 2560×1440 / 14px font size
- Multi-screenshot (scroll) captures: only the first frame is used for this model
- General benchmark capability (MMLU, HellaSwag) should be verified for regression

## How to Use

```python
from llama_cpp import Llama

llm = Llama(
    model_path="path/to/model_q4_k_m.gguf",
    n_gpu_layers=-1,
    n_ctx=8192,
)

response = llm.create_chat_completion(messages=[
    {{"role": "system", "content": "You are a code transcription assistant..."}},
    {{"role": "user", "content": "What code is displayed in this VS Code screenshot?"}},
])
print(response["choices"][0]["message"]["content"])
```

## Experiment Tracking

W&B project: {wandb_url}

## Citation

```
@misc{{code-trainer,
  author = {{combatcougar}},
  title = {{Code-Trainer: Multimodal Code Generation from VS Code Screenshots}},
  year = {{2026}},
}}
```
"""


class GGUFUploader:
    """Upload GGUF model files and model card to HF Hub."""

    def __init__(self, token: str):
        self.api = HfApi(token=token)

    def upload(
        self,
        gguf_path: Path,
        repo_id: str,
        model_card_params: dict,
        quant_type: str = "Q4_K_M",
        private: bool = False,
    ) -> str:
        """
        Upload GGUF + model card to HF Hub.

        Returns:
            Hub URL for the model
        """
        # Create repo if needed
        try:
            self.api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        except Exception as e:
            logger.warning(f"Repo creation warning: {e}")

        # Upload GGUF file
        filename = f"model_{quant_type.lower()}.gguf"
        logger.info(f"Uploading {filename} to {repo_id}...")
        self.api.upload_file(
            path_or_fileobj=str(gguf_path),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add {quant_type} GGUF",
        )

        # Generate and upload model card
        card_content = MODEL_CARD_TEMPLATE.format(
            model_name=repo_id.split("/")[-1],
            **model_card_params,
        )
        self.api.upload_file(
            path_or_fileobj=card_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add model card",
        )

        url = f"https://huggingface.co/{repo_id}"
        logger.info(f"Model uploaded: {url}")
        return url
