"""
phase5_gemma_deployment/gguf/uploader.py

Upload GGUF files to HuggingFace Hub and generate a model card.
"""
import logging
from pathlib import Path

from huggingface_hub import HfApi

logger = logging.getLogger(__name__)

MODEL_CARD_TEMPLATE = """---
license: apache-2.0
base_model: google/gemma-4-12B-it
tags:
  - code
  - gguf
  - gemma-4
  - lora
  - fine-tuned
  - code-generation
language:
  - code
pipeline_tag: text-generation
---

# {model_name}

Fine-tuned [Gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it)
for code generation from chat-formatted instructions.

## Model Description

This model was fine-tuned as part of the **Code-Trainer (RTPI)** project —
a multimodal code generation pipeline trained on {num_samples:,}+ VS Code screenshot captures
across 8 programming languages.

**Task:** Given a code generation instruction, produce clean, correct source code.

## Training

| Parameter | Value |
|---|---|
| Base model | google/gemma-4-12B-it |
| Fine-tuning method | LoRA (PEFT) |
| LoRA rank | {lora_r} |
| LoRA alpha | {lora_alpha} |
| Learning rate | {learning_rate} |
| Epochs | {num_epochs} |
| Hardware | HuggingFace Skills A100-large (80GB) |
| Dataset | {dataset_id} |

## Limitations

- Optimised for code generation across 8 languages (Python, JS, TS, Java, Go, Rust, C++, C#)
- Not safety-tuned — inherits the base model's safety properties
- General benchmark capability should be verified for regression

## How to Use

```python
from llama_cpp import Llama

llm = Llama(
    model_path="path/to/model_q4_k_m.gguf",
    n_gpu_layers=-1,
    n_ctx=4096,
)

response = llm.create_chat_completion(messages=[
    {{"role": "user", "content": "Write a Rust function that parses an ISO-8601 timestamp."}},
])
print(response["choices"][0]["message"]["content"])
```

## Experiment Tracking

W&B project: {wandb_url}
"""


class GGUFUploader:
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
        try:
            self.api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        except Exception as e:
            logger.warning(f"Repo creation warning: {e}")

        filename = f"model_{quant_type.lower()}.gguf"
        logger.info(f"Uploading {filename} to {repo_id}...")
        self.api.upload_file(
            path_or_fileobj=str(gguf_path),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add {quant_type} GGUF",
        )

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
