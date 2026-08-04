"""
phase3_vision_model/hf_skills/train_entry.py

Entry script executed *inside* an HF Skills A100 job container.

Responsibilities:
    1. Parse hyperparams from env / CLI (HF Skills passes a JSON blob).
    2. Load the multimodal dataset revision from the Hub (streaming → to_local).
    3. Build CodeVisionModel (BF16, no INT4 — device_profile="a100").
    4. Run VisionModelTrainer.
    5. Push trained adapter + projector to `adapter_repo` on the Hub.

Expected env vars inside the job:
    HF_TOKEN            — write access to adapter_repo
    WANDB_API_KEY       — for metric streaming
    PHASE3_PARAMS_JSON  — serialized A100VisionConfig (set by launcher)
    PHASE3_ADAPTER_REPO — e.g. combatcougar/code-trainer-vision-adapter

Usage (local smoke test with A100 profile):
    python -m src.phase3_vision_model.hf_skills.train_entry \
        --dataset-id cmndcntrlcyber/code-trainer-offsec-dataset \
        --dataset-revision v2-multimodal \
        --adapter-repo combatcougar/code-trainer-vision-adapter
"""
import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.phase3_vision_model.architecture.vision_model import CodeVisionModel
from src.phase3_vision_model.training.trainer import VisionModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Persistent HF cache so retries don't re-download the dataset.
os.environ.setdefault("HF_HOME", "/workspace/.hf-cache")


def _download_dataset_to_disk(
    dataset_id: str,
    revision: str,
    local_dir: Path,
    limit: int | None = None,
) -> Path:
    """Download Hub dataset revision and save_to_disk so ScreenshotCodeDataset can load it.

    If `limit` is set, slice each split to the first `limit` rows — used by the
    CPU smoke test to keep runs under a few minutes.
    """
    from datasets import load_dataset

    logger.info(f"Downloading dataset {dataset_id}@{revision} → {local_dir}")
    ds = load_dataset(dataset_id, revision=revision)
    if limit is not None:
        logger.info(f"Slicing each split to first {limit} rows (smoke test)")
        ds = ds.__class__({
            split: ds[split].select(range(min(limit, len(ds[split]))))
            for split in ds.keys()
        })
    local_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(local_dir))
    return local_dir


def _push_adapter_to_hub(checkpoint_dir: Path, repo_id: str, token: str) -> None:
    """Upload LoRA adapter + projector weights to HF Hub."""
    from huggingface_hub import HfApi, create_repo

    logger.info(f"Pushing adapter → {repo_id}")
    create_repo(repo_id, token=token, private=False, exist_ok=True)
    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(checkpoint_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Phase 3 vision-model LoRA + projector (A100 run)",
    )
    logger.info(f"Adapter pushed: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Phase 3 HF Skills entry")
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--dataset-revision", default="v2-multimodal")
    parser.add_argument("--adapter-repo", default=None)
    parser.add_argument("--output-dir", default="/tmp/phase3-vision")
    parser.add_argument("--local-dataset-dir", default="/tmp/phase3-dataset")
    parser.add_argument("--skip-push", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Slice dataset splits to first N rows (smoke-test aid)")
    parser.add_argument("--device-profile", default=None,
                        help="Override device_profile (a100|5060ti|cpu)")
    args = parser.parse_args()

    # Hyperparams: prefer env JSON (HF Skills), fall back to CLI/defaults.
    params_json = os.environ.get("PHASE3_PARAMS_JSON", "{}")
    params = json.loads(params_json) if params_json else {}

    dataset_id = args.dataset_id or params.get("dataset_id") or "cmndcntrlcyber/code-trainer-offsec-dataset"
    dataset_revision = args.dataset_revision or params.get("dataset_revision") or "v2-multimodal"
    adapter_repo = (
        args.adapter_repo
        or os.environ.get("PHASE3_ADAPTER_REPO")
        or params.get("adapter_repo")
    )

    output_dir = Path(args.output_dir)
    local_dataset_dir = Path(args.local_dataset_dir)

    logger.info("=" * 60)
    logger.info("PHASE 3 — HF Skills A100 Vision Training")
    logger.info(f"  dataset:     {dataset_id}@{dataset_revision}")
    logger.info(f"  adapter:     {adapter_repo}")
    logger.info(f"  output_dir:  {output_dir}")
    logger.info("=" * 60)

    # 1. Download dataset
    _download_dataset_to_disk(
        dataset_id, dataset_revision, local_dataset_dir, limit=args.limit,
    )

    device_profile = args.device_profile or params.get("device_profile", "a100")
    # 2. Build model (A100 profile → BF16, no INT4)
    model = CodeVisionModel(
        vision_model_id=params.get("vision_encoder", "microsoft/swin-base-patch4-window7-224"),
        decoder_model_id=params.get("decoder", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
        lora_r=params.get("lora_r", 16),
        lora_alpha=params.get("lora_alpha", 32),
        lora_dropout=params.get("lora_dropout", 0.05),
        device_profile=device_profile,
    )

    # 3. Train
    trainer_config = {
        "vision_model": {
            "device_profile": device_profile,
            "batch_size": params.get("batch_size", 8),
            "gradient_accumulation": params.get("gradient_accumulation", 4),
            "learning_rate": params.get("learning_rate", 2e-4),
            "num_epochs": params.get("num_epochs", 3),
            "max_seq_length": params.get("max_seq_length", 2048),
            "use_8bit_optimizer": params.get("use_8bit_optimizer", device_profile == "5060ti"),
            "cloud": {
                "wandb_project": os.environ.get("WANDB_PROJECT", "rtpi-phase3-vision"),
            },
        },
    }
    trainer = VisionModelTrainer(
        model=model,
        dataset_dir=local_dataset_dir,
        output_dir=output_dir,
        config=trainer_config,
    )
    trainer.train()

    # 4. Push adapter to Hub
    best_dir = output_dir / "best"
    if not best_dir.exists():
        # Fall back to last epoch if early-stopping never triggered "best"
        epoch_dirs = sorted(output_dir.glob("epoch-*"))
        if epoch_dirs:
            best_dir = epoch_dirs[-1]
            logger.warning(f"No 'best' checkpoint — using {best_dir}")

    if args.skip_push or not adapter_repo:
        logger.info("Skipping adapter push (skip_push set or no adapter_repo).")
        return

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN env var required to push adapter to Hub")

    # Copy a model card alongside weights before upload
    model_card = best_dir / "README.md"
    model_card.write_text(_render_model_card(dataset_id, dataset_revision, params))
    _push_adapter_to_hub(best_dir, adapter_repo, token)

    # Clean up local dataset to free disk
    shutil.rmtree(local_dataset_dir, ignore_errors=True)
    logger.info("Phase 3 A100 job complete.")


def _render_model_card(dataset_id: str, revision: str, params: dict) -> str:
    return f"""---
tags:
- code-generation
- vision-language
- lora
- qwen2.5-coder
base_model: {params.get("decoder", "Qwen/Qwen2.5-Coder-1.5B-Instruct")}
datasets:
- {dataset_id}
---

# Code-Trainer — Phase 3 Vision Adapter

LoRA adapter + MLP projector trained on `{dataset_id}@{revision}`.

## Architecture
- **Vision encoder:** {params.get("vision_encoder", "microsoft/swin-base-patch4-window7-224")} (frozen)
- **Projector:** 2-layer MLP
- **Decoder:** {params.get("decoder", "Qwen/Qwen2.5-Coder-1.5B-Instruct")} + LoRA r={params.get("lora_r", 16)}

## Training
- Hardware: HF Skills A100-large (40GB)
- Batch size: {params.get("batch_size", 8)} × grad_accum {params.get("gradient_accumulation", 4)}
- Epochs: {params.get("num_epochs", 3)}
- LR: {params.get("learning_rate", 2e-4)}
- Precision: BF16

## Use

```python
from src.phase3_vision_model.architecture.vision_model import CodeVisionModel

model = CodeVisionModel.from_pretrained("<this-repo>")
```
"""


if __name__ == "__main__":
    main()
