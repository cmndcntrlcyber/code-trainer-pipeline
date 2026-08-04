"""
phase4_gemma_finetuning/hf_skills/eval_entry.py

Eval-only entry: load Gemma-4-12B + a published LoRA adapter from Hub, run
SFTTrainer.evaluate() against the dataset's val split, push the result
JSON back to the adapter repo.

Expected env vars:
    HF_TOKEN              — read base model + adapter, write eval JSON back
    PHASE4_PARAMS_JSON    — { "model_id", "dataset_id", "dataset_revision",
                              "max_seq_length", "adapter_repo", "val_limit?",
                              "result_filename?" }
    PHASE4_ADAPTER_REPO   — adapter to evaluate (also used as upload target)
"""
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

os.environ.setdefault("HF_HOME", "/workspace/.hf-cache")


def _format_chat(example, tokenizer):
    """Merge system role into user (Gemma 4 has no system role)."""
    messages = example["messages"]
    reformatted = []
    system_text = ""
    for msg in messages:
        if msg["role"] == "system":
            system_text = msg["content"]
        elif msg["role"] == "user":
            content = f"{system_text}\n\n{msg['content']}" if system_text else msg["content"]
            reformatted.append({"role": "user", "content": content})
            system_text = ""
        else:
            reformatted.append(msg)
    return {"text": tokenizer.apply_chat_template(
        reformatted, tokenize=False, add_generation_prompt=False,
    )}


def main():
    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer

    params = json.loads(os.environ.get("PHASE4_PARAMS_JSON", "{}"))
    adapter_repo = os.environ.get("PHASE4_ADAPTER_REPO") or params.get("adapter_repo")
    if not adapter_repo:
        raise RuntimeError("PHASE4_ADAPTER_REPO required")

    model_id = params.get("model_id", "google/gemma-4-12B-it")
    dataset_id = params.get("dataset_id", "cmndcntrlcyber/code-trainer-offsec-dataset")
    dataset_revision = params.get("dataset_revision", "main")
    max_seq_length = int(params.get("max_seq_length", 4096))
    val_limit = params.get("val_limit") or os.environ.get("PHASE4_VAL_LIMIT")
    result_filename = params.get("result_filename", "phase4-eval-full.json")
    output_dir = Path(params.get("output_dir", "/tmp/phase4-eval"))
    output_dir.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN required to read adapter / push result")

    logger.info("=" * 60)
    logger.info("PHASE 4 GEMMA — Eval-only")
    logger.info(f"  base:    {model_id}")
    logger.info(f"  adapter: {adapter_repo}")
    logger.info(f"  dataset: {dataset_id}@{dataset_revision} val{f' (limit {val_limit})' if val_limit else ''}")
    logger.info(f"  out:     {result_filename}")
    logger.info("=" * 60)

    # 1. Tokenizer + chat formatting
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    logger.info(f"Loading dataset {dataset_id}@{dataset_revision}")
    ds = load_dataset(dataset_id, revision=dataset_revision, split="validation")
    if val_limit:
        n = min(int(val_limit), len(ds))
        ds = ds.select(range(n))
        logger.info(f"  val sliced to first {n} rows")
    ds = ds.map(lambda ex: _format_chat(ex, tokenizer),
                remove_columns=[c for c in ds.column_names if c != "messages"])
    logger.info(f"  val rows: {len(ds)}")

    # 2. Base model + adapter
    logger.info(f"Loading base {model_id} (BF16)")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto",
    )
    model.config.use_cache = False

    logger.info(f"Loading adapter {adapter_repo}")
    adapter_local = Path("/tmp/phase4-eval-adapter")
    snapshot_download(repo_id=adapter_repo, local_dir=str(adapter_local),
                      repo_type="model", token=token)
    model = PeftModel.from_pretrained(model, str(adapter_local))
    model.eval()

    # 3. Evaluate
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_eval_batch_size=4,
        bf16=True,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds.select(range(1)),
        eval_dataset=ds,
        processing_class=tokenizer,
    )
    metrics = trainer.evaluate()
    logger.info(f"Eval metrics: {metrics}")

    payload = {
        "adapter": adapter_repo,
        "base_model": model_id,
        "dataset": f"{dataset_id}@{dataset_revision}",
        "val_split": "validation",
        "val_rows": len(ds),
        "metrics": metrics,
    }
    out_path = output_dir / result_filename
    out_path.write_text(json.dumps(payload, indent=2, default=str))

    logger.info(f"Pushing {result_filename} → {adapter_repo}")
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(out_path),
        path_in_repo=result_filename,
        repo_id=adapter_repo,
        repo_type="model",
        commit_message=f"Phase 4 Gemma eval-only: {result_filename}",
    )
    logger.info(f"Done: https://huggingface.co/{adapter_repo}/blob/main/{result_filename}")


if __name__ == "__main__":
    main()
