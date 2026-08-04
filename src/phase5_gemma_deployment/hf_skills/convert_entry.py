"""
phase5_gemma_deployment/hf_skills/convert_entry.py

Container-side: pull Gemma-4-12B base model + LoRA adapter from Hub, merge
LoRA into base in float16 on CPU, convert merged HF weights to GGUF F16 via
llama.cpp, quantize to the requested levels (default Q4_K_M), then push the
resulting GGUF files to the deployment Hub repo.

Required env vars:
    HF_TOKEN              — read base + adapter, write GGUF repo
    PHASE5_PARAMS_JSON    — { base_model, adapter_repo, gguf_repo, quants[] }
    PHASE5_GGUF_REPO      — also picked up if PARAMS lacks it
"""
import gc
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

os.environ.setdefault("HF_HOME", "/workspace/.hf-cache")

WORK = Path("/workspace/phase5")
LLAMA_DIR = Path("/workspace/llama.cpp")
LLAMA_REPO = "https://github.com/ggerganov/llama.cpp"


def _run(cmd, cwd=None):
    if isinstance(cmd, list):
        printable = " ".join(cmd)
    else:
        printable = cmd
    logger.info(">>> %s", printable)
    subprocess.run(cmd, cwd=cwd, check=True, shell=isinstance(cmd, str))


def _ensure_llama_cpp():
    if (LLAMA_DIR / "build" / "bin" / "llama-quantize").exists():
        logger.info("llama.cpp already built at %s", LLAMA_DIR)
        return
    if not LLAMA_DIR.exists():
        _run(["git", "clone", "--depth", "1", LLAMA_REPO, str(LLAMA_DIR)])
    _run(
        ["cmake", "-B", "build", "-DGGML_CUDA=OFF",
         "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF"],
        cwd=str(LLAMA_DIR),
    )
    _run(
        ["cmake", "--build", "build", "-j", "--target", "llama-quantize"],
        cwd=str(LLAMA_DIR),
    )
    _run([
        "uv", "pip", "install", "-q",
        "--index-strategy", "unsafe-best-match",
        "gguf", "sentencepiece", "protobuf",
    ])


def _merge_adapter(base_model: str, adapter_repo: str, token: str, out_dir: Path):
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = WORK / "adapter"
    if not adapter_dir.exists():
        logger.info("Downloading adapter %s", adapter_repo)
        snapshot_download(
            repo_id=adapter_repo, local_dir=str(adapter_dir),
            repo_type="model", token=token,
        )

    logger.info("Loading base %s in float16 on CPU", base_model)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    logger.info("Applying LoRA adapter %s", adapter_repo)
    merged = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = merged.merge_and_unload()

    logger.info("Saving merged model to %s", out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))

    del merged, base
    gc.collect()


def _convert_to_gguf_f16(merged_dir: Path, out_path: Path):
    convert_script = LLAMA_DIR / "convert_hf_to_gguf.py"
    _run([
        sys.executable, str(convert_script), str(merged_dir),
        "--outfile", str(out_path), "--outtype", "f16",
    ])


def _quantize(f16_path: Path, out_path: Path, quant: str):
    binary = LLAMA_DIR / "build" / "bin" / "llama-quantize"
    _run([str(binary), str(f16_path), str(out_path), quant])


def _upload(local_path: Path, gguf_repo: str, token: str):
    from huggingface_hub import HfApi, create_repo

    create_repo(gguf_repo, token=token, private=False, exist_ok=True)
    api = HfApi(token=token)
    logger.info("Uploading %s (%.1f GB) → %s",
                local_path.name, local_path.stat().st_size / 1e9, gguf_repo)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=local_path.name,
        repo_id=gguf_repo,
        repo_type="model",
        commit_message=f"Phase 5 Gemma: {local_path.name}",
    )


def _render_model_card(base_model, adapter_repo, gguf_files):
    items = "\n".join(f"- `{f}`" for f in gguf_files)
    return f"""---
base_model: {base_model}
tags:
- gguf
- llama-cpp
- quantized
- code-generation
- gemma-4-12b
---

# Code-Trainer — Phase 5 Gemma GGUF

LoRA adapter from [{adapter_repo}](https://hf.co/{adapter_repo}) merged into
{base_model} and quantized via llama.cpp. Use with `llama-cli`, `llama-server`,
Ollama, LM Studio, vLLM (experimental GGUF), or text-generation-webui.

## Files

{items}

## Quick start (llama-server)

```bash
llama-server -m <file>.gguf --host 0.0.0.0 --port 8080 --ctx-size 4096
```
"""


def main():
    params = json.loads(os.environ.get("PHASE5_PARAMS_JSON", "{}"))
    base_model = params.get("base_model", "google/gemma-4-12B-it")
    adapter_repo = params.get("adapter_repo")
    gguf_repo = params.get("gguf_repo") or os.environ.get("PHASE5_GGUF_REPO")
    quants = params.get("quants") or ["Q4_K_M"]

    if not adapter_repo:
        raise RuntimeError("adapter_repo required (set PHASE5_PARAMS_JSON)")
    if not gguf_repo:
        raise RuntimeError("gguf_repo required (set PHASE5_PARAMS_JSON or PHASE5_GGUF_REPO)")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN required")

    WORK.mkdir(parents=True, exist_ok=True)
    name_stem = base_model.split("/")[-1]

    logger.info("=" * 60)
    logger.info("PHASE 5 GEMMA — GGUF Conversion (HF Jobs)")
    logger.info(f"  base:    {base_model}")
    logger.info(f"  adapter: {adapter_repo}")
    logger.info(f"  gguf:    {gguf_repo}")
    logger.info(f"  quants:  {quants}")
    logger.info("=" * 60)

    _ensure_llama_cpp()

    merged_dir = WORK / "merged"
    _merge_adapter(base_model, adapter_repo, token, merged_dir)

    f16_path = WORK / f"{name_stem}-merged-F16.gguf"
    _convert_to_gguf_f16(merged_dir, f16_path)

    shutil.rmtree(merged_dir, ignore_errors=True)

    uploaded = []
    for quant in quants:
        if quant.upper() == "F16":
            _upload(f16_path, gguf_repo, token)
            uploaded.append(f16_path.name)
            continue
        out = WORK / f"{name_stem}-{quant}.gguf"
        _quantize(f16_path, out, quant)
        _upload(out, gguf_repo, token)
        uploaded.append(out.name)
        out.unlink(missing_ok=True)

    card = _render_model_card(base_model, adapter_repo, uploaded)
    readme = WORK / "README.md"
    readme.write_text(card)
    from huggingface_hub import HfApi
    HfApi(token=token).upload_file(
        path_or_fileobj=str(readme),
        path_in_repo="README.md",
        repo_id=gguf_repo,
        repo_type="model",
        commit_message="Phase 5 Gemma: model card",
    )

    f16_path.unlink(missing_ok=True)
    logger.info("Phase 5 Gemma GGUF conversion complete: https://huggingface.co/%s", gguf_repo)


if __name__ == "__main__":
    main()
