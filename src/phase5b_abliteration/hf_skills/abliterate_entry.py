"""
phase5b_abliteration/hf_skills/abliterate_entry.py

Container-side: merge base+LoRA (or download base directly when no adapter),
run 3 abliteration techniques, benchmark all 3 + control, convert to GGUF,
upload everything to Hub.

Supports both post-fine-tuning (adapter_repo set) and pre-fine-tuning baseline
(adapter_repo null/absent) abliteration runs.

Runs on a100-large for GPU (activation caching) + RAM (model merge).

Required env vars:
    HF_TOKEN                  — read base + adapter, write output repos
    PHASE5B_PARAMS_JSON       — { base_model, adapter_repo (or null),
                                  output_base, techniques[], evaluation{},
                                  quants[] }
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

WORK = Path("/workspace/phase5b")
LLAMA_DIR = Path("/workspace/llama.cpp")
LLAMA_REPO = "https://github.com/ggerganov/llama.cpp"


def _run(cmd, cwd=None):
    if isinstance(cmd, list):
        printable = " ".join(cmd)
    else:
        printable = cmd
    logger.info(">>> %s", printable)
    subprocess.run(cmd, cwd=cwd, check=True, shell=isinstance(cmd, str))


def _install_dependencies():
    """Install abliteration technique packages (cloud-only).

    All packages installed with --no-deps to prevent pulling in a kernels
    version that breaks transformers 5.7.0 (LayerRepository init error).
    The venv already has transformers, torch, peft, datasets from uv sync.
    """
    logger.info("Installing abliteration dependencies...")
    _run([
        "uv", "pip", "install", "-q",
        "--index-strategy", "unsafe-best-match",
        "--no-deps",
        "obliteratus", "abliterix", "lm-eval",
    ])


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


def _merge_adapter(base_model: str, adapter_repo: str, token: str, out_dir: Path,
                    adapter_chain: list[str] | None = None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading base %s in float16 on CPU", base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, device_map="cpu",
        low_cpu_mem_usage=True, token=token,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=token)

    all_adapters = list(adapter_chain or []) + [adapter_repo]
    for adapter_id in all_adapters:
        logger.info("Merging adapter: %s", adapter_id)
        model = PeftModel.from_pretrained(model, adapter_id, token=token)
        model = model.merge_and_unload()

    logger.info("Saving merged model to %s", out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))

    del model
    gc.collect()


def _download_base_model(base_model: str, token: str, out_dir: Path):
    """Download a base model directly (no adapter merge) for baseline abliteration."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Downloading base model %s in float16 on CPU (no adapter)", base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.float16, device_map="cpu",
        low_cpu_mem_usage=True, token=token,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=token)

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))

    del model
    gc.collect()
    logger.info("Base model saved to %s", out_dir)


def _convert_to_gguf(merged_dir: Path, out_path: Path, quant: str):
    convert_script = LLAMA_DIR / "convert_hf_to_gguf.py"
    f16_path = WORK / "temp_f16.gguf"

    _run([
        sys.executable, str(convert_script), str(merged_dir),
        "--outfile", str(f16_path), "--outtype", "f16",
    ])

    if quant.upper() == "F16":
        shutil.move(str(f16_path), str(out_path))
    else:
        binary = LLAMA_DIR / "build" / "bin" / "llama-quantize"
        _run([str(binary), str(f16_path), str(out_path), quant])
        f16_path.unlink(missing_ok=True)


def _upload(local_path: Path, repo_id: str, token: str, commit_msg: str):
    from huggingface_hub import HfApi, create_repo

    create_repo(repo_id, token=token, private=False, exist_ok=True)
    api = HfApi(token=token)
    logger.info("Uploading %s (%.1f GB) → %s",
                local_path.name, local_path.stat().st_size / 1e9, repo_id)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=local_path.name,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_msg,
    )


def main():
    params = json.loads(os.environ.get("PHASE5B_PARAMS_JSON", "{}"))
    base_model = params.get("base_model", "Qwen/Qwen2.5-Coder-14B-Instruct")
    adapter_repo = params.get("adapter_repo")
    output_base = params.get("output_base")
    techniques_cfg = params.get("techniques", [])
    eval_cfg = params.get("evaluation", {})
    quants = params.get("quants", ["Q4_K_M"])

    if not output_base:
        raise RuntimeError("output_base required")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN required")

    WORK.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PHASE 5b — Abliteration Benchmarking (HF Jobs)")
    logger.info(f"  base:       {base_model}")
    logger.info(f"  adapter:    {adapter_repo or '(none — baseline run)'}")
    logger.info(f"  output:     {output_base}")
    logger.info(f"  techniques: {[t.get('name') for t in techniques_cfg]}")
    logger.info(f"  quants:     {quants}")
    logger.info("=" * 60)

    _install_dependencies()
    _ensure_llama_cpp()

    # 1. Merge LoRA → HF weights (or download base directly for baseline runs)
    merged_dir = WORK / "merged"
    if not merged_dir.exists():
        if adapter_repo and str(adapter_repo).lower() != "null":
            adapter_chain = params.get("adapter_chain", [])
            _merge_adapter(base_model, adapter_repo, token, merged_dir,
                            adapter_chain=adapter_chain)
        else:
            _download_base_model(base_model, token, merged_dir)
    else:
        logger.info("Merged model already exists at %s", merged_dir)

    # 2. Run abliteration techniques
    from src.phase5b_abliteration.techniques import TECHNIQUES

    abliterated_dirs = {}
    for tech_cfg in techniques_cfg:
        tech_name = tech_cfg["name"]
        if tech_name not in TECHNIQUES:
            logger.warning("Unknown technique %s, skipping", tech_name)
            continue

        tech_class = TECHNIQUES[tech_name]
        technique = tech_class()
        output_dir = WORK / f"abliterated_{tech_name}"

        if output_dir.exists() and any(output_dir.glob("*.safetensors")):
            logger.info("%s: already abliterated, skipping", tech_name)
            abliterated_dirs[tech_name] = output_dir
            continue

        logger.info("Running technique: %s", tech_name)
        try:
            technique.abliterate(merged_dir, output_dir, tech_cfg)
            abliterated_dirs[tech_name] = output_dir
            logger.info("%s: abliteration complete", tech_name)
        except Exception:
            logger.exception("%s: abliteration FAILED", tech_name)

        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 3. Evaluate all variants
    from src.phase5b_abliteration.evaluation.evaluator import AbliterationEvaluator

    full_config = {"evaluation": eval_cfg}
    evaluator = AbliterationEvaluator(full_config, WORK)
    summary = evaluator.evaluate_all(abliterated_dirs, merged_dir)

    logger.info("=" * 60)
    logger.info("RANKING:")
    for i, row in enumerate(summary["ranking"], 1):
        logger.info(
            "  %d. %s — refusal=%.2f%% kl=%.4f",
            i,
            row["variant"],
            row.get("refusal_rate", -1) * 100,
            row.get("kl_divergence", -1),
        )
    logger.info("=" * 60)

    # 4. Convert each abliterated variant to GGUF + upload
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    for tech_name, abl_dir in abliterated_dirs.items():
        gguf_repo = f"{output_base}-{tech_name}-abliterated-gguf"

        for quant in quants:
            gguf_path = WORK / f"{tech_name}_{quant}.gguf"
            logger.info("Converting %s to GGUF %s...", tech_name, quant)
            try:
                _convert_to_gguf(abl_dir, gguf_path, quant)
                _upload(gguf_path, gguf_repo, token,
                        f"Phase 5b: {tech_name} abliterated {quant}")
                gguf_path.unlink(missing_ok=True)
            except Exception:
                logger.exception("GGUF conversion failed for %s/%s", tech_name, quant)

    # 5. Upload evaluation results
    results_repo = f"{output_base}-abliteration-results"
    from huggingface_hub import create_repo
    create_repo(results_repo, token=token, private=False, exist_ok=True)

    results_dir = WORK / "evaluation_results"
    for json_file in results_dir.glob("*.json"):
        api.upload_file(
            path_or_fileobj=str(json_file),
            path_in_repo=json_file.name,
            repo_id=results_repo,
            repo_type="model",
            commit_message=f"Phase 5b: {json_file.stem}",
        )

    logger.info(
        "Phase 5b abliteration complete. Results: https://huggingface.co/%s",
        results_repo,
    )


if __name__ == "__main__":
    main()
