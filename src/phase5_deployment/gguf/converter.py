"""
phase5_deployment/gguf/converter.py

GGUF conversion pipeline: merge LoRA → convert to GGUF F16 → quantize.

Prerequisites:
    - llama.cpp built with CUDA sm_120 support (for Blackwell RTX 5060 Ti)
    - ~30GB free disk space for intermediate files

Steps:
    1. Download fine-tuned LoRA adapter from HF Hub
    2. Merge LoRA with base Qwen2.5-Coder-14B (CPU, float16)
    3. Convert merged model to GGUF F16
    4. Quantize to Q4_K_M (~9 GB)
"""
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


class GGUFConverter:
    """
    Orchestrates the full LoRA → GGUF → quantize pipeline.

    Args:
        llama_cpp_dir: Path to compiled llama.cpp directory
        work_dir:      Scratch space for intermediate files (~30GB needed)
    """

    def __init__(
        self,
        llama_cpp_dir: str | Path,
        work_dir: str | Path = Path("data/gguf_work"),
    ):
        self.llama_cpp_dir = Path(llama_cpp_dir)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self._validate_llama_cpp()

    def _validate_llama_cpp(self):
        convert_script = self.llama_cpp_dir / "convert_hf_to_gguf.py"
        quantize_bin = self.llama_cpp_dir / "llama-quantize"
        if not convert_script.exists():
            raise FileNotFoundError(
                f"llama.cpp convert script not found: {convert_script}\n"
                f"Build llama.cpp with: cmake -B build -DGGML_CUDA=ON && cmake --build build -j"
            )
        if not quantize_bin.exists():
            raise FileNotFoundError(f"llama-quantize not found: {quantize_bin}")

    def download_adapter(self, hf_repo_id: str, local_dir: Path | None = None) -> Path:
        """Download LoRA adapter from HF Hub."""
        local_dir = local_dir or self.work_dir / "lora_adapter"
        logger.info(f"Downloading adapter: {hf_repo_id}")
        snapshot_download(repo_id=hf_repo_id, local_dir=str(local_dir))
        return local_dir

    def download_base_model(
        self,
        model_id: str = "Qwen/Qwen2.5-Coder-14B-Instruct",
    ) -> Path:
        """Download base model from HF Hub (CPU only)."""
        local_dir = self.work_dir / "base_model"
        if local_dir.exists() and any(local_dir.glob("*.safetensors")):
            logger.info(f"Base model already downloaded: {local_dir}")
            return local_dir
        logger.info(f"Downloading base model: {model_id} (large download ~30GB)")
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            ignore_patterns=["*.gguf"],
        )
        return local_dir

    def merge_lora(self, base_model_dir: Path, adapter_dir: Path) -> Path:
        """Merge LoRA adapter into base model weights (CPU, float16)."""
        merged_dir = self.work_dir / "merged_model"
        merged_dir.mkdir(exist_ok=True)

        logger.info("Merging LoRA into base model (CPU, float16)...")
        script = f"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "{base_model_dir}",
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base, "{adapter_dir}")
merged = model.merge_and_unload()
merged.save_pretrained("{merged_dir}", safe_serialization=True)
tok = AutoTokenizer.from_pretrained("{base_model_dir}", trust_remote_code=True)
tok.save_pretrained("{merged_dir}")
print("Merge complete:", "{merged_dir}")
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=3600
        )
        if result.returncode != 0:
            raise RuntimeError(f"Merge failed:\n{result.stderr}")
        logger.info(f"Merged model saved to {merged_dir}")
        return merged_dir

    def convert_to_gguf_f16(self, merged_dir: Path) -> Path:
        """Convert merged HF model to GGUF F16 format."""
        gguf_f16 = self.work_dir / "model_f16.gguf"
        convert_script = self.llama_cpp_dir / "convert_hf_to_gguf.py"

        logger.info("Converting to GGUF F16...")
        result = subprocess.run(
            [
                sys.executable, str(convert_script),
                str(merged_dir),
                "--outfile", str(gguf_f16),
                "--outtype", "f16",
            ],
            capture_output=True, text=True, timeout=3600
        )
        if result.returncode != 0:
            raise RuntimeError(f"GGUF conversion failed:\n{result.stderr}")
        logger.info(f"GGUF F16 saved: {gguf_f16} ({gguf_f16.stat().st_size / 1e9:.1f} GB)")
        return gguf_f16

    def quantize(self, gguf_f16: Path, quant_type: str = "Q5_K_M") -> Path:
        """Quantize GGUF F16 to target quantization."""
        quantize_bin = self.llama_cpp_dir / "llama-quantize"
        output = self.work_dir / f"model_{quant_type.lower()}.gguf"

        logger.info(f"Quantizing to {quant_type}...")
        result = subprocess.run(
            [str(quantize_bin), str(gguf_f16), str(output), quant_type],
            capture_output=True, text=True, timeout=7200
        )
        if result.returncode != 0:
            raise RuntimeError(f"Quantization failed:\n{result.stderr}")
        logger.info(f"{quant_type} GGUF saved: {output} ({output.stat().st_size / 1e9:.1f} GB)")
        return output

    def run_full_pipeline(
        self,
        adapter_repo_id: str,
        base_model_id: str = "Qwen/Qwen2.5-Coder-14B-Instruct",
        quant_type: str = "Q5_K_M",
        output_path: Path | None = None,
    ) -> Path:
        """
        Run the complete LoRA → GGUF pipeline.

        Returns:
            Path to the final quantized GGUF file
        """
        adapter_dir = self.download_adapter(adapter_repo_id)
        base_dir = self.download_base_model(base_model_id)
        merged_dir = self.merge_lora(base_dir, adapter_dir)
        gguf_f16 = self.convert_to_gguf_f16(merged_dir)
        gguf_quantized = self.quantize(gguf_f16, quant_type)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(gguf_quantized, output_path)
            logger.info(f"Final GGUF: {output_path}")
            return output_path

        return gguf_quantized
