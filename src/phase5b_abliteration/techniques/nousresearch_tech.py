"""
NousResearch/llm-abliteration technique wrapper.

Diff-of-means refusal direction extraction with norm-preserving biprojected
ablation. Uses native HF transformers — no TransformerLens dependency.
Supports 4-bit/8-bit quantization via BitsAndBytes for lower VRAM usage.

Ref: https://github.com/NousResearch/llm-abliteration
"""
import gc
import json
import logging
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import AbliterationTechnique

logger = logging.getLogger(__name__)

HARMFUL_DATASET = "Undi95/orthogonal-activation-steering-TOXIC"
HARMLESS_DATASET = "tatsu-lab/alpaca"


def _load_instructions(dataset_id: str, n: int, split: str = "train") -> list[str]:
    """Load instruction strings from a HF dataset."""
    try:
        ds = load_dataset(dataset_id, split=split)
    except ValueError:
        for fallback in ("test", "validation"):
            try:
                ds = load_dataset(dataset_id, split=fallback)
                logger.info("Split '%s' not in %s, using '%s'", split, dataset_id, fallback)
                break
            except ValueError:
                continue
        else:
            raise
    if "instruction" in ds.column_names:
        col = "instruction"
    elif "prompt" in ds.column_names:
        col = "prompt"
    elif "text" in ds.column_names:
        col = "text"
    else:
        col = ds.column_names[0]
    instructions = [row[col] for row in ds if row[col] and row[col].strip()]
    return instructions[:n]


def _get_last_token_activations(
    model, tokenizer, instructions: list[str], batch_size: int = 8
) -> torch.Tensor:
    """Run instructions through model, collect last-token hidden states per layer."""
    all_activations = []
    for i in range(0, len(instructions), batch_size):
        batch = instructions[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        attention_mask = inputs["attention_mask"]
        seq_lengths = attention_mask.sum(dim=1) - 1

        per_sample = []
        for s in range(len(batch)):
            last_pos = seq_lengths[s].item()
            layer_acts = torch.stack(
                [hidden_states[layer][s, last_pos] for layer in range(len(hidden_states))]
            )
            per_sample.append(layer_acts)
        all_activations.append(torch.stack(per_sample))

    return torch.cat(all_activations, dim=0)


class NousResearchTechnique(AbliterationTechnique):
    name = "nousresearch"

    def get_dependencies(self) -> list[str]:
        return ["git+https://github.com/NousResearch/llm-abliteration.git"]

    def abliterate(
        self, model_dir: Path, output_dir: Path, config: dict
    ) -> Path:
        self.validate_model_dir(model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        n_samples = config.get("n_samples", 512)
        method = config.get("method", "biprojected")
        batch_size = 8

        logger.info(
            "NousResearch: method=%s, n_samples=%d", method, n_samples,
        )

        logger.info("NousResearch: loading model from %s", model_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        logger.info("NousResearch: loading contrastive datasets...")
        harmful = _load_instructions(HARMFUL_DATASET, n_samples)
        harmless = _load_instructions(HARMLESS_DATASET, n_samples)
        logger.info(
            "NousResearch: %d harmful, %d harmless instructions loaded",
            len(harmful), len(harmless),
        )

        logger.info("NousResearch: caching harmful activations...")
        harmful_acts = _get_last_token_activations(
            model, tokenizer, harmful, batch_size
        )
        logger.info("NousResearch: caching harmless activations...")
        harmless_acts = _get_last_token_activations(
            model, tokenizer, harmless, batch_size
        )

        n_layers = harmful_acts.shape[1]
        logger.info("NousResearch: computing refusal directions across %d layers", n_layers)

        harmful_mean = harmful_acts.mean(dim=0)
        harmless_mean = harmless_acts.mean(dim=0)
        refusal_dirs = harmful_mean - harmless_mean

        norms = refusal_dirs.norm(dim=-1, keepdim=True)
        refusal_dirs_normed = refusal_dirs / norms.clamp(min=1e-8)

        del harmful_acts, harmless_acts
        gc.collect()

        best_layer = _select_best_layer(refusal_dirs_normed, norms)
        logger.info("NousResearch: best refusal layer = %d (norm=%.4f)", best_layer, norms[best_layer].item())

        logger.info("NousResearch: applying %s ablation...", method)
        _apply_ablation(model, refusal_dirs_normed, method)

        logger.info("NousResearch: saving abliterated model to %s", output_dir)
        model.save_pretrained(str(output_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(output_dir))

        metadata = {
            "technique": "nousresearch",
            "method": method,
            "n_samples": n_samples,
            "best_layer": best_layer,
            "best_layer_norm": norms[best_layer].item(),
            "n_layers_modified": n_layers,
        }
        (output_dir / "abliteration_metadata.json").write_text(
            json.dumps(metadata, indent=2)
        )

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output_dir


def _select_best_layer(
    refusal_dirs: torch.Tensor, norms: torch.Tensor
) -> int:
    """Select the layer with the strongest refusal signal (highest norm)."""
    skip_first = 1
    best = norms[skip_first:].argmax().item() + skip_first
    return int(best)


def _apply_ablation(
    model, refusal_dirs: torch.Tensor, method: str = "biprojected"
) -> None:
    """
    Orthogonalize refusal direction out of residual-stream-writing weights.

    For each transformer layer, modifies W_O (attn output) and W_out (MLP
    down_proj). With biprojected method, restores the original Frobenius
    norm after projection to preserve magnitude.
    """
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError(
            f"Unsupported architecture: {type(model).__name__} — expected "
            f"model.model.layers (LlamaForCausalLM, Qwen2ForCausalLM, "
            f"Gemma2ForCausalLM, etc.). Found attributes: "
            f"{[a for a in dir(model) if not a.startswith('_')][:20]}"
        )

    n_layers = len(model.model.layers)

    for layer_idx in range(1, n_layers):
        layer = model.model.layers[layer_idx]
        r_hat = refusal_dirs[layer_idx + 1].to(layer.self_attn.o_proj.weight.device)

        attn_proj = getattr(layer.self_attn, "o_proj", None)
        mlp_proj = getattr(layer.mlp, "down_proj", None)
        if attn_proj is None or mlp_proj is None:
            raise RuntimeError(
                f"Layer {layer_idx}: expected self_attn.o_proj and mlp.down_proj "
                f"but found attn attrs={[a for a in dir(layer.self_attn) if 'proj' in a]}, "
                f"mlp attrs={[a for a in dir(layer.mlp) if 'proj' in a]}"
            )

        for proj_name in ["o_proj", "down_proj"]:
            if proj_name == "o_proj":
                module = attn_proj
            elif proj_name == "down_proj":
                module = mlp_proj
            else:
                continue

            W = module.weight.data
            orig_norm = W.norm().item()

            proj = torch.outer(W @ r_hat, r_hat)
            W.sub_(proj)

            if method == "biprojected":
                new_norm = W.norm().item()
                if new_norm > 0:
                    W.mul_(orig_norm / new_norm)
