"""
Per-model-family chat template registry for Ollama Modelfile generation.

Maps base model family identifiers to their Go-template TEMPLATE blocks
and stop token sequences.
"""

from __future__ import annotations


GEMMA4_TEMPLATE = """\
{{ if .System }}<start_of_turn>user
{{ .System }}<end_of_turn>
{{ end }}{{ range .Messages }}{{ if eq .Role "user" }}<start_of_turn>user
{{ .Content }}<end_of_turn>
{{ else if eq .Role "assistant" }}<start_of_turn>model
{{ .Content }}<end_of_turn>
{{ else if eq .Role "tool" }}<start_of_turn>tool
{{ .Content }}<end_of_turn>
{{ end }}{{ end }}<start_of_turn>model
"""

QWEN25_TEMPLATE = """\
{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ range .Messages }}{{ if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ .Content }}<|im_end|>
{{ else if eq .Role "tool" }}<|im_start|>tool
{{ .Content }}<|im_end|>
{{ end }}{{ end }}<|im_start|>assistant
"""

QWEN3_TEMPLATE = QWEN25_TEMPLATE  # Qwen3 shares the ChatML format.


CHAT_TEMPLATES: dict[str, dict] = {
    "gemma4": {
        "template": GEMMA4_TEMPLATE,
        "stop_tokens": ["<start_of_turn>", "<end_of_turn>"],
        "llama_server_template": "gemma",
    },
    "qwen25": {
        "template": QWEN25_TEMPLATE,
        "stop_tokens": ["<|im_start|>", "<|im_end|>"],
        "llama_server_template": "chatml",
    },
    "qwen3": {
        "template": QWEN3_TEMPLATE,
        "stop_tokens": ["<|im_start|>", "<|im_end|>"],
        "llama_server_template": "chatml",
    },
}


def detect_model_family(base_model: str) -> str:
    """Detect the chat template family from a HF model ID.

    >>> detect_model_family("Qwen/Qwen2.5-Coder-14B-Instruct")
    'qwen25'
    >>> detect_model_family("google/gemma-4-26B-A4B-it")
    'gemma4'
    >>> detect_model_family("Qwen/Qwen3-8B")
    'qwen3'
    """
    lower = base_model.lower()
    if "gemma" in lower:
        return "gemma4"
    if "qwen3" in lower or "qwen/qwen3" in lower:
        return "qwen3"
    if "qwen" in lower:
        return "qwen25"
    return "qwen25"  # default fallback


def get_template(family: str) -> dict:
    """Return the template dict for a model family."""
    if family not in CHAT_TEMPLATES:
        raise ValueError(
            f"Unknown model family {family!r}. "
            f"Available: {sorted(CHAT_TEMPLATES)}"
        )
    return CHAT_TEMPLATES[family]
