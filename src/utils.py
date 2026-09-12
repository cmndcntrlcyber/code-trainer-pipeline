"""Shared utilities for the RTPI training pipeline."""
import logging

logger = logging.getLogger(__name__)


def unwrap_clippable_linear(model):
    """Unwrap Gemma4ClippableLinear -> nn.Linear for PEFT compatibility.

    Gemma 4 models use ClippableLinear wrappers that PEFT doesn't recognize.
    Call this immediately after loading the base model, before any
    PeftModel.from_pretrained or get_peft_model calls.

    No-op for non-Gemma models (ImportError is caught).
    """
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
        count = 0
        for name, module in list(model.named_modules()):
            if isinstance(module, Gemma4ClippableLinear):
                parts = name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], module.linear)
                count += 1
        if count:
            logger.info("Unwrapped %d Gemma4ClippableLinear modules for PEFT compatibility", count)
    except ImportError:
        pass
    return model
