"""
Abstract base class for abliteration techniques.

Each technique wrapper loads a merged HF model, applies its abliteration
method, and saves the modified weights to an output directory.
"""
import logging
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


class AbliterationTechnique(ABC):
    """Base class for all abliteration technique wrappers."""

    name: str

    @abstractmethod
    def abliterate(
        self, model_dir: Path, output_dir: Path, config: dict
    ) -> Path:
        """
        Apply abliteration to a merged HF model.

        Args:
            model_dir:  Path to the merged HF model (safetensors + tokenizer).
            output_dir: Where to save the abliterated model weights.
            config:     Technique-specific config dict from config.yaml.

        Returns:
            Path to the output directory containing the abliterated model.
        """

    @abstractmethod
    def get_dependencies(self) -> list[str]:
        """Pip install specs needed beyond pyproject.toml."""

    def validate_model_dir(self, model_dir: Path) -> None:
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        safetensors = list(model_dir.glob("*.safetensors"))
        if not safetensors:
            raise FileNotFoundError(
                f"No safetensors files in {model_dir}"
            )
        logger.info(
            "%s: validated model dir %s (%d safetensors files)",
            self.name, model_dir, len(safetensors),
        )
