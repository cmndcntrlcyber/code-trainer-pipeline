"""
OBLITERATUS abliteration technique wrapper.

Uses the OBLITERATUS AbliterationPipeline with the 'advanced' preset:
4-direction SVD + norm-preserve + bias projection. Operates on native HF
transformers — no TransformerLens dependency.

Ref: https://github.com/elder-plinius/OBLITERATUS
"""
import gc
import logging
from pathlib import Path

import torch

from .base import AbliterationTechnique

logger = logging.getLogger(__name__)


class ObliteratusTechnique(AbliterationTechnique):
    name = "obliteratus"

    def get_dependencies(self) -> list[str]:
        return ["obliteratus"]

    def abliterate(
        self, model_dir: Path, output_dir: Path, config: dict
    ) -> Path:
        self.validate_model_dir(model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        preset = config.get("preset", "advanced")
        n_directions = config.get("n_directions", 4)
        norm_preserve = config.get("norm_preserve", True)

        logger.info(
            "OBLITERATUS: preset=%s, n_directions=%d, norm_preserve=%s",
            preset, n_directions, norm_preserve,
        )

        from obliteratus import AbliterationPipeline
        from obliteratus.config import ModelConfig, StrategyConfig

        model_cfg = ModelConfig(
            model_name_or_path=str(model_dir),
            torch_dtype="bfloat16",
            device_map="auto",
            trust_remote_code=True,
        )

        strategy_cfg = StrategyConfig(
            preset=preset,
            n_directions=n_directions,
            norm_preserve=norm_preserve,
        )

        pipeline = AbliterationPipeline(
            model_config=model_cfg,
            strategy_config=strategy_cfg,
        )

        logger.info("OBLITERATUS: running abliteration pipeline...")
        result = pipeline.run()

        logger.info("OBLITERATUS: saving abliterated model to %s", output_dir)
        pipeline.save(str(output_dir))

        if hasattr(result, "metrics"):
            logger.info("OBLITERATUS metrics: %s", result.metrics)

        del pipeline, result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output_dir
