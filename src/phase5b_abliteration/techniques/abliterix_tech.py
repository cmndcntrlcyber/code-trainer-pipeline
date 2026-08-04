"""
abliterix technique wrapper.

Multi-objective Optuna TPE optimization with 9 steering methods.
Default: SRA (Surgical Refusal Ablation) for best capability preservation.
Can output direct weight modification or LoRA adapter.

Ref: https://github.com/wuwangzhang1216/abliterix
"""
import gc
import json
import logging
import subprocess
import sys
from pathlib import Path

import torch

from .base import AbliterationTechnique

logger = logging.getLogger(__name__)


class AbliterixTechnique(AbliterationTechnique):
    name = "abliterix"

    def get_dependencies(self) -> list[str]:
        return ["abliterix"]

    def abliterate(
        self, model_dir: Path, output_dir: Path, config: dict
    ) -> Path:
        self.validate_model_dir(model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        steering_method = config.get("steering_method", "SRA")
        optuna_trials = config.get("optuna_trials", 50)

        logger.info(
            "abliterix: steering_method=%s, optuna_trials=%d",
            steering_method, optuna_trials,
        )

        try:
            return self._run_via_python_api(
                model_dir, output_dir, steering_method, optuna_trials, config
            )
        except (ImportError, AttributeError):
            logger.info("abliterix: Python API unavailable, falling back to CLI")
            return self._run_via_cli(
                model_dir, output_dir, steering_method, optuna_trials
            )

    def _run_via_python_api(
        self,
        model_dir: Path,
        output_dir: Path,
        steering_method: str,
        optuna_trials: int,
        config: dict,
    ) -> Path:
        """Run abliterix via its Python API."""
        import abliterix

        logger.info("abliterix: running via Python API...")
        result = abliterix.abliterate(
            model=str(model_dir),
            output=str(output_dir),
            steering_method=steering_method,
            n_trials=optuna_trials,
            save_model=True,
        )

        metadata = {
            "technique": "abliterix",
            "steering_method": steering_method,
            "optuna_trials": optuna_trials,
        }
        if hasattr(result, "best_params"):
            metadata["best_params"] = result.best_params
        if hasattr(result, "metrics"):
            metadata["metrics"] = result.metrics

        (output_dir / "abliteration_metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str)
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output_dir

    def _run_via_cli(
        self,
        model_dir: Path,
        output_dir: Path,
        steering_method: str,
        optuna_trials: int,
    ) -> Path:
        """Fall back to abliterix CLI."""
        cmd = [
            sys.executable, "-m", "abliterix",
            "--model", str(model_dir),
            "--output", str(output_dir),
            "--steering-method", steering_method,
            "--n-trials", str(optuna_trials),
            "--save-model",
        ]
        logger.info("abliterix CLI: %s", " ".join(cmd))
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=7200
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"abliterix CLI failed (rc={result.returncode}):\n{result.stderr}"
            )

        metadata = {
            "technique": "abliterix",
            "steering_method": steering_method,
            "optuna_trials": optuna_trials,
            "ran_via": "cli",
        }
        (output_dir / "abliteration_metadata.json").write_text(
            json.dumps(metadata, indent=2)
        )

        return output_dir
