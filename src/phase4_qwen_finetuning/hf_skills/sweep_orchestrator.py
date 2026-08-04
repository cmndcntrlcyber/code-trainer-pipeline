"""
phase4_qwen_finetuning/hf_skills/sweep_orchestrator.py

Orchestrates parallel launch of 3 validation sweep jobs on HF Skills A100-large.
Waits for all to complete, ranks by eval_loss, returns best configs.
"""
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .job_client import HFSkillsClient, JobSpec, JobStatus
from ..configs.sweep_configs import SWEEP_CONFIGS, SweepConfig, LORA_TARGET_MODULES

logger = logging.getLogger(__name__)


class SweepOrchestrator:
    """
    Launch all sweep configs in parallel, wait for completion, rank results.

    Usage:
        orch = SweepOrchestrator(client, dataset_id, model_id, results_dir)
        results = orch.run_sweep()
        best = results[0]  # lowest eval_loss
    """

    def __init__(
        self,
        client: HFSkillsClient,
        dataset_id: str,
        model_id: str = "Qwen/Qwen2.5-Coder-14B-Instruct",
        hardware: str = "a100-large",
        results_dir: Path = Path("data/sweep_results"),
    ):
        self.client = client
        self.dataset_id = dataset_id
        self.model_id = model_id
        self.hardware = hardware
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _build_job_spec(self, cfg: SweepConfig) -> JobSpec:
        return JobSpec(
            name=f"phase4-sweep-{cfg.name}",
            model_id=self.model_id,
            dataset_id=self.dataset_id,
            hardware=self.hardware,
            script_path="src/phase4_qwen_finetuning/scripts/train_qwen.py",
            num_epochs=1,
            hyperparams={
                "lora_r": cfg.lora_r,
                "lora_alpha": cfg.lora_alpha,
                "learning_rate": cfg.learning_rate,
                "per_device_train_batch_size": cfg.batch_size,
                "gradient_accumulation_steps": cfg.gradient_accumulation,
                "target_modules": ",".join(LORA_TARGET_MODULES),
                "bf16": True,
                "gradient_checkpointing": True,
                "report_to": "wandb",
                "wandb_project": "code-trainer-phase4",
                "run_name": f"sweep-{cfg.name}",
            },
        )

    def _launch_and_wait(self, cfg: SweepConfig) -> dict[str, Any]:
        """Launch a single sweep job and wait for completion."""
        spec = self._build_job_spec(cfg)
        try:
            job_id = self.client.launch_job(spec)
            logger.info(f"[{cfg.name}] Launched job {job_id}")
            status = self.client.wait_for_completion(job_id, poll_interval=60)
            result = {
                "config_name": cfg.name,
                "job_id": job_id,
                "status": status.status,
                "config": {
                    "lora_r": cfg.lora_r,
                    "lora_alpha": cfg.lora_alpha,
                    "learning_rate": cfg.learning_rate,
                    "batch_size": cfg.batch_size,
                    "effective_batch": cfg.effective_batch,
                },
                "eval_loss": status.eval_loss,
                "logs_url": status.logs_url,
            }
        except Exception as e:
            logger.error(f"[{cfg.name}] Job failed: {e}")
            result = {
                "config_name": cfg.name,
                "status": "error",
                "error": str(e),
                "eval_loss": float("inf"),
            }

        # Save individual result
        out = self.results_dir / f"sweep_{cfg.name}.json"
        out.write_text(json.dumps(result, indent=2))
        return result

    def run_sweep(self, configs: list[SweepConfig] | None = None) -> list[dict[str, Any]]:
        """
        Launch all sweep configs in parallel and wait for all to complete.

        Returns:
            List of result dicts sorted by eval_loss (best first).
        """
        configs = configs or SWEEP_CONFIGS
        logger.info(f"Launching {len(configs)} parallel sweep jobs...")

        results = []
        with ThreadPoolExecutor(max_workers=len(configs)) as pool:
            futures = {pool.submit(self._launch_and_wait, cfg): cfg for cfg in configs}
            for future in as_completed(futures):
                cfg = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(
                        f"[{cfg.name}] Done — eval_loss: {result.get('eval_loss', 'N/A')}"
                    )
                except Exception as e:
                    logger.error(f"[{cfg.name}] Exception: {e}")

        # Sort by eval_loss (ascending)
        results.sort(key=lambda r: r.get("eval_loss") or float("inf"))

        # Save summary
        summary = {"sweep_results": results, "best": results[0] if results else None}
        (self.results_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2))

        logger.info("Sweep complete. Rankings:")
        for i, r in enumerate(results):
            logger.info(f"  #{i+1} {r['config_name']}: eval_loss={r.get('eval_loss', 'N/A')}")

        return results
