"""Phase 3 HuggingFace Jobs client.

Submits the vision-model training script (`train_entry.py`) as an arbitrary
containerized job via `HfApi.run_job` — not AutoTrain — because the custom
Swin + MLP-projector + Qwen LoRA model isn't expressible as an AutoTrain task.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from huggingface_hub import HfApi
from huggingface_hub._jobs_api import JobStage

logger = logging.getLogger(__name__)


@dataclass
class VisionJobSpec:
    """Spec for a Phase 3 vision-training job."""
    image: str                       # Docker image or `hf.co/spaces/<org>/<space>`
    command: list[str]               # argv-style command
    flavor: str = "a100-large"       # HF Spaces hardware flavor
    env: dict[str, Any] = field(default_factory=dict)
    secrets: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int | None = 36000
    labels: dict[str, str] = field(default_factory=dict)


def submit_vision_job(spec: VisionJobSpec, token: str) -> str:
    """Launch a Phase 3 vision job. Returns job_id."""
    api = HfApi(token=token)
    logger.info(f"Submitting vision job: flavor={spec.flavor} image={spec.image}")
    job = api.run_job(
        image=spec.image,
        command=spec.command,
        env=spec.env or None,
        secrets=spec.secrets or None,
        flavor=spec.flavor,
        timeout=spec.timeout_seconds,
        labels=spec.labels or None,
    )
    logger.info(f"Job launched: {job.id}  {job.url}")
    return job.id


def wait_for_job(
    job_id: str,
    token: str,
    poll_interval: int = 60,
    timeout: int = 36000,
    max_retries: int = 5,
) -> str:
    """Block until a job leaves RUNNING. Returns final JobStage value (e.g. 'COMPLETED')."""
    api = HfApi(token=token)
    terminal = {JobStage.COMPLETED.value, JobStage.CANCELED.value,
                JobStage.ERROR.value, JobStage.DELETED.value}
    start = time.time()
    consecutive_errors = 0
    while True:
        try:
            info = api.inspect_job(job_id=job_id)
            stage = info.status.stage if info.status else "UNKNOWN"
            consecutive_errors = 0
        except (OSError, Exception) as e:
            consecutive_errors += 1
            if consecutive_errors > max_retries:
                logger.error(f"Job {job_id}: {consecutive_errors} consecutive poll failures, giving up: {e}")
                raise
            logger.warning(f"Job {job_id}: poll failed ({consecutive_errors}/{max_retries}), retrying: {e}")
            time.sleep(poll_interval)
            continue
        logger.info(f"Job {job_id} stage: {stage}")
        if stage in terminal:
            if stage != JobStage.COMPLETED.value:
                logger.error(f"Job {job_id} ended in {stage}; last logs:")
                try:
                    for line in api.fetch_job_logs(job_id=job_id):
                        logger.error(line.rstrip())
                except Exception as e:
                    logger.error(f"Could not fetch logs: {e}")
            return stage
        if time.time() - start > timeout:
            logger.error(f"Job {job_id} exceeded {timeout}s; cancelling")
            api.cancel_job(job_id=job_id)
            return "TIMEOUT"
        time.sleep(poll_interval)


__all__ = ["VisionJobSpec", "submit_vision_job", "wait_for_job"]
