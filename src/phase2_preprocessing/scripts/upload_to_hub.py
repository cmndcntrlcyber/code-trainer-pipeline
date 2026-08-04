"""
phase2_preprocessing/scripts/upload_to_hub.py

Upload the locally-built HuggingFace dataset to HF Hub.

Usage:
    python -m src.phase2_preprocessing.scripts.upload_to_hub \
        --config src/config/config.yaml \
        --dataset-dir data/hf_dataset

    # Upload as public repo:
    python -m src.phase2_preprocessing.scripts.upload_to_hub \
        --config src/config/config.yaml --public

    # Upload multimodal build to a named branch/revision:
    python -m src.phase2_preprocessing.scripts.upload_to_hub \
        --config src/config/config.yaml \
        --dataset-dir data/hf_dataset_multimodal \
        --revision v2-multimodal --public
"""
import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from datasets import load_from_disk
from huggingface_hub import HfApi, create_branch

from src.config.settings import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Upload Phase 2 dataset to HF Hub")
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--dataset-dir", default="data/hf_dataset",
                        help="Path to the locally-saved HF dataset")
    parser.add_argument("--public", action="store_true",
                        help="Upload as public repo (default: private)")
    parser.add_argument("--revision", default=None,
                        help="Named branch/revision to push to "
                             "(default: preprocessing.multimodal_revision from config, or main)")
    parser.add_argument("--commit-message", default=None,
                        help="Override the default commit message")
    args = parser.parse_args()

    config = load_config(args.config)
    pp_cfg = config.get("preprocessing", {})
    dataset_name = pp_cfg.get("dataset_name")
    private = not args.public and pp_cfg.get("private", True)
    revision = args.revision  # explicit CLI takes precedence; None = push to default (main)

    if not dataset_name:
        logger.error("preprocessing.dataset_name not set in config")
        sys.exit(1)

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        logger.error("Run build_dataset.py first.")
        sys.exit(1)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    logger.info(f"Loading dataset from {dataset_dir}")
    dataset = load_from_disk(str(dataset_dir))

    # Ensure target branch exists before pushing
    if revision:
        logger.info(f"Ensuring branch exists: {dataset_name}@{revision}")
        create_branch(
            repo_id=dataset_name,
            repo_type="dataset",
            branch=revision,
            token=token,
            exist_ok=True,
        )

    commit_message = args.commit_message or (
        f"Phase 2: multimodal build ({revision})"
        if revision
        else "Phase 2: Upload code-trainer screenshot dataset"
    )

    logger.info(
        f"Uploading to HF Hub: {dataset_name} "
        f"(private={private}, revision={revision or 'main'})"
    )
    push_kwargs = dict(private=private, commit_message=commit_message)
    if revision:
        push_kwargs["revision"] = revision
    dataset.push_to_hub(dataset_name, **push_kwargs)

    branch_suffix = f"/tree/{revision}" if revision else ""
    logger.info(
        f"Dataset uploaded: https://huggingface.co/datasets/{dataset_name}{branch_suffix}"
    )

    # Upload statistics.json as a dataset card artifact
    stats_path = dataset_dir / "statistics.json"
    if stats_path.exists():
        api = HfApi(token=token)
        upload_kwargs = dict(
            path_or_fileobj=str(stats_path),
            path_in_repo="statistics.json",
            repo_id=dataset_name,
            repo_type="dataset",
            commit_message="Add dataset statistics",
        )
        if revision:
            upload_kwargs["revision"] = revision
        api.upload_file(**upload_kwargs)
        logger.info("statistics.json uploaded")


if __name__ == "__main__":
    main()
