"""
src/scripts/sync_model_cards.py

Upload local model-card sources from `docs/model_cards/<stem>.md` to the
matching Hugging Face Hub repos as `README.md`. Idempotent — re-runnable
after edits.

Mapping is derived from the filename stem: `<stem>.md` → `<HF_USERNAME>/<stem>`.
Override with --user to push to a different namespace.

Usage:
    set -a && source .env && set +a
    python -m src.scripts.sync_model_cards --dry-run
    python -m src.scripts.sync_model_cards
    python -m src.scripts.sync_model_cards --only qwen14b-code-trainer-aggressive
"""
import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

CARDS_DIR = Path(__file__).resolve().parents[2] / "docs" / "model_cards"


def main():
    parser = argparse.ArgumentParser(
        description="Sync docs/model_cards/*.md → Hub README.md")
    parser.add_argument("--user", default=os.environ.get("HF_USERNAME"),
                        help="HF namespace. Default: $HF_USERNAME.")
    parser.add_argument("--cards-dir", default=str(CARDS_DIR),
                        help="Directory containing the .md sources.")
    parser.add_argument("--only", default=None,
                        help="Only sync this card stem (filename without .md).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the file→repo mapping and exit.")
    args = parser.parse_args()

    if not args.user:
        raise SystemExit("HF_USERNAME not set (and --user not supplied)")

    cards = sorted(Path(args.cards_dir).glob("*.md"))
    if args.only:
        cards = [c for c in cards if c.stem == args.only]
    if not cards:
        raise SystemExit(f"No model-card sources found under {args.cards_dir}")

    mapping = [(c, f"{args.user}/{c.stem}") for c in cards]
    logger.info("Sync plan:")
    for src, repo in mapping:
        logger.info(f"  {src.name}  →  {repo}/README.md")

    if args.dry_run:
        logger.info("--dry-run set — not uploading.")
        return

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN env var required to push")

    from huggingface_hub import HfApi
    api = HfApi(token=token)

    failures = []
    for src, repo in mapping:
        try:
            logger.info(f"Uploading {src.name} → {repo}")
            api.upload_file(
                path_or_fileobj=str(src),
                path_in_repo="README.md",
                repo_id=repo,
                repo_type="model",
                commit_message=f"Sync model card from docs/model_cards/{src.name}",
            )
        except Exception as e:
            logger.error(f"Failed to push {repo}: {e}")
            failures.append((repo, str(e)))

    if failures:
        logger.error("%d card(s) failed to upload:", len(failures))
        for repo, err in failures:
            logger.error("  %s — %s", repo, err)
        sys.exit(1)
    logger.info("All %d card(s) synced.", len(mapping))


if __name__ == "__main__":
    main()
