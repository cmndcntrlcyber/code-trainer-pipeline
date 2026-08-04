"""
phase1_data_collection/scripts/run_collection.py

Main orchestrator for Phase 1 data collection pipeline.
Flow: scrape repos -> filter files -> parallel capture -> update catalog
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings import load_config
from phase1_data_collection.scrapers.sqlite_catalog import SQLiteCatalog
from phase1_data_collection.scrapers.github_scraper import GitHubScraper
from phase1_data_collection.scrapers.file_filter import FileFilter
from phase1_data_collection.capture.vscode_automation import CaptureConfig
from phase1_data_collection.capture.screenshot_manager import ScreenshotManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data_collection.log")
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Code-Trainer Data Collection")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--repos-per-language", type=int, default=None,
                        help="Override repos per language (default: from config)")
    parser.add_argument("--skip-scraping", action="store_true",
                        help="Skip GitHub scraping, use existing cloned repos")
    parser.add_argument("--skip-capture", action="store_true",
                        help="Skip screenshot capture, only scrape repos")
    args = parser.parse_args()

    config = load_config(args.config)
    dc = config["data_collection"]

    repos_per_language = args.repos_per_language or dc["repos_per_language"]

    # Initialize catalog
    catalog = SQLiteCatalog(dc["catalog_db"])
    logger.info(f"Catalog initialized: {dc['catalog_db']}")

    repos_dir = Path(dc["repos_dir"])
    captures_dir = Path(dc["captures_dir"])
    repos_dir.mkdir(parents=True, exist_ok=True)
    captures_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1A: Scrape and clone repositories
    repo_paths = []
    if not args.skip_scraping:
        logger.info(f"Scraping {repos_per_language} repos/language across {len(dc['languages'])} languages")

        scraper = GitHubScraper(
            token=dc["github_token"],
            output_dir=repos_dir,
            catalog=catalog,
            languages=dc["languages"],
            min_quality_score=dc["min_quality_score"]
        )

        for repo_path in scraper.collect_repositories(repos_per_language=repos_per_language):
            repo_paths.append(repo_path)

        logger.info(f"Cloned {len(repo_paths)} repositories")
    else:
        # Use existing repos
        if repos_dir.exists():
            repo_paths = [p for p in repos_dir.iterdir() if p.is_dir()]
            logger.info(f"Using {len(repo_paths)} existing repositories")
        else:
            logger.error(f"Repos directory not found: {repos_dir}")
            return

    if args.skip_capture:
        logger.info("Skipping capture phase (--skip-capture)")
        stats = catalog.get_statistics()
        logger.info(f"Catalog stats: {stats}")
        return

    # Phase 1B: Filter files and capture screenshots
    capture_config = CaptureConfig(
        viewport_width=dc.get("viewport_width", 2560),
        viewport_height=dc.get("viewport_height", 1440),
        font_size=dc.get("font_size", 14),
        theme=dc.get("theme", "Default Dark+")
    )

    screenshot_mgr = ScreenshotManager(
        catalog=catalog,
        output_dir=captures_dir,
        config=capture_config,
        num_workers=dc.get("capture_workers", 8),
        rotate_themes=True
    )

    total_files = 0
    total_captured = 0

    for repo_path in repo_paths:
        repo_name = repo_path.name
        files = list(FileFilter.get_code_files(repo_path))

        if not files:
            logger.debug(f"No valid files in {repo_name}")
            continue

        total_files += len(files)
        logger.info(f"Processing {repo_name}: {len(files)} files")

        results = screenshot_mgr.capture_files(files, repo_name)
        captured = sum(1 for r in results if r.success)
        total_captured += captured

    # Final stats
    stats = catalog.get_statistics()
    logger.info(f"Collection complete: {total_captured}/{total_files} files captured")
    logger.info(f"Catalog: {stats['total_repos']} repos, {stats['total_captures']} captures")
    logger.info(f"By language: {stats['by_language']}")
    logger.info(f"By category: {stats['by_category']}")
    logger.info(f"Avg quality: {stats['avg_quality']:.1f}")


if __name__ == "__main__":
    main()
