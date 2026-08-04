"""
phase1_data_collection/scripts/validate_samples.py

Validate captured samples for integrity and report statistics.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings import load_config
from phase1_data_collection.scrapers.sqlite_catalog import SQLiteCatalog

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def validate_capture_dir(capture_dir: Path) -> dict:
    """Validate a single capture directory."""
    result = {
        "path": str(capture_dir),
        "valid": True,
        "errors": [],
        "num_screenshots": 0,
        "has_source": False,
        "has_metadata": False,
    }

    # Check source.txt
    source_file = capture_dir / "source.txt"
    if source_file.exists() and source_file.stat().st_size > 0:
        result["has_source"] = True
    else:
        result["valid"] = False
        result["errors"].append("Missing or empty source.txt")

    # Check metadata.json
    metadata_file = capture_dir / "metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file) as f:
                metadata = json.load(f)
            result["has_metadata"] = True
            result["metadata"] = metadata
        except json.JSONDecodeError:
            result["valid"] = False
            result["errors"].append("Invalid metadata.json")
    else:
        result["valid"] = False
        result["errors"].append("Missing metadata.json")

    # Check screenshots
    screenshots = sorted(capture_dir.glob("*.webp"))
    result["num_screenshots"] = len(screenshots)

    if not screenshots:
        result["valid"] = False
        result["errors"].append("No screenshots found")
    else:
        for screenshot in screenshots:
            try:
                with Image.open(screenshot) as img:
                    img.verify()
            except Exception as e:
                result["valid"] = False
                result["errors"].append(f"Corrupt screenshot {screenshot.name}: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Validate captured samples")
    parser.add_argument("--config", default="src/config/config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--captures-dir", default=None,
                        help="Override captures directory")
    parser.add_argument("--fix", action="store_true",
                        help="Remove invalid captures")
    args = parser.parse_args()

    config = load_config(args.config)
    captures_dir = Path(args.captures_dir or config["data_collection"]["captures_dir"])

    if not captures_dir.exists():
        logger.error(f"Captures directory not found: {captures_dir}")
        return

    # Scan all capture directories
    valid_count = 0
    invalid_count = 0
    total_screenshots = 0
    errors_by_type = {}

    for prefix_dir in sorted(captures_dir.iterdir()):
        if not prefix_dir.is_dir():
            continue
        # Handle both flat and worker-prefixed layouts
        for capture_dir in sorted(prefix_dir.iterdir()):
            if not capture_dir.is_dir():
                continue

            # Check if this is a hash directory (has metadata.json)
            if (capture_dir / "metadata.json").exists() or (capture_dir / "source.txt").exists():
                result = validate_capture_dir(capture_dir)
            else:
                # Might be another level (worker dirs)
                for sub_dir in sorted(capture_dir.iterdir()):
                    if sub_dir.is_dir():
                        result = validate_capture_dir(sub_dir)
                        if result["valid"]:
                            valid_count += 1
                            total_screenshots += result["num_screenshots"]
                        else:
                            invalid_count += 1
                            for error in result["errors"]:
                                error_type = error.split(":")[0]
                                errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1
                continue

            if result["valid"]:
                valid_count += 1
                total_screenshots += result["num_screenshots"]
            else:
                invalid_count += 1
                for error in result["errors"]:
                    error_type = error.split(":")[0]
                    errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1

                if args.fix:
                    import shutil
                    shutil.rmtree(capture_dir)
                    logger.info(f"Removed invalid capture: {capture_dir}")

    # Report
    total = valid_count + invalid_count
    logger.info("=" * 60)
    logger.info("VALIDATION REPORT")
    logger.info("=" * 60)
    logger.info(f"Total captures:    {total}")
    logger.info(f"Valid:             {valid_count} ({100*valid_count/max(total,1):.1f}%)")
    logger.info(f"Invalid:           {invalid_count} ({100*invalid_count/max(total,1):.1f}%)")
    logger.info(f"Total screenshots: {total_screenshots}")

    if errors_by_type:
        logger.info("\nErrors by type:")
        for error_type, count in sorted(errors_by_type.items(), key=lambda x: -x[1]):
            logger.info(f"  {error_type}: {count}")

    # Catalog stats
    catalog_db = config["data_collection"]["catalog_db"]
    if Path(catalog_db).exists():
        catalog = SQLiteCatalog(catalog_db)
        stats = catalog.get_statistics()
        logger.info(f"\nCatalog stats:")
        logger.info(f"  Repos:      {stats['total_repos']}")
        logger.info(f"  Captures:   {stats['total_captures']}")
        logger.info(f"  Processed:  {stats['processed_captures']}")
        logger.info(f"  Avg quality: {stats['avg_quality']:.1f}")
        logger.info(f"  By language: {stats['by_language']}")


if __name__ == "__main__":
    main()
