"""
phase2_preprocessing/scripts/build_dataset.py

Full Phase 2 preprocessing pipeline:
  1. Scan Phase 1 capture directories
  2. Load source code + metadata
  3. Apply quality filtering & deduplication
  4. Format into Qwen chat messages
  5. Encode screenshots to base64 WebP
  6. Split into train/validation/test
  7. Build HuggingFace Dataset and save to disk

Usage:
    uv run python -m src.phase2_preprocessing.scripts.build_dataset \\
        --config src/config/config.yaml \\
        --captures-dir data/sample-data/captures \\
        --output-dir data/hf_dataset

    # Skip image encoding (text-only dataset, faster):
    uv run python -m src.phase2_preprocessing.scripts.build_dataset \\
        --config src/config/config.yaml --no-images

    # Include PDF knowledge captures alongside code captures:
    uv run python -m src.phase2_preprocessing.scripts.build_dataset \\
        --config src/config/config.yaml --include-pdf

    # PDF only (no code captures):
    uv run python -m src.phase2_preprocessing.scripts.build_dataset \\
        --config src/config/config.yaml --include-pdf --no-code
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from datasets import Dataset, DatasetDict

from src.config.settings import load_config
from src.phase2_preprocessing.converters.hf_dataset_converter import (
    convert_captures_to_records,
    convert_pdf_captures_to_records,
    split_records,
)
from src.phase2_preprocessing.converters.image_encoder import (
    encode_png_to_base64_webp,
    encode_capture_dir,
)
from src.phase2_preprocessing.validation.quality_filter import filter_records
from src.phase2_preprocessing.validation.statistics import (
    compute_statistics,
    log_statistics,
    save_statistics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def encode_images_in_records(
    records: list[dict],
    show_progress: bool = True,
    max_dim: int = 1024,
) -> list[dict]:
    """
    Encode screenshots for all records in-place. Drops records whose image fails.
    Handles both code records (cap_dir) and PDF records (page_png).
    """
    out = []
    for i, rec in enumerate(records):
        try:
            if rec.get("source_type") == "pdf":
                rec["image"] = encode_png_to_base64_webp(
                    Path(rec["page_png"]), max_dim=max_dim
                )
            else:
                rec["image"] = encode_capture_dir(Path(rec["cap_dir"]), max_dim=max_dim)
            out.append(rec)
        except Exception as e:
            label = rec.get("page_png") or rec.get("cap_dir", "?")
            logger.warning(f"Skipping {Path(label).name}: image encode failed: {e}")
        if show_progress and (i + 1) % 500 == 0:
            logger.info(f"  Encoded {i + 1}/{len(records)} images")
    return out


def records_to_hf_dataset(records: list[dict], include_images: bool) -> Dataset:
    """Convert a list of record dicts to a HuggingFace Dataset.

    Unified schema handles both code and PDF records — fields absent in one
    source type are filled with sensible defaults so the schema stays flat.
    """
    data: dict[str, list] = {
        "file_hash":   [],
        "language":    [],
        "source_type": [],   # "code" | "pdf"
        "source_code": [],
        "prompt_idx":  [],
        "messages":    [],
        # code-only fields (default 0/"" for PDF records)
        "line_count":  [],
        "theme":       [],
        # pdf-only fields (default 0/"" for code records)
        "domain":      [],
        "page_idx":    [],
        "num_pages":   [],
    }
    if include_images:
        data["image"] = []

    for rec in records:
        is_pdf = rec.get("source_type") == "pdf"
        data["file_hash"].append(rec["file_hash"])
        data["language"].append(rec["language"])
        data["source_type"].append("pdf" if is_pdf else "code")
        data["source_code"].append(rec["source_code"])
        data["prompt_idx"].append(rec["prompt_idx"])
        data["messages"].append(rec["messages"])
        data["line_count"].append(0 if is_pdf else rec.get("line_count", 0))
        data["theme"].append("" if is_pdf else rec.get("theme", ""))
        data["domain"].append(rec.get("domain", ""))
        data["page_idx"].append(rec.get("page_idx", 0))
        data["num_pages"].append(rec.get("num_pages", 0))
        if include_images:
            data["image"].append(rec.get("image", ""))

    return Dataset.from_dict(data)


def main():
    parser = argparse.ArgumentParser(description="Build Phase 2 HuggingFace dataset")
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--captures-dir", default=None,
                        help="Override code captures directory from config")
    parser.add_argument("--pdf-captures-dir", default=None,
                        help="Override PDF captures directory from config")
    parser.add_argument("--output-dir", default="data/hf_dataset",
                        help="Directory to save the HF dataset")
    parser.add_argument("--no-images", action="store_true",
                        help="Skip image encoding (faster, text-only dataset)")
    parser.add_argument("--include-pdf", action="store_true",
                        help="Merge PDF knowledge captures with code captures")
    parser.add_argument("--include-repos", action="store_true",
                        help="Merge GitHub Stars repo captures (repo_ingestion.captures_dir) with code captures")
    parser.add_argument("--no-code", action="store_true",
                        help="Exclude code captures (use with --include-pdf for PDF-only)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit total records (for testing)")
    parser.add_argument("--image-max-dim", type=int, default=None,
                        help="Max pixel dimension for WebP images "
                             "(default: preprocessing.image_max_dim in config, or 1024)")
    args = parser.parse_args()

    if args.no_code and not args.include_pdf:
        parser.error("--no-code requires --include-pdf")

    config = load_config(args.config)
    pp_cfg = config.get("preprocessing", {})
    pdf_cfg = config.get("pdf_ingestion", {})
    repo_cfg = config.get("repo_ingestion", {})

    captures_dir = Path(args.captures_dir or config["data_collection"]["captures_dir"])
    pdf_captures_dir = Path(args.pdf_captures_dir or pdf_cfg.get("captures_dir", "data/pdf_captures"))
    repo_captures_dir = Path(repo_cfg.get("captures_dir", "data/captures"))
    output_dir = Path(args.output_dir)
    max_code_length = pp_cfg.get("max_code_length", 8192)
    train_ratio = pp_cfg.get("train_split", 0.8)
    val_ratio = pp_cfg.get("val_split", 0.1)
    image_max_dim = args.image_max_dim or pp_cfg.get("image_max_dim", 1024)

    logger.info(
        f"Phase 2: images={'off' if args.no_images else 'on'}  "
        f"code={'off' if args.no_code else captures_dir}  "
        f"pdf={'on → ' + str(pdf_captures_dir) if args.include_pdf else 'off'}  "
        f"repos={'on → ' + str(repo_captures_dir) if args.include_repos else 'off'}"
    )

    # Step 1: Load records
    records: list[dict] = []

    if not args.no_code:
        code_records = convert_captures_to_records(
            captures_dir, max_code_length=max_code_length
        )
        for r in code_records:
            r.setdefault("source_type", "code")
        records.extend(code_records)
        logger.info(f"Loaded {len(code_records)} code capture records")

    if args.include_pdf:
        if not pdf_captures_dir.exists():
            logger.error(f"PDF captures dir not found: {pdf_captures_dir}")
            sys.exit(1)
        pdf_records = convert_pdf_captures_to_records(
            pdf_captures_dir, max_text_length=max_code_length
        )
        records.extend(pdf_records)
        logger.info(f"Loaded {len(pdf_records)} PDF page records")

    if args.include_repos:
        if not repo_captures_dir.exists():
            logger.warning(f"Repo captures dir not found: {repo_captures_dir} — skipping")
        else:
            repo_records = convert_captures_to_records(
                repo_captures_dir, max_code_length=max_code_length
            )
            for r in repo_records:
                r.setdefault("source_type", "code")
            records.extend(repo_records)
            logger.info(f"Loaded {len(repo_records)} repo capture records from {repo_captures_dir}")

    if not records:
        logger.error("No records loaded — nothing to build.")
        sys.exit(1)

    if args.limit:
        records = records[: args.limit]
        logger.info(f"Limited to {len(records)} records")

    # Step 2: Quality filter (code records only — PDF records skip the code filter)
    code_recs = [r for r in records if r.get("source_type") != "pdf"]
    pdf_recs  = [r for r in records if r.get("source_type") == "pdf"]
    if code_recs:
        code_recs, filter_stats = filter_records(code_recs)
    records = code_recs + pdf_recs

    # Step 3: Encode images
    if not args.no_images:
        logger.info(f"Encoding screenshots to base64 WebP (max_dim={image_max_dim})...")
        records = encode_images_in_records(records, max_dim=image_max_dim)

    # Step 4: Split
    splits = split_records(records, train_ratio=train_ratio, val_ratio=val_ratio)
    logger.info(
        f"Split: train={len(splits['train'])}  val={len(splits['validation'])}  "
        f"test={len(splits['test'])}"
    )

    # Step 5: Build HF DatasetDict
    dataset_dict = DatasetDict({
        split: records_to_hf_dataset(recs, include_images=not args.no_images)
        for split, recs in splits.items()
    })

    # Step 6: Save to disk
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))
    logger.info(f"Dataset saved to {output_dir}")

    # Step 7: Statistics
    stats = compute_statistics(splits)
    log_statistics(stats)
    save_statistics(stats, output_dir / "statistics.json")

    logger.info("Phase 2 build complete.")


if __name__ == "__main__":
    main()
