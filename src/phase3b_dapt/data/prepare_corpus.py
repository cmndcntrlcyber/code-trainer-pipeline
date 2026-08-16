"""
phase3b_dapt/data/prepare_corpus.py

Prepare the code pretraining corpus for Domain-Adaptive Pre-Training (DAPT).
Walks data/offensive-security/repositories/, extracts qualifying code files,
deduplicates by SHA256, tokenizes and chunks into max_seq_length documents,
and saves as a HuggingFace dataset with a "text" column.

Usage:
    python -m src.phase3b_dapt.data.prepare_corpus \
        --config src/config/config.yaml \
        --output-dir data/dapt_corpus

    # Push to Hub:
    python -m src.phase3b_dapt.data.prepare_corpus \
        --config src/config/config.yaml \
        --output-dir data/dapt_corpus \
        --push-to-hub
"""
import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.config.settings import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".py", ".c", ".cpp", ".cs", ".go", ".rs",
    ".js", ".ts", ".ps1", ".sh", ".bash",
}

SKIP_DIRS = {
    "vendor", "node_modules", "__pycache__",
    "test", "tests", "build", "dist",
    ".git", ".svn", ".venv", "venv",
    "target", "bin", "obj",
}


def _is_valid_file(path: Path, min_lines: int, max_lines: int) -> bool:
    """Check if a file meets extension and line-count requirements."""
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
        line_count = content.count("\n") + 1
        return min_lines <= line_count <= max_lines
    except (OSError, PermissionError):
        return False


def collect_files(
    corpus_dir: Path, min_lines: int, max_lines: int
) -> list[Path]:
    """Walk corpus_dir and return qualifying code files, skipping excluded dirs."""
    files = []
    for root, dirs, filenames in os.walk(corpus_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in filenames:
            fpath = Path(root) / fname
            if _is_valid_file(fpath, min_lines, max_lines):
                files.append(fpath)
    return files


def deduplicate(files: list[Path]) -> list[Path]:
    """Deduplicate files by SHA256 of content. Returns unique file list."""
    seen: set[str] = set()
    unique: list[Path] = []
    for fpath in files:
        try:
            content = fpath.read_bytes()
        except (OSError, PermissionError):
            continue
        digest = hashlib.sha256(content).hexdigest()
        if digest not in seen:
            seen.add(digest)
            unique.append(fpath)
    return unique


def chunk_text(text: str, tokenizer, max_seq_length: int) -> list[str]:
    """Tokenize text and split into chunks of max_seq_length tokens.

    Each chunk is decoded back to a string for the dataset "text" column.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(token_ids), max_seq_length):
        chunk_ids = token_ids[i : i + max_seq_length]
        if len(chunk_ids) < 64:
            # Skip very short trailing chunks
            continue
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=False)
        chunks.append(chunk_text)
    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Prepare DAPT code corpus from offensive-security repositories"
    )
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--output-dir", default="data/dapt_corpus")
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    dapt_cfg = config.get("dapt", {})

    min_lines = int(dapt_cfg.get("min_lines", 10))
    max_lines = int(dapt_cfg.get("max_lines", 1000))
    max_seq_length = int(dapt_cfg.get("max_seq_length", 4096))
    base_model = dapt_cfg.get("base_model", "Qwen/Qwen2.5-Coder-14B-Instruct")

    corpus_dir = Path("data/offensive-security/repositories")
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect qualifying files
    logger.info("Collecting code files from %s (lines: %d-%d)", corpus_dir, min_lines, max_lines)
    files = collect_files(corpus_dir, min_lines, max_lines)
    logger.info("Found %d qualifying files", len(files))

    # 2. Deduplicate
    files = deduplicate(files)
    logger.info("After dedup: %d unique files", len(files))

    if not files:
        logger.warning("No files to process. Exiting.")
        return

    # 3. Load tokenizer
    from transformers import AutoTokenizer

    logger.info("Loading tokenizer: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # 4. Read, tokenize, and chunk
    all_chunks: list[str] = []
    for fpath in files:
        try:
            text = fpath.read_text(encoding="utf-8", errors="ignore")
        except (OSError, PermissionError):
            continue
        chunks = chunk_text(text, tokenizer, max_seq_length)
        all_chunks.extend(chunks)

    logger.info("Total documents (chunks): %d", len(all_chunks))

    # 5. Build HuggingFace dataset
    from datasets import Dataset

    ds = Dataset.from_dict({"text": all_chunks})
    ds.save_to_disk(str(output_dir))
    logger.info("Dataset saved to %s (%d rows)", output_dir, len(ds))

    # 6. Optionally push to Hub
    if args.push_to_hub:
        hub_name = dapt_cfg.get("output_adapter", "").replace("-adapter", "") + "-corpus"
        if not hub_name or hub_name == "-corpus":
            hub_name = "dapt-offsec-corpus"
        logger.info("Pushing dataset to Hub: %s", hub_name)
        ds.push_to_hub(hub_name, private=True)
        logger.info("Dataset pushed to Hub: %s", hub_name)


if __name__ == "__main__":
    main()
