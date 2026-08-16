"""
phase3b_dapt/data/extract_pdf_text.py

Extract text from PDF captures for DAPT corpus building.
Walks data/pdf_captures/ directories, extracts text from PDFs (via
pdfplumber) or reads pre-extracted text files, filters to readable text,
chunks into max_seq_length token documents, and saves as a HuggingFace
dataset with a "text" column.

Usage:
    python -m src.phase3b_dapt.data.extract_pdf_text \
        --config src/config/config.yaml \
        --output-dir data/dapt_pdf_corpus

    # Push to Hub:
    python -m src.phase3b_dapt.data.extract_pdf_text \
        --config src/config/config.yaml \
        --output-dir data/dapt_pdf_corpus \
        --push-to-hub
"""
import argparse
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

MIN_CHARS_PER_PAGE = 50


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF file using pdfplumber. Falls back to PyPDF2."""
    pages: list[str] = []

    try:
        import pdfplumber

        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if len(text.strip()) >= MIN_CHARS_PER_PAGE:
                    pages.append(text)
        if pages:
            return "\n\n".join(pages)
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("pdfplumber failed for %s: %s", pdf_path, exc)

    # Fallback to PyPDF2
    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(str(pdf_path))
        for page in reader.pages:
            text = page.extract_text() or ""
            if len(text.strip()) >= MIN_CHARS_PER_PAGE:
                pages.append(text)
        if pages:
            return "\n\n".join(pages)
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("PyPDF2 failed for %s: %s", pdf_path, exc)

    return ""


def collect_texts(captures_dir: Path) -> list[str]:
    """Walk captures_dir and extract text from PDFs and text files."""
    texts: list[str] = []

    for root, _dirs, files in os.walk(captures_dir):
        for fname in files:
            fpath = Path(root) / fname

            if fname.lower().endswith(".pdf"):
                text = extract_text_from_pdf(fpath)
                if text and len(text.strip()) >= MIN_CHARS_PER_PAGE:
                    texts.append(text)
                    logger.debug("Extracted %d chars from %s", len(text), fpath)

            elif fname.lower().endswith(".txt"):
                try:
                    text = fpath.read_text(encoding="utf-8", errors="ignore")
                    if len(text.strip()) >= MIN_CHARS_PER_PAGE:
                        texts.append(text)
                except (OSError, PermissionError):
                    continue

    return texts


def chunk_text(text: str, tokenizer, max_seq_length: int) -> list[str]:
    """Tokenize text and split into chunks of max_seq_length tokens."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(token_ids), max_seq_length):
        chunk_ids = token_ids[i : i + max_seq_length]
        if len(chunk_ids) < 64:
            continue
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=False)
        chunks.append(chunk_text)
    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Extract text from PDF captures for DAPT corpus"
    )
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--output-dir", default="data/dapt_pdf_corpus")
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    dapt_cfg = config.get("dapt", {})

    max_seq_length = int(dapt_cfg.get("max_seq_length", 4096))
    base_model = dapt_cfg.get("base_model", "Qwen/Qwen2.5-Coder-14B-Instruct")

    captures_dir = Path("data/pdf_captures")
    if not captures_dir.exists():
        raise FileNotFoundError(f"PDF captures directory not found: {captures_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect texts from PDFs and text files
    logger.info("Collecting text from %s", captures_dir)
    texts = collect_texts(captures_dir)
    logger.info("Extracted text from %d documents", len(texts))

    if not texts:
        logger.warning("No extractable text found. Exiting.")
        return

    # 2. Load tokenizer
    from transformers import AutoTokenizer

    logger.info("Loading tokenizer: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # 3. Tokenize and chunk
    all_chunks: list[str] = []
    for text in texts:
        chunks = chunk_text(text, tokenizer, max_seq_length)
        all_chunks.extend(chunks)

    logger.info("Total documents (chunks): %d", len(all_chunks))

    # 4. Build HuggingFace dataset
    from datasets import Dataset

    ds = Dataset.from_dict({"text": all_chunks})
    ds.save_to_disk(str(output_dir))
    logger.info("Dataset saved to %s (%d rows)", output_dir, len(ds))

    # 5. Optionally push to Hub
    if args.push_to_hub:
        hub_name = dapt_cfg.get("output_adapter", "").replace("-adapter", "") + "-pdf-corpus"
        if not hub_name or hub_name == "-pdf-corpus":
            hub_name = "dapt-offsec-pdf-corpus"
        logger.info("Pushing dataset to Hub: %s", hub_name)
        ds.push_to_hub(hub_name, private=True)
        logger.info("Dataset pushed to Hub: %s", hub_name)


if __name__ == "__main__":
    main()
