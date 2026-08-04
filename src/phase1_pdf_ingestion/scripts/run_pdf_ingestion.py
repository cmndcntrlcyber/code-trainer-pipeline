"""
phase1_pdf_ingestion/scripts/run_pdf_ingestion.py

CLI orchestrator for PDF screenshot ingestion.
Mirrors run_collection.py: accepts a knowledge directory (or a single domain),
renders every PDF page to PNG, extracts text, and records results in catalog.db.

PDFs within each domain are processed in parallel (up to `workers` threads,
default 4). Each worker gets its own PDFPageCapture + PDFTextExtractor instance
so there is no shared mutable state.

Usage:
  # Single domain:
  uv run python -m src.phase1_pdf_ingestion.scripts.run_pdf_ingestion \\
      --config src/config/config.yaml \\
      --input-dir /mnt/ssd/knowledge/Theory \\
      --domain Theory

  # All domains listed in config (sequential domains, parallel PDFs per domain):
  uv run python -m src.phase1_pdf_ingestion.scripts.run_pdf_ingestion \\
      --config src/config/config.yaml \\
      --all-domains

  # Override worker count:
  uv run python -m src.phase1_pdf_ingestion.scripts.run_pdf_ingestion \\
      --config src/config/config.yaml \\
      --all-domains --workers 8
"""
import argparse
import json
import logging
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from src.config.settings import load_config
from src.phase1_pdf_ingestion.pdf_catalog import PDFCatalog
from src.phase1_pdf_ingestion.pdf_renderer import PDFCaptureConfig, PDFPageCapture
from src.phase1_pdf_ingestion.text_extractor import PDFTextExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# catalog writes are serialized across worker threads
_catalog_lock = threading.Lock()


def _process_one_pdf(
    pdf_path: Path,
    domain: str,
    output_dir: Path,
    capture_cfg: PDFCaptureConfig,
) -> dict:
    """
    Worker function: extract text + render pages for a single PDF.
    Returns a result dict; does NOT touch the catalog (caller serializes that).
    Each call creates its own extractor/renderer so there is no shared state.
    """
    extractor = PDFTextExtractor()
    renderer = PDFPageCapture(config=capture_cfg, output_dir=output_dir)

    source_text = extractor.extract_full(pdf_path)
    result = renderer.capture_pdf(
        pdf_path=pdf_path,
        domain=domain,
        source_text=source_text,
        pdf_title=pdf_path.stem,
    )
    return {"pdf_path": pdf_path, "source_text": source_text, "result": result}


def ingest_domain(
    input_dir: Path,
    domain: str,
    output_dir: Path,
    catalog: PDFCatalog,
    config: PDFCaptureConfig,
    workers: int = 4,
) -> dict:
    """Process all PDFs in a single domain directory in parallel. Returns stats dict."""
    pdfs = sorted(input_dir.rglob("*.pdf"))
    logger.info(f"[{domain}] Found {len(pdfs)} PDFs in {input_dir} (workers={workers})")

    total_pages = 0
    total_success = 0
    total_failed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_one_pdf, pdf_path, domain, output_dir, config): pdf_path
            for pdf_path in pdfs
        }
        for future in as_completed(futures):
            pdf_path = futures[future]
            try:
                item = future.result()
            except Exception as exc:
                logger.warning(f"  FAILED {pdf_path.name}: {exc}")
                total_failed += 1
                continue

            result = item["result"]
            source_text = item["source_text"]

            if result.success:
                with _catalog_lock:
                    catalog.add_capture(
                        file_hash=result.file_hash,
                        file_path=pdf_path,
                        domain=domain,
                        num_pages=len(result.screenshots),
                        has_text=bool(source_text.strip()),
                        metadata=result.metadata,
                    )
                total_pages += len(result.screenshots)
                total_success += 1
                logger.info(
                    f"  [{domain}] OK {pdf_path.name} — "
                    f"{len(result.screenshots)} pages, "
                    f"{'has text' if result.metadata['has_text'] else 'image-only'}"
                )
            else:
                total_failed += 1
                logger.warning(f"  [{domain}] FAILED {pdf_path.name}: {result.error}")

    return {
        "domain": domain,
        "pdfs": len(pdfs),
        "success": total_success,
        "failed": total_failed,
        "pages": total_pages,
    }


def _verify_captures(captures_dir: Path) -> list[Path]:
    """Return capture dirs containing at least one unreadable PNG."""
    bad: list[Path] = []
    for meta_path in sorted(captures_dir.rglob("metadata.json")):
        cap_dir = meta_path.parent
        for png in sorted(cap_dir.glob("*.png")):
            try:
                with Image.open(png) as im:
                    im.verify()
            except Exception:
                bad.append(cap_dir)
                break
    return bad


def _rerender_corrupt(
    bad_dirs: list[Path],
    captures_dir: Path,
    capture_cfg: PDFCaptureConfig,
    workers: int,
) -> dict:
    """Delete corrupt capture dirs and re-render from their source PDFs."""
    skipped = 0
    rerendered = 0

    def _rerender_one(cap_dir: Path) -> str:
        try:
            meta = json.loads((cap_dir / "metadata.json").read_text(encoding="utf-8"))
        except Exception as exc:
            return f"SKIP (can't read metadata): {cap_dir}: {exc}"

        pdf_path = Path(meta.get("file_path", ""))
        if not pdf_path.exists():
            return f"SKIP (source missing): {pdf_path}"

        domain = meta.get("domain", "")
        shutil.rmtree(cap_dir, ignore_errors=True)

        extractor = PDFTextExtractor()
        renderer = PDFPageCapture(config=capture_cfg, output_dir=captures_dir)
        source_text = extractor.extract_full(pdf_path)
        result = renderer.capture_pdf(
            pdf_path=pdf_path,
            domain=domain,
            source_text=source_text,
            pdf_title=pdf_path.stem,
        )
        if result.success:
            return f"OK: {pdf_path.name} ({len(result.screenshots)} pages)"
        return f"FAIL: {pdf_path.name}: {result.error}"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_rerender_one, d): d for d in bad_dirs}
        for future in as_completed(futures):
            msg = future.result()
            if msg.startswith("SKIP"):
                skipped += 1
                logger.warning(f"  {msg}")
            elif msg.startswith("FAIL"):
                logger.error(f"  {msg}")
            else:
                rerendered += 1
                logger.info(f"  {msg}")

    return {"bad": len(bad_dirs), "rerendered": rerendered, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs as training screenshots")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--input-dir", help="Path to a single domain PDF directory")
    parser.add_argument("--domain", help="Domain name label (e.g. AI, Rust)")
    parser.add_argument(
        "--all-domains",
        action="store_true",
        help="Ingest all domains listed in config pdf_ingestion.domains",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel PDF workers (overrides config pdf_ingestion.workers)",
    )
    parser.add_argument(
        "--verify-and-rerender",
        action="store_true",
        help="Scan captures dir for unreadable PNGs and re-render those PDFs",
    )
    parser.add_argument(
        "--captures-dir",
        default=None,
        help="Override captures dir for --verify-and-rerender (default: from config)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    pdf_cfg_section = cfg.get("pdf_ingestion", {})

    capture_cfg = PDFCaptureConfig(
        dpi=pdf_cfg_section.get("dpi", 300),
        output_width=pdf_cfg_section.get("output_width", 2560),
    )
    workers = args.workers or pdf_cfg_section.get("workers", 4)
    output_dir = Path(pdf_cfg_section.get("captures_dir", "./data/pdf_captures"))
    catalog_db = Path(cfg.get("data_collection", {}).get("catalog_db", "./data/catalog.db"))

    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_db.parent.mkdir(parents=True, exist_ok=True)
    catalog = PDFCatalog(catalog_db)

    if args.verify_and_rerender:
        captures_dir = Path(args.captures_dir or output_dir)
        logger.info(f"Scanning {captures_dir} for corrupt PNG captures...")
        bad_dirs = _verify_captures(captures_dir)
        if not bad_dirs:
            logger.info("All captures verified — no corrupt PNGs found.")
        else:
            logger.warning(f"Found {len(bad_dirs)} corrupt capture dir(s). Re-rendering...")
            stats = _rerender_corrupt(bad_dirs, captures_dir, capture_cfg, workers)
            logger.info(
                f"\n=== Verify & Re-render Summary ===\n"
                f"  Corrupt dirs found : {stats['bad']}\n"
                f"  Successfully re-rendered: {stats['rerendered']}\n"
                f"  Skipped (source PDF missing): {stats['skipped']}"
            )
        catalog.close()
        return

    if args.all_domains:
        knowledge_dir = Path(pdf_cfg_section.get("knowledge_dir", "/mnt/ssd/knowledge"))
        domains: list[str] = pdf_cfg_section.get("domains", [])
        if not domains:
            logger.error("No domains configured under pdf_ingestion.domains")
            sys.exit(1)
        all_stats = []
        for domain in domains:
            domain_dir = knowledge_dir / domain
            if not domain_dir.exists():
                logger.warning(f"Domain directory not found, skipping: {domain_dir}")
                continue
            stats = ingest_domain(domain_dir, domain, output_dir, catalog, capture_cfg, workers)
            all_stats.append(stats)
        _print_summary(all_stats)

    elif args.input_dir and args.domain:
        stats = ingest_domain(
            Path(args.input_dir), args.domain, output_dir, catalog, capture_cfg, workers
        )
        _print_summary([stats])

    else:
        parser.error("Provide --all-domains OR both --input-dir and --domain")

    catalog.close()


def _print_summary(stats: list[dict]) -> None:
    logger.info("\n=== PDF Ingestion Summary ===")
    total_pdfs = total_pages = total_ok = total_fail = 0
    for s in stats:
        logger.info(
            f"  {s['domain']:30s}  PDFs: {s['pdfs']:4d}  "
            f"OK: {s['success']:4d}  Failed: {s['failed']:3d}  "
            f"Pages: {s['pages']:6d}"
        )
        total_pdfs += s["pdfs"]
        total_pages += s["pages"]
        total_ok += s["success"]
        total_fail += s["failed"]
    logger.info(
        f"  {'TOTAL':30s}  PDFs: {total_pdfs:4d}  OK: {total_ok:4d}  "
        f"Failed: {total_fail:3d}  Pages: {total_pages:6d}"
    )


if __name__ == "__main__":
    main()
