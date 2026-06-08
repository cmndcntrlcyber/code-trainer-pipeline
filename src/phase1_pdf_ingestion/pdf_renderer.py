"""
phase1_pdf_ingestion/pdf_renderer.py

Renders PDF pages to PNG screenshots using pypdfium2 (no system deps).
Mirrors the output structure of MonacoCapture:
  data/pdf_captures/<hash2>/<hash>/
    0000.png, 0001.png, ...
    source.txt
    metadata.json
"""
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pypdfium2 as pdfium
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class PDFCaptureConfig:
    dpi: int = 300
    output_width: int = 2560
    fmt: str = "PNG"


@dataclass
class PDFCaptureResult:
    pdf_path: Path
    screenshots: list[Path] = field(default_factory=list)
    source_text: str = ""
    file_hash: str = ""
    metadata: dict = field(default_factory=dict)
    success: bool = False
    error: Optional[str] = None


class PDFPageCapture:
    """Renders PDF pages to PNG files in the Phase 1 capture directory structure."""

    def __init__(self, config: PDFCaptureConfig, output_dir: Path):
        self.config = config
        self.output_dir = output_dir

    def capture_pdf(
        self,
        pdf_path: Path,
        domain: str,
        source_text: str = "",
        pdf_title: str = "",
    ) -> PDFCaptureResult:
        """
        Render all pages of a PDF to PNGs and write the capture directory.

        Args:
            pdf_path: Absolute path to the PDF file.
            domain: Knowledge folder name (e.g., "AI", "Rust").
            source_text: Pre-extracted text (from PDFTextExtractor). May be empty
                         for scanned/image-only PDFs.
            pdf_title: PDF document title metadata, if available.

        Returns:
            PDFCaptureResult with populated fields on success.
        """
        result = PDFCaptureResult(pdf_path=pdf_path)
        try:
            file_hash = _hash_file(pdf_path)
            cap_dir = self.output_dir / file_hash[:2] / file_hash
            cap_dir.mkdir(parents=True, exist_ok=True)

            screenshots = self._render_pages(pdf_path, cap_dir)
            if not screenshots:
                result.error = "No pages rendered"
                return result

            (cap_dir / "source.txt").write_text(source_text, encoding="utf-8")

            metadata = {
                "file_path": str(pdf_path),
                "file_hash": file_hash,
                "source_type": "pdf",
                "pdf_title": pdf_title or pdf_path.stem,
                "num_pages": len(screenshots),
                "num_screenshots": len(screenshots),
                "domain": domain,
                "dpi": self.config.dpi,
                "output_width": self.config.output_width,
                "has_text": bool(source_text.strip()),
                "language": "text",
            }
            (cap_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )

            result.screenshots = screenshots
            result.source_text = source_text
            result.file_hash = file_hash
            result.metadata = metadata
            result.success = True
            return result

        except Exception as exc:
            logger.error(f"Failed to capture {pdf_path}: {exc}")
            result.error = str(exc)
            return result

    def _render_pages(self, pdf_path: Path, cap_dir: Path) -> list[Path]:
        """Render each PDF page to a PNG file. Returns list of written paths."""
        doc = pdfium.PdfDocument(str(pdf_path))
        paths: list[Path] = []
        scale = self.config.dpi / 72.0  # pdfium native unit is 1/72 inch

        try:
            for page_idx, page in enumerate(doc):
                bitmap = page.render(scale=scale, rotation=0)
                pil_img = bitmap.to_pil()

                if pil_img.width != self.config.output_width:
                    ratio = self.config.output_width / pil_img.width
                    new_height = int(pil_img.height * ratio)
                    pil_img = pil_img.resize(
                        (self.config.output_width, new_height), Image.LANCZOS
                    )

                out_path = cap_dir / f"{page_idx:04d}.png"
                pil_img.save(out_path, format="PNG", optimize=True)
                paths.append(out_path)
                page.close()
        finally:
            doc.close()

        return paths


def _hash_file(path: Path) -> str:
    """SHA256 of file contents, return first 16 hex chars."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
