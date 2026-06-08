"""
phase1_pdf_ingestion/text_extractor.py

Extracts text from PDFs page-by-page using pdfminer.six.
Falls back to empty string for image-only (scanned) pages.
"""
import logging
from io import StringIO
from pathlib import Path

from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams

logger = logging.getLogger(__name__)


class PDFTextExtractor:
    """Extract text from PDF files, one string per page."""

    def __init__(self, laparams: LAParams | None = None):
        self._laparams = laparams or LAParams()

    def extract_pages(self, pdf_path: Path) -> list[str]:
        """
        Return a list of text strings, one per page.
        Pages with no extractable text (scanned images) return empty string.
        """
        import pdfminer.high_level as hl

        pages: list[str] = []
        try:
            for page_text in hl.extract_pages(str(pdf_path), laparams=self._laparams):
                buf = StringIO()
                # Collect LTChar/LTAnon text from the page layout
                text = _layout_to_text(page_text)
                pages.append(text)
        except Exception as exc:
            logger.warning(f"Text extraction failed for {pdf_path}: {exc}")
        return pages

    def extract_full(self, pdf_path: Path) -> str:
        """Return all pages joined with form-feed separators."""
        output = StringIO()
        try:
            extract_text_to_fp(
                open(str(pdf_path), "rb"),
                output,
                laparams=self._laparams,
                output_type="text",
                codec="utf-8",
            )
            return output.getvalue()
        except Exception as exc:
            logger.warning(f"Full text extraction failed for {pdf_path}: {exc}")
            return ""


def _layout_to_text(page_layout) -> str:
    """Flatten an LTPage layout object to a plain text string."""
    from pdfminer.layout import LTAnno, LTChar, LTTextContainer

    parts: list[str] = []
    for element in page_layout:
        if isinstance(element, LTTextContainer):
            for text_line in element:
                for char in text_line:
                    if isinstance(char, (LTChar, LTAnno)):
                        parts.append(char.get_text())
    return "".join(parts)
