"""
Tests for src/phase1_pdf_ingestion/pdf_renderer.py and text_extractor.py.
Requires a small test PDF at tests/fixtures/sample.pdf.
"""
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from src.phase1_pdf_ingestion.pdf_renderer import PDFCaptureConfig, PDFPageCapture, _hash_file
from src.phase1_pdf_ingestion.text_extractor import PDFTextExtractor

FIXTURE_PDF = Path(__file__).parent / "fixtures" / "sample.pdf"


def _has_fixture():
    return FIXTURE_PDF.exists()


@pytest.fixture()
def tmp_output(tmp_path):
    return tmp_path / "pdf_captures"


@pytest.mark.skipif(not _has_fixture(), reason="No fixture PDF at tests/fixtures/sample.pdf")
class TestPDFPageCapture:
    def test_capture_creates_output_structure(self, tmp_output):
        cfg = PDFCaptureConfig(dpi=72, output_width=800)
        renderer = PDFPageCapture(config=cfg, output_dir=tmp_output)
        result = renderer.capture_pdf(FIXTURE_PDF, domain="Theory", source_text="test text")

        assert result.success, result.error
        assert len(result.screenshots) >= 1

        cap_dir = tmp_output / result.file_hash[:2] / result.file_hash
        assert (cap_dir / "metadata.json").exists()
        assert (cap_dir / "source.txt").exists()
        assert (cap_dir / "0000.png").exists()

    def test_metadata_fields(self, tmp_output):
        cfg = PDFCaptureConfig(dpi=72, output_width=800)
        renderer = PDFPageCapture(config=cfg, output_dir=tmp_output)
        result = renderer.capture_pdf(FIXTURE_PDF, domain="Theory", source_text="hello")

        meta = json.loads(
            (tmp_output / result.file_hash[:2] / result.file_hash / "metadata.json").read_text()
        )
        assert meta["source_type"] == "pdf"
        assert meta["domain"] == "Theory"
        assert meta["has_text"] is True
        assert meta["language"] == "text"
        assert meta["num_pages"] == len(result.screenshots)

    def test_duplicate_pdf_is_idempotent(self, tmp_output):
        cfg = PDFCaptureConfig(dpi=72, output_width=800)
        renderer = PDFPageCapture(config=cfg, output_dir=tmp_output)
        r1 = renderer.capture_pdf(FIXTURE_PDF, domain="Theory")
        r2 = renderer.capture_pdf(FIXTURE_PDF, domain="Theory")
        assert r1.file_hash == r2.file_hash


@pytest.mark.skipif(not _has_fixture(), reason="No fixture PDF at tests/fixtures/sample.pdf")
class TestPDFTextExtractor:
    def test_extract_full_returns_string(self):
        extractor = PDFTextExtractor()
        text = extractor.extract_full(FIXTURE_PDF)
        assert isinstance(text, str)

    def test_extract_pages_returns_list(self):
        extractor = PDFTextExtractor()
        pages = extractor.extract_pages(FIXTURE_PDF)
        assert isinstance(pages, list)
        assert len(pages) >= 1


class TestHashFile:
    def test_hash_is_16_chars(self, tmp_path):
        f = tmp_path / "x.pdf"
        f.write_bytes(b"fake pdf content")
        h = _hash_file(f)
        assert len(h) == 16
        assert h.isalnum()

    def test_hash_is_deterministic(self, tmp_path):
        f = tmp_path / "x.pdf"
        f.write_bytes(b"deterministic content")
        assert _hash_file(f) == _hash_file(f)
