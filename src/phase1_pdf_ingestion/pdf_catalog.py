"""
phase1_pdf_ingestion/pdf_catalog.py

Extends the existing SQLite catalog (data/catalog.db) with a pdf_captures table.
Uses the same database file as SQLiteCatalog so both capture types share one DB.
"""
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DDL = """
CREATE TABLE IF NOT EXISTS pdf_captures (
    id              INTEGER PRIMARY KEY,
    file_hash       TEXT NOT NULL UNIQUE,
    file_path       TEXT NOT NULL,
    domain          TEXT,
    num_pages       INTEGER,
    has_text        BOOLEAN DEFAULT FALSE,
    processed       BOOLEAN DEFAULT FALSE,
    captured_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata_json   TEXT
);
CREATE INDEX IF NOT EXISTS idx_pdf_domain    ON pdf_captures(domain);
CREATE INDEX IF NOT EXISTS idx_pdf_processed ON pdf_captures(processed);
"""


class PDFCatalog:
    """Manages the pdf_captures table in data/catalog.db."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            for stmt in DDL.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    self._conn.execute(stmt)

    def add_capture(
        self,
        file_hash: str,
        file_path: Path,
        domain: str,
        num_pages: int,
        has_text: bool,
        metadata: dict,
    ) -> None:
        """Insert or replace a PDF capture record."""
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO pdf_captures
                    (file_hash, file_path, domain, num_pages, has_text,
                     captured_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_hash,
                    str(file_path),
                    domain,
                    num_pages,
                    has_text,
                    datetime.utcnow().isoformat(),
                    json.dumps(metadata),
                ),
            )

    def mark_processed(self, file_hash: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE pdf_captures SET processed=1 WHERE file_hash=?",
                (file_hash,),
            )

    def get_unprocessed(self, domain: str | None = None) -> list[sqlite3.Row]:
        query = "SELECT * FROM pdf_captures WHERE processed=0"
        params: list = []
        if domain:
            query += " AND domain=?"
            params.append(domain)
        return self._conn.execute(query, params).fetchall()

    def stats(self) -> dict:
        rows = self._conn.execute(
            "SELECT domain, COUNT(*) as n, SUM(num_pages) as pages "
            "FROM pdf_captures GROUP BY domain ORDER BY domain"
        ).fetchall()
        return {r["domain"]: {"count": r["n"], "pages": r["pages"]} for r in rows}

    def close(self) -> None:
        self._conn.close()
