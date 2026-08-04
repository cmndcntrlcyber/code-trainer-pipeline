"""
phase1_repo_ingestion/repo_catalog.py

Extends data/catalog.db with a repo_captures table that tracks which
repository source files have been captured via the Monaco pipeline.
"""
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DDL = """
CREATE TABLE IF NOT EXISTS repo_captures (
    id              INTEGER PRIMARY KEY,
    repo_full_name  TEXT NOT NULL,
    domain          TEXT,
    stars_list      TEXT,
    capture_hash    TEXT,
    file_path       TEXT,
    language        TEXT,
    cloned_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(repo_full_name, file_path)
);
CREATE INDEX IF NOT EXISTS idx_repo_domain     ON repo_captures(domain);
CREATE INDEX IF NOT EXISTS idx_repo_full_name  ON repo_captures(repo_full_name);
CREATE INDEX IF NOT EXISTS idx_repo_stars_list ON repo_captures(stars_list);
"""


class RepoCatalog:
    """Manages the repo_captures table in data/catalog.db."""

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

    def is_captured(self, repo_full_name: str, file_path: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM repo_captures WHERE repo_full_name=? AND file_path=?",
            (repo_full_name, file_path),
        ).fetchone()
        return row is not None

    def add_capture(
        self,
        repo_full_name: str,
        domain: str,
        stars_list: str,
        capture_hash: str,
        file_path: str,
        language: str,
    ) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO repo_captures
                    (repo_full_name, domain, stars_list, capture_hash,
                     file_path, language, cloned_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    repo_full_name,
                    domain,
                    stars_list,
                    capture_hash,
                    file_path,
                    language,
                    datetime.utcnow().isoformat(),
                ),
            )

    def get_processed_repos(self, domain: str) -> set[str]:
        """Return repo full_names that already have at least one capture for this domain."""
        rows = self._conn.execute(
            "SELECT DISTINCT repo_full_name FROM repo_captures WHERE domain = ?",
            (domain,),
        ).fetchall()
        return {r["repo_full_name"] for r in rows}

    def stats(self) -> dict:
        rows = self._conn.execute(
            "SELECT domain, stars_list, COUNT(*) as files "
            "FROM repo_captures GROUP BY domain, stars_list ORDER BY domain, stars_list"
        ).fetchall()
        return {
            f"{r['domain']} / {r['stars_list']}": r["files"]
            for r in rows
        }

    def close(self) -> None:
        self._conn.close()
