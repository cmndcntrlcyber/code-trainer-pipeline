"""
phase1_repo_ingestion/repo_cloner.py

Shallow-clones GitHub repositories and walks their source files by language
extension. Used by run_repo_ingestion.py to feed files into the existing
MonacoCapture / ParallelCapture pipeline.
"""
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

LANG_EXTENSIONS: dict[str, list[str]] = {
    "python": [".py"],
    "rust": [".rs", ".toml"],
    "go": [".go"],
    "c": [".c", ".h", ".cpp", ".hpp", ".cc", ".cxx"],
    "java": [".java"],
    "powershell": [".ps1", ".psm1", ".psd1"],
    "javascript": [".js", ".ts", ".jsx", ".tsx", ".mjs"],
    "csharp": [".cs"],
    "bash": [".sh", ".bash"],
    "yaml": [".yaml", ".yml"],
}

# Directories to always skip when walking source files
_SKIP_DIRS = {
    ".git", "node_modules", "vendor", "target", "__pycache__",
    ".cargo", "dist", "build", "out", ".next", "venv", ".venv",
}

_MAX_FILE_BYTES = 500 * 1024  # 500 KB — skip generated/minified files


@dataclass
class RepoCloneConfig:
    clone_dir: Path
    depth: int = 1
    timeout_s: int = 120
    max_file_size_kb: int = 500


@dataclass
class CloneResult:
    full_name: str
    repo_path: Optional[Path] = None
    success: bool = False
    error: Optional[str] = None
    source_files: list[Path] = field(default_factory=list)


class RepoCloner:
    """Shallow-clones repos from github.com and collects source files."""

    def __init__(self, config: RepoCloneConfig):
        self.config = config
        self.config.clone_dir.mkdir(parents=True, exist_ok=True)

    def clone(self, full_name: str) -> CloneResult:
        """
        Shallow-clone github.com/<full_name>.
        Returns CloneResult with repo_path set on success.
        Re-uses an existing clone if present (no fetch — offline safe).
        """
        owner, repo = full_name.split("/", 1)
        dest = self.config.clone_dir / owner / repo
        result = CloneResult(full_name=full_name)

        if dest.exists():
            logger.debug(f"  [{full_name}] already cloned at {dest}")
            result.repo_path = dest
            result.success = True
            return result

        url = f"https://github.com/{full_name}.git"
        cmd = [
            "git", "clone",
            "--depth", str(self.config.depth),
            "--single-branch",
            "--no-tags",
            "--quiet",
            url,
            str(dest),
        ]
        logger.info(f"  Cloning {url}")
        try:
            subprocess.run(
                cmd,
                timeout=self.config.timeout_s,
                check=True,
                capture_output=True,
            )
            result.repo_path = dest
            result.success = True
        except subprocess.TimeoutExpired:
            result.error = f"Clone timed out after {self.config.timeout_s}s"
        except subprocess.CalledProcessError as exc:
            result.error = exc.stderr.decode(errors="replace").strip()

        return result

    def walk_source_files(
        self,
        repo_path: Path,
        extensions: list[str],
    ) -> list[Path]:
        """
        Walk repo_path and return files matching extensions.
        Skips known vendor/generated directories and oversized files.
        """
        max_bytes = self.config.max_file_size_kb * 1024
        found: list[Path] = []

        for path in repo_path.rglob("*"):
            # skip blacklisted ancestor dirs
            if any(part in _SKIP_DIRS for part in path.parts):
                continue
            if not path.is_file():
                continue
            if path.suffix.lower() not in extensions:
                continue
            try:
                if path.stat().st_size > max_bytes:
                    logger.debug(f"  Skipping oversized file: {path}")
                    continue
            except OSError:
                continue
            found.append(path)

        return sorted(found)
