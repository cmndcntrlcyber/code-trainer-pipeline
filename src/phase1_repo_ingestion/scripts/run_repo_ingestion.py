"""
phase1_repo_ingestion/scripts/run_repo_ingestion.py

CLI orchestrator for GitHub Stars repo ingestion.
Reads repos.json catalogue files from each knowledge branch, shallow-clones
each repo, walks source files by language, and feeds them through the
existing ParallelCapture / MonacoCapture pipeline (same as Phase 1 code
collection). All captures land in data/captures/ with `domain` and
`stars_list` metadata recorded in the repo_captures SQLite table.

Usage:
  # Single branch catalogue:
  uv run python -m src.phase1_repo_ingestion.scripts.run_repo_ingestion \\
      --config src/config/config.yaml \\
      --catalogue /mnt/ssd/knowledge/repos.json \\
      --domain "Red Team Knowledge" \\
      --clone-dir /tmp/repo_clones \\
      --workers 4

  # All branches that have a repos.json:
  uv run python -m src.phase1_repo_ingestion.scripts.run_repo_ingestion \\
      --config src/config/config.yaml \\
      --all-branches \\
      --knowledge-dir /mnt/ssd/knowledge \\
      --clone-dir /tmp/repo_clones \\
      --workers 4
"""
import argparse
import asyncio
import json
import logging
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.config.settings import load_config
from src.phase1_data_collection.capture.parallel_capture import ParallelCapture
from src.phase1_data_collection.capture.vscode_automation import CaptureConfig
from src.phase1_repo_ingestion.repo_catalog import RepoCatalog
from src.phase1_repo_ingestion.repo_cloner import LANG_EXTENSIONS, RepoCloneConfig, RepoCloner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_catalog_lock = threading.Lock()

# Extension → canonical language name for capture tagging
_EXT_TO_LANG: dict[str, str] = {}
for lang, exts in LANG_EXTENSIONS.items():
    for ext in exts:
        _EXT_TO_LANG[ext] = lang

# All extensions we care about
_ALL_EXTENSIONS = list(_EXT_TO_LANG.keys())


def _ingest_one_repo(
    repo_info: dict,
    domain: str,
    captures_dir: Path,
    clone_cfg: RepoCloneConfig,
) -> dict:
    """
    Worker: clone a single repo and run captures on its source files.
    Returns stats dict. Does NOT write to catalog (caller serializes that).
    """
    full_name = repo_info["full_name"]
    stars_list = repo_info.get("stars_list", "")

    cloner = RepoCloner(clone_cfg)
    clone_result = cloner.clone(full_name)

    if not clone_result.success:
        return {
            "full_name": full_name,
            "stars_list": stars_list,
            "success": False,
            "error": clone_result.error,
            "files_captured": 0,
        }

    MAX_FILES = 80
    source_files = cloner.walk_source_files(clone_result.repo_path, _ALL_EXTENSIONS)
    total_found = len(source_files)
    if total_found > MAX_FILES:
        source_files = source_files[:MAX_FILES]
        logger.info(f"  [{full_name}] capped to {MAX_FILES}/{total_found} source files")
    else:
        logger.info(f"  [{full_name}] {total_found} source files found")

    file_records = []

    if source_files:
        pc = ParallelCapture(
            num_workers=1,
            config=CaptureConfig(),
            output_dir=captures_dir,
        )
        try:
            capture_results = asyncio.run(pc.capture_all(source_files))
        except Exception as exc:
            logger.warning(f"  [{full_name}] capture run failed: {exc}")
            capture_results = []

        for cr, src_file in zip(capture_results, source_files):
            if cr.success:
                language = _EXT_TO_LANG.get(src_file.suffix.lower(), "unknown")
                file_records.append({
                    "file_path": str(src_file.relative_to(clone_result.repo_path)),
                    "language": language,
                    "capture_hash": cr.file_hash,
                })

    return {
        "full_name": full_name,
        "stars_list": stars_list,
        "success": True,
        "files_captured": len(file_records),
        "file_records": file_records,
        "error": None,
    }


def ingest_catalogue(
    catalogue_path: Path,
    domain: str,
    captures_dir: Path,
    catalog: RepoCatalog,
    clone_cfg: RepoCloneConfig,
    workers: int,
) -> dict:
    """Process all repos in a repos.json catalogue. Returns summary stats."""
    with open(catalogue_path, encoding="utf-8") as f:
        catalogue = json.load(f)

    repos = catalogue.get("repositories", [])
    already_done = catalog.get_processed_repos(domain)
    if already_done:
        logger.info(f"[{domain}] {len(already_done)} repos already captured — skipping")
        repos = [r for r in repos if r["full_name"] not in already_done]
    logger.info(
        f"[{domain}] {len(repos)} repos to process from catalogue "
        f"{catalogue_path} (workers={workers})"
    )

    total_repos = len(repos)
    total_ok = 0
    total_failed = 0
    total_files = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_ingest_one_repo, repo, domain, captures_dir, clone_cfg): repo
            for repo in repos
        }
        for future in as_completed(futures):
            repo = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                logger.warning(f"  FAILED {repo['full_name']}: {exc}")
                total_failed += 1
                continue

            if result["success"]:
                with _catalog_lock:
                    for rec in result.get("file_records", []):
                        catalog.add_capture(
                            repo_full_name=result["full_name"],
                            domain=domain,
                            stars_list=result["stars_list"],
                            capture_hash=rec["capture_hash"],
                            file_path=rec["file_path"],
                            language=rec["language"],
                        )
                total_ok += 1
                total_files += result["files_captured"]
                logger.info(
                    f"  OK  {result['full_name']} — {result['files_captured']} files captured"
                )
            else:
                total_failed += 1
                logger.warning(f"  FAIL {result['full_name']}: {result['error']}")

    return {
        "domain": domain,
        "repos": total_repos,
        "success": total_ok,
        "failed": total_failed,
        "files": total_files,
    }


def _print_summary(stats: list[dict]) -> None:
    logger.info("\n=== Repo Ingestion Summary ===")
    total_repos = total_ok = total_fail = total_files = 0
    for s in stats:
        logger.info(
            f"  {s['domain']:30s}  Repos: {s['repos']:4d}  "
            f"OK: {s['success']:4d}  Failed: {s['failed']:3d}  "
            f"Files: {s['files']:6d}"
        )
        total_repos += s["repos"]
        total_ok += s["success"]
        total_fail += s["failed"]
        total_files += s["files"]
    logger.info(
        f"  {'TOTAL':30s}  Repos: {total_repos:4d}  OK: {total_ok:4d}  "
        f"Failed: {total_fail:3d}  Files: {total_files:6d}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest GitHub Stars repos as training captures")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--catalogue", help="Path to a repos.json catalogue file")
    parser.add_argument("--domain", help="Domain label for captures (e.g. 'Red Team Knowledge')")
    parser.add_argument(
        "--all-branches",
        action="store_true",
        help="Process all repos.json files found on knowledge branches",
    )
    parser.add_argument(
        "--knowledge-dir",
        default=None,
        help="Root of knowledge repo (for --all-branches); overrides config",
    )
    parser.add_argument(
        "--clone-dir",
        default="/tmp/repo_clones",
        help="Directory for shallow git clones (default: /tmp/repo_clones)",
    )
    parser.add_argument(
        "--captures-dir",
        default=None,
        help="Override output directory for captures (default: repo_ingestion.captures_dir in config)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel repo workers (overrides config repo_ingestion.workers)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_cfg_section = cfg.get("repo_ingestion", {})
    pdf_cfg_section = cfg.get("pdf_ingestion", {})

    workers = args.workers or repo_cfg_section.get("workers", 4)
    captures_dir = Path(
        args.captures_dir
        or repo_cfg_section.get("captures_dir", cfg.get("data_collection", {}).get("captures_dir", "./data/captures"))
    )
    catalog_db = Path(cfg.get("data_collection", {}).get("catalog_db", "./data/catalog.db"))
    clone_dir = Path(args.clone_dir or repo_cfg_section.get("clone_dir", "/tmp/repo_clones"))
    depth = repo_cfg_section.get("depth", 1)
    max_file_size_kb = repo_cfg_section.get("max_file_size_kb", 500)

    captures_dir.mkdir(parents=True, exist_ok=True)
    catalog_db.parent.mkdir(parents=True, exist_ok=True)
    clone_dir.mkdir(parents=True, exist_ok=True)

    catalog = RepoCatalog(catalog_db)
    clone_cfg = RepoCloneConfig(
        clone_dir=clone_dir,
        depth=depth,
        max_file_size_kb=max_file_size_kb,
    )

    if args.all_branches:
        knowledge_dir = Path(
            args.knowledge_dir
            or repo_cfg_section.get("knowledge_dir")
            or pdf_cfg_section.get("knowledge_dir", "/mnt/ssd/knowledge")
        )
        # Find all repos.json files across knowledge/* branches
        repo_root = knowledge_dir
        result = subprocess.run(
            ["git", "branch", "--list", "knowledge/*"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        branches = [b.strip().lstrip("* ") for b in result.stdout.splitlines() if b.strip()]

        all_stats = []
        current_branch_result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root, capture_output=True, text=True,
        )
        original_branch = current_branch_result.stdout.strip()

        for branch in branches:
            subprocess.run(["git", "checkout", branch], cwd=repo_root, capture_output=True)
            repos_json = repo_root / "repos.json"
            if not repos_json.exists():
                logger.info(f"[{branch}] no repos.json — skipping")
                continue

            with open(repos_json, encoding="utf-8") as f:
                meta = json.load(f)
            domain = meta.get("branch", branch).replace("knowledge/", "").replace("-", " ").title()

            stats = ingest_catalogue(repos_json, domain, captures_dir, catalog, clone_cfg, workers)
            all_stats.append(stats)

        subprocess.run(["git", "checkout", original_branch], cwd=repo_root, capture_output=True)
        _print_summary(all_stats)

    elif args.catalogue and args.domain:
        stats = ingest_catalogue(
            Path(args.catalogue), args.domain, captures_dir, catalog, clone_cfg, workers
        )
        _print_summary([stats])

    else:
        parser.error("Provide --all-branches OR both --catalogue and --domain")

    catalog.close()


if __name__ == "__main__":
    main()
