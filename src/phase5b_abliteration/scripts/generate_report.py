"""
phase5b_abliteration/scripts/generate_report.py

Download evaluation results from Hub and produce a comparative report
at docs/sweep/phase5b-abliteration-report.md.

Usage:
    set -a && source .env && set +a
    python -m src.phase5b_abliteration.scripts.generate_report \
        --config src/config/config.yaml

    # Override results repo:
    python -m src.phase5b_abliteration.scripts.generate_report \
        --results-repo cmndcntrlcyber/qwen14b-code-trainer-abliteration-results
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from huggingface_hub import hf_hub_download, list_repo_files
from src.config.settings import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

REPORT_DIR = Path("docs/sweep")


def download_results(results_repo: str, token: str, local_dir: Path) -> list[Path]:
    local_dir.mkdir(parents=True, exist_ok=True)
    files = list_repo_files(results_repo, repo_type="model", token=token)
    json_files = [f for f in files if f.endswith(".json")]

    downloaded = []
    for fname in json_files:
        local_path = local_dir / fname
        hf_hub_download(
            results_repo,
            filename=fname,
            repo_type="model",
            token=token,
            local_dir=str(local_dir),
        )
        downloaded.append(local_path)
        logger.info("Downloaded %s", fname)

    return downloaded


def build_report(results_dir: Path) -> str:
    summary_path = results_dir / "comparative_summary.json"
    if not summary_path.exists():
        return "# Phase 5b Abliteration Report\n\nNo comparative_summary.json found.\n"

    summary = json.loads(summary_path.read_text())
    ranking = summary.get("ranking", [])

    lines = [
        "# Phase 5b: Abliteration Benchmarking Report",
        "",
        "## Ranking",
        "",
        "| Rank | Variant | Refusal Rate | KL Divergence | Perplexity |"
        " GSM8K | MMLU | TruthfulQA |",
        "|------|---------|-------------|---------------|------------|"
        "-------|------|------------|",
    ]

    for i, row in enumerate(ranking, 1):
        refusal = row.get("refusal_rate", -1)
        kl = row.get("kl_divergence", -1)
        ppl = row.get("perplexity", -1)
        gsm8k = row.get("gsm8k_acc", -1)
        mmlu = row.get("mmlu_acc", -1)
        truthful = row.get("truthfulqa_mc2_acc", -1)

        lines.append(
            f"| {i} | **{row['variant']}** "
            f"| {refusal:.2%} " if refusal >= 0 else f"| N/A "
            f"| {kl:.4f} " if kl >= 0 else f"| N/A "
            f"| {ppl:.2f} " if ppl >= 0 else f"| N/A "
            f"| {gsm8k:.2%} " if gsm8k >= 0 else f"| N/A "
            f"| {mmlu:.2%} " if mmlu >= 0 else f"| N/A "
            f"| {truthful:.2%} |" if truthful >= 0 else f"| N/A |"
        )

    lines.extend([
        "",
        f"**Best by refusal rate**: {summary.get('best_by_refusal', 'N/A')}",
        f"**Best by KL divergence**: {summary.get('best_by_kl', 'N/A')}",
        "",
        "## Technique Details",
        "",
    ])

    for json_file in sorted(results_dir.glob("*_results.json")):
        if json_file.name == "comparative_summary.json":
            continue
        data = json.loads(json_file.read_text())
        variant = data.get("variant", json_file.stem)
        lines.append(f"### {variant}")
        lines.append("")

        if "refusal" in data:
            r = data["refusal"]
            lines.append(
                f"- Refusal rate: {r['refusal_rate']:.2%} "
                f"({r['n_refused']}/{r['n_total']})"
            )
        if "kl_divergence" in data:
            lines.append(
                f"- KL divergence: {data['kl_divergence']['kl_divergence_mean']:.4f}"
            )
        if "perplexity" in data:
            lines.append(f"- Perplexity: {data['perplexity']['perplexity']:.2f}")

        if "benchmarks" in data:
            lines.append("- Benchmarks:")
            for task, result in data["benchmarks"].items():
                if isinstance(result, dict):
                    acc = result.get("acc") or result.get("acc,none", "N/A")
                    lines.append(f"  - {task}: {acc}")
                else:
                    lines.append(f"  - {task}: {result}")

        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Phase 5b abliteration comparative report"
    )
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--results-repo", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    abl_cfg = config.get("abliteration", {})
    output_base = abl_cfg.get("output_base", "")
    results_repo = args.results_repo or f"{output_base}-abliteration-results"

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""
    if not token:
        raise SystemExit("HF_TOKEN required")

    local_dir = Path("data/abliteration_results")
    logger.info("Downloading results from %s...", results_repo)
    download_results(results_repo, token, local_dir)

    report = build_report(local_dir)

    output_path = Path(args.output) if args.output else REPORT_DIR / "phase5b-abliteration-report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    logger.info("Report written to %s", output_path)

    summary_path = local_dir / "comparative_summary.json"
    if summary_path.exists():
        json_out = REPORT_DIR / "phase5b-abliteration-summary.json"
        json_out.write_text(summary_path.read_text())
        logger.info("Summary JSON written to %s", json_out)


if __name__ == "__main__":
    main()
