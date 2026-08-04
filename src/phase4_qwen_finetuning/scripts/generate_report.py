"""
phase4_qwen_finetuning/scripts/generate_report.py

Aggregate Phase 4A sweep results: download phase4-result.json from each
config's adapter repo on the Hub, rank by eval_loss, write a summary
table to docs/sweep/phase4a-summary.md, and select the best config.

Usage:
    set -a && source .env && set +a
    python -m src.phase4_qwen_finetuning.scripts.generate_report

    # Custom adapter base / output dir:
    python -m src.phase4_qwen_finetuning.scripts.generate_report \
        --adapter-base cmndcntrlcyber/qwen14b-code-trainer \
        --out docs/sweep/phase4a-summary.md
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.config.settings import load_config
from src.phase4_qwen_finetuning.configs.sweep_configs import SWEEP_CONFIGS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _fetch_result(adapter_repo: str, token: str) -> dict | None:
    """Download phase4-result.json from the adapter repo, or None if absent."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    try:
        path = hf_hub_download(
            repo_id=adapter_repo,
            filename="phase4-result.json",
            repo_type="model",
            token=token,
        )
        return json.loads(Path(path).read_text())
    except (EntryNotFoundError, RepositoryNotFoundError) as e:
        logger.warning(f"  result not on hub for {adapter_repo}: {type(e).__name__}")
        return None
    except Exception as e:
        logger.warning(f"  download failed for {adapter_repo}: {e}")
        return None


def _job_state(job_ids_path: Path, name: str, token: str) -> tuple[str, str | None]:
    """Read data/sweep_results/job_ids.json + ask HF for stage. Returns (stage, job_id)."""
    if not job_ids_path.exists():
        return "?", None
    ids = json.loads(job_ids_path.read_text())
    jid = ids.get(name)
    if not jid:
        return "?", None
    try:
        import urllib.request
        req = urllib.request.Request(
            f"https://huggingface.co/api/jobs/cmndcntrlcyber/{jid}",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            j = json.loads(r.read())
            return j.get("status", {}).get("stage", "?"), jid
    except Exception as e:
        logger.warning(f"  stage lookup failed for {jid}: {e}")
        return "?", jid


def main():
    parser = argparse.ArgumentParser(description="Aggregate Phase 4A sweep results")
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--adapter-base", default=None,
                        help="Override; defaults to qwen_finetuning.cloud.adapter_base")
    parser.add_argument("--out", default="docs/sweep/phase4a-summary.md")
    parser.add_argument("--results-dir", default="data/sweep_results")
    args = parser.parse_args()

    config = load_config(args.config)
    qf_cfg = config.get("qwen_finetuning", {})
    cloud_cfg = qf_cfg.get("cloud", {})

    adapter_base = (
        args.adapter_base
        or cloud_cfg.get("adapter_base")
        or qf_cfg.get("output_base")
    )
    if not adapter_base:
        raise SystemExit("adapter base not set; pass --adapter-base or configure qwen_finetuning.cloud.adapter_base")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN required to read result files from the adapter repo")

    job_ids_path = Path(args.results_dir) / "job_ids.json"

    rows = []
    for cfg in SWEEP_CONFIGS:
        adapter_repo = f"{adapter_base}-{cfg.name}"
        logger.info(f"[{cfg.name}] {adapter_repo}")
        stage, jid = _job_state(job_ids_path, cfg.name, token)
        result = _fetch_result(adapter_repo, token)
        rows.append({
            "name": cfg.name,
            "adapter_repo": adapter_repo,
            "job_id": jid,
            "stage": stage,
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "learning_rate": cfg.learning_rate,
            "batch_size": cfg.batch_size,
            "gradient_accumulation": cfg.gradient_accumulation,
            "result": result,
        })

    # Rank: completed runs by eval_loss ascending; missing/failed last.
    def sort_key(r):
        if r["result"] and r["result"].get("eval_loss") is not None:
            return (0, r["result"]["eval_loss"])
        return (1, 0.0)
    rows.sort(key=sort_key)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Phase 4A — Validation Sweep Summary")
    lines.append("")
    lines.append(f"**Adapter base:** `{adapter_base}-{{conservative,standard,aggressive}}`")
    lines.append("")
    lines.append("## Ranking by eval_loss")
    lines.append("")
    lines.append("| Rank | Config | LoRA r/α | LR | bs × accum | Stage | eval_loss | Adapter |")
    lines.append("|---:|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, start=1):
        loss = r["result"].get("eval_loss") if r["result"] else None
        loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else "—"
        lines.append(
            f"| {i} | `{r['name']}` "
            f"| {r['lora_r']}/{r['lora_alpha']} "
            f"| {r['learning_rate']:.0e} "
            f"| {r['batch_size']} × {r['gradient_accumulation']} "
            f"| {r['stage']} "
            f"| {loss_str} "
            f"| [link](https://hf.co/{r['adapter_repo']}) |"
        )
    lines.append("")

    completed = [r for r in rows if r["result"] and r["result"].get("eval_loss") is not None]
    if completed:
        best = completed[0]
        lines.append(f"## Best config: `{best['name']}` (eval_loss = {best['result']['eval_loss']:.4f})")
        lines.append("")
        lines.append("Recommended next step — Phase 4B full training (3 epochs) on the best config:")
        lines.append("")
        lines.append("```bash")
        lines.append("python -m src.phase4_qwen_finetuning.scripts.launch_full_training \\")
        lines.append("  --config src/config/config.yaml \\")
        lines.append(f"  --best-config {best['name']} --wait")
        lines.append("```")
    else:
        lines.append("## No completed runs yet")
        lines.append("")
        lines.append("Re-run this script once at least one job has reached `COMPLETED`.")
    lines.append("")

    lines.append("## Raw results (per config)")
    lines.append("")
    for r in rows:
        lines.append(f"### `{r['name']}` — job `{r['job_id'] or '?'}`")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(r["result"], indent=2, default=str) if r["result"] else "null")
        lines.append("```")
        lines.append("")

    out.write_text("\n".join(lines))
    logger.info(f"Wrote {out} ({len(rows)} configs, {len(completed)} completed)")

    # Also drop a machine-readable JSON for downstream tooling
    json_out = out.with_suffix(".json")
    json_out.write_text(json.dumps({
        "adapter_base": adapter_base,
        "rows": rows,
        "best": completed[0] if completed else None,
    }, indent=2, default=str))
    logger.info(f"Wrote {json_out}")


if __name__ == "__main__":
    main()
