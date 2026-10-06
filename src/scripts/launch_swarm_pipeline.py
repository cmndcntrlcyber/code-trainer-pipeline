"""
src/scripts/launch_swarm_pipeline.py

Top-level orchestrator for NEXUS swarm multi-role training. Reads
pipeline-swarm-v4a-max.yml and launches each role's training phases as
independent HF Jobs. Roles use different base models so they can
run in parallel.

Role phase chains (max quality):
    orchestrator: DAPT -> SFT -> validation -> FARCA-GRPO -> DPO -> validation -> GGUF -> abliteration
    worker:       DAPT -> SFT -> validation -> FARCA-GRPO -> DPO -> validation -> GGUF
    explore:      DAPT -> SFT -> validation -> GRPO -> DPO -> validation -> GGUF
    triage:       SFT -> validation -> GRPO -> validation -> RKLLM

Usage:
    set -a && source .env && set +a

    # Launch all four roles in parallel:
    uv run python -m src.scripts.launch_swarm_pipeline \
        --config src/config/pipeline-swarm-v4a-max.yml --role all --wait

    # Launch specific roles:
    uv run python -m src.scripts.launch_swarm_pipeline \
        --config src/config/pipeline-swarm-v4a-max.yml --role orchestrator worker --wait

    # Single role:
    uv run python -m src.scripts.launch_swarm_pipeline \
        --config src/config/pipeline-swarm-v4a-max.yml --role orchestrator --wait

    # Single phase across roles:
    uv run python -m src.scripts.launch_swarm_pipeline \
        --config src/config/pipeline-swarm-v4a-max.yml --role all --phase sft --wait

    # Dry-run:
    uv run python -m src.scripts.launch_swarm_pipeline \
        --config src/config/pipeline-swarm-v4a-max.yml --role all --dry-run
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.settings import ensure_cwd, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Role definitions
# ---------------------------------------------------------------------------

ALL_ROLES = ["orchestrator", "worker", "explore", "triage"]

# Each step: (phase_group, step_name, python_module, extra_cli_args)
# phase_group is "sft", "rl", or "deploy" — used for --phase filtering.

@dataclass(frozen=True)
class Step:
    """A single dispatchable step in a role's training chain."""
    phase_group: str
    name: str
    module: str
    extra_args: tuple[str, ...] = ()


ROLE_CHAINS: dict[str, list[Step]] = {
    "orchestrator": [
        Step("sft", "DAPT", "src.phase3b_dapt.scripts.launch_dapt"),
        Step("sft", "SFT", "src.phase4_qwen_finetuning.scripts.launch_full_training"),
        Step("sft", "Validation (post-SFT)", "src.phase4_qwen_finetuning.scripts.launch_v7_validation"),
        Step("rl", "FARCA-GRPO", "src.phase4c_rl.scripts.launch_farca_grpo"),
        Step("rl", "DPO", "src.phase4c_rl.scripts.launch_dpo"),
        Step("rl", "Validation (post-RL)", "src.phase4_qwen_finetuning.scripts.launch_v7_validation"),
        Step("deploy", "GGUF Conversion", "src.phase5_deployment.scripts.launch_convert"),
        Step("deploy", "Abliteration", "src.phase5b_abliteration.scripts.launch_abliteration"),
    ],
    "worker": [
        Step("sft", "DAPT", "src.phase3b_dapt.scripts.launch_dapt"),
        Step("sft", "SFT", "src.phase4_qwen_finetuning.scripts.launch_full_training"),
        Step("sft", "Validation (post-SFT)", "src.phase4_qwen_finetuning.scripts.launch_v7_validation"),
        Step("rl", "FARCA-GRPO", "src.phase4c_rl.scripts.launch_farca_grpo"),
        Step("rl", "DPO", "src.phase4c_rl.scripts.launch_dpo"),
        Step("rl", "Validation (post-RL)", "src.phase4_qwen_finetuning.scripts.launch_v7_validation"),
        Step("deploy", "GGUF Conversion", "src.phase5_deployment.scripts.launch_convert"),
    ],
    "explore": [
        Step("sft", "DAPT", "src.phase3b_dapt.scripts.launch_dapt"),
        Step("sft", "SFT", "src.phase4_gemma_finetuning.scripts.launch_full_training"),
        Step("sft", "Validation (post-SFT)", "src.phase4_qwen_finetuning.scripts.launch_v7_validation"),
        Step("rl", "GRPO", "src.phase4c_rl.scripts.launch_grpo"),
        Step("rl", "DPO", "src.phase4c_rl.scripts.launch_dpo"),
        Step("rl", "Validation (post-RL)", "src.phase4_qwen_finetuning.scripts.launch_v7_validation"),
        Step("deploy", "GGUF Conversion", "src.phase5_gemma_deployment.scripts.launch_convert"),
    ],
    "triage": [
        Step("sft", "SFT", "src.phase4_qwen_finetuning.scripts.launch_full_training"),
        Step("sft", "Validation (post-SFT)", "src.phase4_qwen_finetuning.scripts.launch_v7_validation"),
        Step("rl", "GRPO", "src.phase4c_rl.scripts.launch_grpo"),
        Step("rl", "Validation (post-RL)", "src.phase4_qwen_finetuning.scripts.launch_v7_validation"),
        Step("deploy", "RKLLM Conversion", "src.phase5_triage_deployment.scripts.convert_to_rkllm"),
    ],
}

PHASE_CHOICES = ["sft", "rl", "deploy"]

REQUIRED_ENV_VARS = ["HF_TOKEN", "HF_USERNAME"]

PREFLIGHT_FILES = [
    ("data/identity_examples/nexus_identity.jsonl", "Run: uv run python -m src.phase2_preprocessing.scripts.build_identity_examples --output data/identity_examples/nexus_identity.jsonl --count 400"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_roles(role_args: list[str]) -> list[str]:
    """Expand 'all' into the four concrete role names."""
    if "all" in role_args:
        return list(ALL_ROLES)
    for r in role_args:
        if r not in ALL_ROLES:
            raise SystemExit(f"Unknown role: {r!r}. Choose from: {ALL_ROLES}")
    return list(dict.fromkeys(role_args))  # deduplicate, preserve order


def get_steps_for_role(role: str, phase_filter: Optional[str]) -> list[Step]:
    """Return the steps for *role*, optionally filtered to one phase group."""
    chain = ROLE_CHAINS[role]
    if phase_filter:
        chain = [s for s in chain if s.phase_group == phase_filter]
    return chain


def read_role_config(config: dict, role: str) -> dict:
    """Extract the per-role section from the pipeline config."""
    role_cfg = config.get(role, config.get("roles", {}).get(role, {}))
    return role_cfg if isinstance(role_cfg, dict) else {}


def preflight_checks(config: dict, roles: list[str], dry_run: bool) -> list[str]:
    """Validate environment and prerequisites before launching.

    Returns a list of error strings (empty = all clear).
    Checks drawn from recurring failure modes:
      - Missing/invalid HF token (403 on Hub push)
      - Missing env vars (HF_USERNAME defaults to wrong namespace)
      - Missing prerequisite data files (identity examples, datasets)
      - Stale CWD from /mnt/ssd remount
    """
    errors: list[str] = []

    ensure_cwd()

    for var in REQUIRED_ENV_VARS:
        if not os.environ.get(var):
            errors.append(f"Environment variable {var} is not set")

    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token and not dry_run:
        try:
            import urllib.request
            req = urllib.request.Request(
                "https://huggingface.co/api/whoami",
                headers={"Authorization": f"Bearer {hf_token}"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status != 200:
                    errors.append(f"HF token validation returned HTTP {resp.status}")
        except Exception as exc:
            errors.append(f"HF token invalid or expired: {exc}")

    for rel_path, fix_hint in PREFLIGHT_FILES:
        p = Path(rel_path)
        if not p.exists():
            errors.append(f"Missing: {rel_path} — {fix_hint}")

    for role in roles:
        role_cfg = read_role_config(config, role)
        sft_cfg = role_cfg.get("sft", {})
        dataset_id = sft_cfg.get("dataset_id", "")
        if dataset_id and "${" in dataset_id:
            errors.append(
                f"[{role}] Unresolved variable in sft.dataset_id: {dataset_id} "
                "— check that HF_USERNAME is exported before sourcing .env"
            )

    return errors


def format_summary_table(
    roles: list[str],
    config: dict,
    phase_filter: Optional[str],
) -> str:
    """Build a human-readable summary table of what will be launched."""
    lines = [
        "",
        "=" * 78,
        "  NEXUS Swarm Pipeline — Launch Summary",
        "=" * 78,
        f"  {'Role':<15} {'Base Model':<35} {'Phases':<10} {'Est. Cost':>8}",
        "-" * 78,
    ]
    for role in roles:
        role_cfg = read_role_config(config, role)
        model = role_cfg.get("base_model", role_cfg.get("model", "—"))
        if isinstance(model, str) and "/" in model:
            model = model.split("/")[-1]  # show short name
        steps = get_steps_for_role(role, phase_filter)
        phases_str = ", ".join(sorted({s.phase_group for s in steps}))
        cost = role_cfg.get("estimated_cost", "N/A")
        if isinstance(cost, (int, float)):
            cost = f"${cost:.2f}"
        lines.append(f"  {role:<15} {model:<35} {phases_str:<10} {cost:>8}")
    lines.append("-" * 78)

    step_detail: list[str] = []
    for role in roles:
        steps = get_steps_for_role(role, phase_filter)
        names = " -> ".join(s.name for s in steps)
        step_detail.append(f"  {role}: {names}")
    lines.append("")
    lines.append("  Step chains:")
    lines.extend(step_detail)
    lines.append("=" * 78)
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step execution
# ---------------------------------------------------------------------------

def run_step(
    step: Step,
    role: str,
    config_path: str,
    wait: bool,
    dry_run: bool,
) -> int:
    """Execute a single step as a subprocess. Returns the exit code."""
    ensure_cwd()

    cmd = [
        sys.executable, "-m", step.module,
        "--config", config_path,
        "--role", role,
    ]
    if wait:
        cmd.append("--wait")
    if dry_run:
        cmd.append("--dry-run")
    cmd.extend(step.extra_args)

    logger.info("[%s] Starting step: %s", role, step.name)
    logger.info("[%s]   cmd: %s", role, " ".join(cmd))

    if dry_run:
        logger.info("[%s]   (dry-run) Would execute: %s", role, step.module)
        return 0

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        for line in result.stdout.strip().splitlines()[-20:]:
            logger.info("[%s] %s | %s", role, step.name, line)

    if result.returncode != 0:
        logger.error(
            "[%s] Step %r FAILED (exit %d)", role, step.name, result.returncode
        )
        if result.stderr:
            for line in result.stderr.strip().splitlines()[-30:]:
                logger.error("[%s] %s | %s", role, step.name, line)
    else:
        logger.info("[%s] Step %r completed successfully.", role, step.name)
    return result.returncode


def run_role_chain(
    role: str,
    config_path: str,
    phase_filter: Optional[str],
    wait: bool,
    dry_run: bool,
) -> tuple[str, int]:
    """Run all steps for a role sequentially. Returns (role, exit_code).

    Steps within a role are sequential because each depends on the
    previous (e.g. GRPO needs the SFT adapter). ``--wait`` is always
    passed to individual steps so each blocks before the next starts.
    """
    ensure_cwd()
    steps = get_steps_for_role(role, phase_filter)
    if not steps:
        logger.warning("[%s] No steps to run (phase filter: %s).", role, phase_filter)
        return role, 0

    logger.info(
        "[%s] Starting chain: %s",
        role,
        " -> ".join(s.name for s in steps),
    )
    start = time.monotonic()

    for i, step in enumerate(steps, 1):
        logger.info(
            "[%s] Step %d/%d: %s (%s)",
            role, i, len(steps), step.name, step.module,
        )
        # Always wait within a chain — the next step depends on this one.
        rc = run_step(step, role, config_path, wait=True, dry_run=dry_run)
        if rc != 0:
            logger.error(
                "[%s] Chain aborted at step %d/%d (%s) with exit code %d.",
                role, i, len(steps), step.name, rc,
            )
            return role, rc

    elapsed = time.monotonic() - start
    logger.info(
        "[%s] All %d steps completed in %.1f min.",
        role, len(steps), elapsed / 60,
    )
    return role, 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="NEXUS Swarm Pipeline — multi-role training orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Roles (max quality):\n"
            "  orchestrator  DAPT -> SFT -> validation -> FARCA-GRPO -> DPO -> validation -> GGUF -> abliteration\n"
            "  worker        DAPT -> SFT -> validation -> FARCA-GRPO -> DPO -> validation -> GGUF\n"
            "  explore       DAPT -> SFT -> validation -> GRPO -> DPO -> validation -> GGUF\n"
            "  triage        SFT -> validation -> GRPO -> validation -> RKLLM\n"
        ),
    )
    parser.add_argument(
        "--config",
        default="src/config/pipeline-swarm-v4a-max.yml",
        help="Path to the swarm pipeline YAML config (default: %(default)s)",
    )
    parser.add_argument(
        "--role",
        nargs="+",
        required=True,
        metavar="ROLE",
        help="Roles to launch: 'all', or one/more of: orchestrator, worker, explore, triage",
    )
    parser.add_argument(
        "--phase",
        choices=PHASE_CHOICES,
        default=None,
        help="Run only this phase group (sft, rl, deploy). Default: all phases.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Block until all launched jobs complete.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be launched without executing.",
    )
    args = parser.parse_args()

    # -- Load config --------------------------------------------------------
    ensure_cwd()
    config_path = str(Path(args.config).resolve())
    config = load_config(args.config)
    roles = resolve_roles(args.role)

    # -- Preflight checks ---------------------------------------------------
    errors = preflight_checks(config, roles, dry_run=args.dry_run)
    if errors:
        print("\n" + "=" * 60)
        print("  PREFLIGHT FAILED — fix before launching")
        print("=" * 60)
        for err in errors:
            print(f"  ✗ {err}")
        print("=" * 60 + "\n")
        return 1

    # -- Print summary table ------------------------------------------------
    summary = format_summary_table(roles, config, args.phase)
    print(summary)

    if args.dry_run:
        logger.info("--dry-run set. Walking chains without submitting jobs.")
        # Still run through the chain logic so the user sees what modules
        # would be invoked; run_step will short-circuit on dry_run.
        for role in roles:
            run_role_chain(
                role, config_path, args.phase, wait=False, dry_run=True,
            )
        logger.info("Dry-run complete. No jobs were submitted.")
        return 0

    # -- Launch roles -------------------------------------------------------
    # Different roles use different base models, so they can run in parallel.
    # Steps *within* a role are sequential (each depends on the prior).
    if len(roles) == 1:
        # Single role — run in the main process.
        _, rc = run_role_chain(
            roles[0], config_path, args.phase, args.wait, dry_run=False,
        )
        return rc

    # Multiple roles — run in parallel.
    ensure_cwd()
    logger.info("Launching %d roles in parallel.", len(roles))
    results: dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=len(roles)) as pool:
        futures = {
            pool.submit(
                run_role_chain,
                role, config_path, args.phase, args.wait, False,
            ): role
            for role in roles
        }
        for fut in as_completed(futures):
            role = futures[fut]
            try:
                _, rc = fut.result()
            except Exception:
                logger.exception("[%s] Unexpected error in role chain.", role)
                rc = 1
            results[role] = rc
            status = "OK" if rc == 0 else f"FAILED (exit {rc})"
            logger.info("[%s] Chain finished: %s", role, status)

    # -- Final report -------------------------------------------------------
    print()
    print("=" * 60)
    print("  Swarm Pipeline Results")
    print("=" * 60)
    all_ok = True
    for role in roles:
        rc = results.get(role, -1)
        marker = "PASS" if rc == 0 else "FAIL"
        if rc != 0:
            all_ok = False
        print(f"  [{marker}]  {role:<15}  exit={rc}")
    print("=" * 60)

    if not all_ok:
        failed = [r for r, rc in results.items() if rc != 0]
        logger.error(
            "Pipeline finished with failures: %s", ", ".join(failed)
        )
        return 1

    logger.info("All roles completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
