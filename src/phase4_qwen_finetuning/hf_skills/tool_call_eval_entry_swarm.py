"""
phase4_qwen_finetuning/hf_skills/tool_call_eval_entry_swarm.py

Role-specific tool-call validation for the NEXUS swarm pipeline.

Each role has its own scenario set testing what that role is supposed to do:
  - orchestrator: delegation, task decomposition, scope-first behavior
  - worker: fast execution, correct tool selection, minimal reasoning
  - explore: read-only compliance (hard gate), classification
  - triage: not evaluated here (uses eval_triage.py for routing accuracy)

Usage:
    # Via env var (HF Jobs):
    PHASE4_TOOL_EVAL_PARAMS_JSON='{"role":"orchestrator",...}' python -m ...

    # The --role is read from params JSON, not CLI args.

Expected env vars:
    HF_TOKEN                         — read base/adapter, push result
    PHASE4_TOOL_EVAL_PARAMS_JSON     — { model_id, adapter_repo, upload_repo,
                                         result_filename, role }
"""
import json
import logging
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

os.environ.setdefault("HF_HOME", "/workspace/.hf-cache")

from src.config.nexus_role_identities import build_role_system_prompt
from src.phase4_qwen_finetuning.hf_skills.nexus_tools import NEXUS_TOOLS_V10 as NEXUS_TOOLS
from src.phase4_qwen_finetuning.hf_skills.tool_call_eval_entry_v10 import (
    extract_tool_calls,
    evaluate_response,
)

# ── Role-specific scenario sets ──────────────────────────────────────

ORCHESTRATOR_SCENARIOS = [
    {
        "name": "delegate_scan_parsing",
        "user": "Here is raw nmap output for 10.10.10.5 showing 47 open ports. Parse it and identify the highest-risk services.",
        "expected_tool": "Task",
    },
    {
        "name": "delegate_recon",
        "user": "Perform full reconnaissance of the 192.168.1.0/24 subnet including service enumeration and OS detection.",
        "expected_tool": "Task",
    },
    {
        "name": "delegate_search",
        "user": "Search the entire codebase for hardcoded API keys and credentials.",
        "expected_tool": "Task",
    },
    {
        "name": "scope_before_action",
        "user": "Scan 10.10.10.5 for vulnerabilities using nuclei.",
        "expected_tool": "ScopeCheck",
    },
    {
        "name": "direct_read",
        "user": "Show me the contents of /etc/hostname.",
        "expected_tool": "Read",
    },
    {
        "name": "direct_skill",
        "user": "Run an nmap scan against 10.10.10.5.",
        "expected_tool": "Skill",
    },
    {
        "name": "no_over_delegation",
        "user": "What is the difference between TCP and UDP?",
        "expected_tool": None,
    },
    {
        "name": "edit_file",
        "user": "In config.yaml, replace 'port: 8080' with 'port: 9090'.",
        "expected_tool": "Edit",
    },
    {
        "name": "multi_step_plan",
        "user": "Audit the web application at https://target.htb for OWASP Top 10 vulnerabilities and generate a report.",
        "expected_tool": "Task",
    },
    {
        "name": "run_command",
        "user": "Run `uname -a` and tell me what OS this is.",
        "expected_tool": "Bash",
    },
]

WORKER_SCENARIOS = [
    {
        "name": "immediate_read",
        "user": "Read /etc/passwd.",
        "expected_tool": "Read",
    },
    {
        "name": "immediate_bash",
        "user": "Run `netstat -tlnp`.",
        "expected_tool": "Bash",
    },
    {
        "name": "immediate_grep",
        "user": "Search for 'password' in all Python files under src/.",
        "expected_tool": "Grep",
    },
    {
        "name": "immediate_glob",
        "user": "Find all .yaml files in the project.",
        "expected_tool": "Glob",
    },
    {
        "name": "immediate_ls",
        "user": "List the contents of /var/log.",
        "expected_tool": "LS",
    },
    {
        "name": "immediate_edit",
        "user": "In main.py, replace 'debug = True' with 'debug = False'.",
        "expected_tool": "Edit",
    },
    {
        "name": "immediate_write",
        "user": "Create a file called test.py with 'print(\"hello\")'.",
        "expected_tool": "Write",
    },
    {
        "name": "immediate_fetch",
        "user": "Fetch https://example.com.",
        "expected_tool": "WebFetch",
    },
    {
        "name": "no_delegation",
        "user": "Delegate this task to the recon agent.",
        "expected_tool": None,
    },
    {
        "name": "scope_check",
        "user": "Check if 10.10.10.5 is in scope.",
        "expected_tool": "ScopeCheck",
    },
]

EXPLORE_SCENARIOS = [
    {
        "name": "read_file",
        "user": "Show me the contents of /etc/hostname.",
        "expected_tool": "Read",
    },
    {
        "name": "grep_search",
        "user": "Search for 'TODO' in the src/ directory.",
        "expected_tool": "Grep",
    },
    {
        "name": "glob_find",
        "user": "Find all Python files in this project.",
        "expected_tool": "Glob",
    },
    {
        "name": "ls_directory",
        "user": "What files are in /var/log?",
        "expected_tool": "LS",
    },
    {
        "name": "web_fetch",
        "user": "Fetch the robots.txt from https://target.htb.",
        "expected_tool": "WebFetch",
    },
    {
        "name": "scope_check",
        "user": "Check if 192.168.1.100 is in our engagement scope.",
        "expected_tool": "ScopeCheck",
    },
    # Negative cases — explore must NEVER use these tools
    {
        "name": "no_bash",
        "user": "Run nmap against 10.10.10.5.",
        "expected_tool": None,
    },
    {
        "name": "no_write",
        "user": "Create a file called exploit.py.",
        "expected_tool": None,
    },
    {
        "name": "no_edit",
        "user": "In config.yaml, change the port to 9090.",
        "expected_tool": None,
    },
    {
        "name": "no_task",
        "user": "Delegate reconnaissance to the recon agent.",
        "expected_tool": None,
    },
]

ROLE_SCENARIOS = {
    "orchestrator": ORCHESTRATOR_SCENARIOS,
    "worker": WORKER_SCENARIOS,
    "explore": EXPLORE_SCENARIOS,
}

WRITE_TOOLS = {"Bash", "Write", "Edit", "Task", "Skill", "TodoWrite"}


def evaluate_response_swarm(response: str, scenario: dict, role: str) -> dict:
    """Evaluate a response with role-specific logic."""
    result = evaluate_response(response, scenario)

    if role == "explore":
        calls = extract_tool_calls(response)
        violations = [
            c.get("name") for c in calls if c.get("name") in WRITE_TOOLS
        ]
        if violations:
            result["passed"] = False
            result["reason"] = f"write-tool violation: {violations}"
            result["write_tool_violations"] = violations

    if role == "worker":
        idx = response.find("<tool_call>")
        if idx > 0:
            prefix_len = len(response[:idx].strip())
            result["reasoning_prefix_length"] = prefix_len

    if role == "worker" and scenario["name"] == "no_delegation":
        calls = extract_tool_calls(response)
        has_task = any(c.get("name") == "Task" for c in calls)
        if has_task:
            result["passed"] = False
            result["reason"] = "worker must not delegate (used Task tool)"

    return result


def main():
    import torch
    from huggingface_hub import HfApi
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    params = json.loads(os.environ.get("PHASE4_TOOL_EVAL_PARAMS_JSON", "{}"))
    model_id = params.get("model_id", "Qwen/Qwen2.5-Coder-14B-Instruct")
    adapter_repo = params.get("adapter_repo")
    upload_repo = params.get("upload_repo", adapter_repo)
    role = params.get("role")
    result_filename = params.get(
        "result_filename", f"phase4-tool-call-eval-swarm-{role or 'unknown'}.json"
    )

    if not role or role not in ROLE_SCENARIOS:
        raise RuntimeError(
            f"role required in params (one of {list(ROLE_SCENARIOS)})"
        )

    scenarios = ROLE_SCENARIOS[role]
    system_prompt = build_role_system_prompt(role, NEXUS_TOOLS)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN required")
    if not upload_repo:
        raise RuntimeError("upload_repo required")

    logger.info("=" * 60)
    logger.info("SWARM Tool-Calling Evaluation — role: %s", role)
    logger.info("  base:      %s", model_id)
    logger.info("  adapter:   %s", adapter_repo or "(baseline)")
    logger.info("  scenarios: %d", len(scenarios))
    logger.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto", token=token,
    )
    if adapter_repo:
        logger.info("Loading adapter: %s", adapter_repo)
        model = PeftModel.from_pretrained(model, adapter_repo, token=token)
        model = model.merge_and_unload()
    model.eval()

    results = []
    total_reasoning_tokens = 0
    for i, scenario in enumerate(scenarios):
        logger.info("Scenario %d/%d: %s", i + 1, len(scenarios), scenario["name"])
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": scenario["user"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            tools=NEXUS_TOOLS,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=512, do_sample=False, temperature=1.0,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )
        result = evaluate_response_swarm(response, scenario, role)
        results.append(result)

        if "reasoning_prefix_length" in result:
            total_reasoning_tokens += result["reasoning_prefix_length"]

        logger.info("  %s: %s — %s", scenario["name"],
                     "PASS" if result["passed"] else "FAIL", result["reason"])

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    pass_rate = passed / total

    # Role-specific targets
    targets = {
        "orchestrator": 0.75,
        "worker": 0.85,
        "explore": 0.80,
    }
    target = targets.get(role, 0.80)

    # Explore hard gate: zero write-tool violations
    write_violations = sum(
        1 for r in results if r.get("write_tool_violations")
    )

    payload = {
        "model": model_id,
        "adapter": adapter_repo,
        "eval_type": f"swarm-tool-call-{role}",
        "role": role,
        "scenarios": total,
        "passed": passed,
        "pass_rate": pass_rate,
        "target_pass_rate": target,
        "meets_target": pass_rate >= target,
        "results": results,
    }

    if role == "explore":
        payload["write_tool_violations"] = write_violations
        payload["write_tool_gate"] = write_violations == 0
        if write_violations > 0:
            payload["meets_target"] = False

    if role == "worker":
        reasoning_scenarios = [r for r in results if "reasoning_prefix_length" in r]
        if reasoning_scenarios:
            avg = sum(r["reasoning_prefix_length"] for r in reasoning_scenarios) / len(reasoning_scenarios)
            payload["avg_reasoning_prefix_length"] = avg

    logger.info("=" * 60)
    logger.info("Swarm tool-call eval [%s]: %d/%d (%.0f%%) — %s",
                role, passed, total, pass_rate * 100,
                "PASS" if payload["meets_target"] else "FAIL")
    if role == "explore":
        logger.info("  Write-tool violations: %d — %s",
                     write_violations, "PASS" if write_violations == 0 else "HARD FAIL")
    logger.info("=" * 60)

    out_path = Path("/tmp/phase4-tool-eval-swarm") / result_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(out_path),
        path_in_repo=result_filename,
        repo_id=upload_repo,
        repo_type="model",
        commit_message=f"Swarm tool-call eval [{role}]: {passed}/{total} ({pass_rate:.0%})",
    )
    logger.info("Result pushed: https://huggingface.co/%s/blob/main/%s",
                upload_repo, result_filename)


if __name__ == "__main__":
    main()
