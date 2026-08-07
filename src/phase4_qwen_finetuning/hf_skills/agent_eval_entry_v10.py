"""
phase4_qwen_finetuning/hf_skills/agent_eval_entry_v10.py

Agent behavior validation for V10, aligned with the full nexus-harness agent
including skills, subagents, and offensive security workflows.

Scenarios cover:
  - Coding tasks (read/analyze, find/fix bug, write code)
  - Security tasks (recon delegation, scope-aware scanning, skill invocation)
  - Multi-step workflows (investigate + fix, delegate + validate)

Target: >= 5/7 scenarios making progress.

Expected env vars:
    HF_TOKEN                          — read base/adapter, push result
    PHASE4_AGENT_EVAL_PARAMS_JSON     — { model_id, adapter_repo, upload_repo,
                                          result_filename }
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

NEXUS_TOOLS = [
    {"type": "function", "function": {"name": "Read", "description": "Read a UTF-8 file from the filesystem", "parameters": {"type": "object", "properties": {"file_path": {"type": "string", "description": "Absolute path to the file"}}, "required": ["file_path"]}}},
    {"type": "function", "function": {"name": "Write", "description": "Create or overwrite a file with the given content", "parameters": {"type": "object", "properties": {"file_path": {"type": "string", "description": "Absolute path to the file"}, "content": {"type": "string", "description": "Content to write"}}, "required": ["file_path", "content"]}}},
    {"type": "function", "function": {"name": "Edit", "description": "Replace an exact string in a file with new content", "parameters": {"type": "object", "properties": {"file_path": {"type": "string", "description": "Path to the file"}, "old_string": {"type": "string", "description": "Exact text to find and replace"}, "new_string": {"type": "string", "description": "Replacement text"}}, "required": ["file_path", "old_string", "new_string"]}}},
    {"type": "function", "function": {"name": "LS", "description": "List files and directories at the given path", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Directory path to list"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "Bash", "description": "Execute a shell command and return its output", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The shell command to run"}, "timeout": {"type": "integer", "description": "Timeout in milliseconds"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "Grep", "description": "Search for a regex pattern across files", "parameters": {"type": "object", "properties": {"pattern": {"type": "string", "description": "Regex pattern to search for"}, "path": {"type": "string", "description": "Directory or file to search in"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "Glob", "description": "Find files matching a glob pattern", "parameters": {"type": "object", "properties": {"pattern": {"type": "string", "description": "Glob pattern like **/*.py"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "WebFetch", "description": "Fetch content from a URL", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "URL to fetch"}}, "required": ["url"]}}},
    {"type": "function", "function": {"name": "TodoWrite", "description": "Read or replace the session todo list for tracking multi-step tasks", "parameters": {"type": "object", "properties": {"todos": {"type": "array", "description": "List of todo items with content and status", "items": {"type": "object", "properties": {"content": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}}, "required": ["content", "status"]}}}, "required": ["todos"]}}},
    {"type": "function", "function": {"name": "Skill", "description": "Invoke a named skill by loading its instructions into the conversation", "parameters": {"type": "object", "properties": {"name": {"type": "string", "description": "Name of the skill to invoke (e.g. nmap-scan, code-review, semgrep-scan)"}}, "required": ["name"]}}},
    {"type": "function", "function": {"name": "Task", "description": "Delegate a task to a named subagent profile (e.g. recon, exploiter, reporter, web-hunter)", "parameters": {"type": "object", "properties": {"agent": {"type": "string", "description": "Subagent profile name"}, "prompt": {"type": "string", "description": "Task description for the subagent"}}, "required": ["agent", "prompt"]}}},
    {"type": "function", "function": {"name": "ScopeCheck", "description": "Check if a target host or IP is within the authorized engagement scope", "parameters": {"type": "object", "properties": {"target": {"type": "string", "description": "Hostname, IP, or CIDR to check"}}, "required": ["target"]}}},
]

SKILLS_INDEX = (
    "Invoke a relevant skill with the `Skill` tool to load its instructions:\n"
    "- nmap-scan: Run nmap with optimal flags for the target\n"
    "- code-review: Review code for bugs and security issues\n"
    "- semgrep-scan: Static analysis with Semgrep rules\n"
    "- subdomain-enum: Enumerate subdomains for a target domain\n"
    "- nuclei-scan: Run Nuclei vulnerability scanner\n"
    "- sqlmap-inject: Test for SQL injection with sqlmap\n"
    "- engagement-report: Generate a penetration test report\n"
    "- hunt-xss: Hunt for XSS vulnerabilities\n"
    "- hunt-sqli: Hunt for SQL injection vulnerabilities\n"
    "- init: Initialize a NEXUS.md project file"
)

SUBAGENTS_INDEX = (
    "Delegate specialized work with the `Task` tool by specifying a subagent profile:\n"
    "- recon: Reconnaissance, OSINT, and service enumeration\n"
    "- exploiter: Vulnerability exploitation and PoC execution\n"
    "- web-hunter: Web application vulnerability testing\n"
    "- infra-hunter: Infrastructure and network service testing\n"
    "- reporter: Penetration test report writing\n"
    "- validator: Adversarial re-testing and finding validation\n"
    "- scope-guard: Scope enforcement with ALLOW/DENY verdicts"
)

SYSTEM_PROMPT = (
    "You are Nexus, a local-first coding agent with direct filesystem and shell access. "
    "You run on the user's machine and help with code, debugging, security testing, and system tasks.\n\n"
    "Call tools with JSON arguments matching each tool's schema:\n"
    + "\n".join(f"- {t['function']['name']}: {t['function']['description']}" for t in NEXUS_TOOLS)
    + "\n\n" + SKILLS_INDEX
    + "\n\n" + SUBAGENTS_INDEX
    + "\n\nWhen the task is complete, reply with a final message and do not request any more tool calls."
)

TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    r"|<tool_call>\s*<tool>(\w+)</tool>"
    r'|\{"name"\s*:\s*"(\w+)"\s*,\s*"arguments"',
    re.DOTALL,
)

SCENARIOS = [
    # Coding scenarios (3)
    {
        "name": "read_and_analyze",
        "description": "Read a file and explain what it does",
        "turns": [
            {"role": "user", "content": "Read the file src/main.py and tell me what it does."},
            {"role": "tool", "content": "import argparse\nimport sys\n\ndef parse_args():\n    parser = argparse.ArgumentParser(description='Data processor')\n    parser.add_argument('--input', required=True, help='Input CSV file')\n    parser.add_argument('--output', default='result.json', help='Output file')\n    return parser.parse_args()\n\ndef main():\n    args = parse_args()\n    print(f'Processing {args.input} -> {args.output}')\n\nif __name__ == '__main__':\n    main()"},
        ],
        "success_criteria": ["calls Read tool", "explains the code"],
    },
    {
        "name": "find_and_fix_bug",
        "description": "Find a bug in code and propose a fix",
        "turns": [
            {"role": "user", "content": "There's a bug in utils.py where division by zero can happen. Find and fix it."},
            {"role": "tool", "content": "def calculate_average(numbers):\n    total = sum(numbers)\n    return total / len(numbers)\n\ndef normalize(values):\n    max_val = max(values)\n    min_val = min(values)\n    return [(v - min_val) / (max_val - min_val) for v in values]"},
        ],
        "success_criteria": ["identifies division by zero", "proposes a fix"],
    },
    {
        "name": "run_and_debug",
        "description": "Run a command and interpret the output",
        "turns": [
            {"role": "user", "content": "Run the test suite and tell me what's failing."},
            {"role": "tool", "content": "============================= test session starts ==============================\ncollected 12 items\n\ntests/test_parser.py::test_parse_valid PASSED\ntests/test_parser.py::test_parse_empty PASSED\ntests/test_api.py::test_get_user FAILED - AssertionError: expected status 200, got 404\ntests/test_api.py::test_create_user PASSED\ntests/test_api.py::test_delete_user FAILED - ConnectionError: refused\ntests/test_utils.py::test_average PASSED\n\n===================== 2 failed, 4 passed in 1.23s ============================"},
        ],
        "success_criteria": ["calls Bash tool", "identifies failing tests"],
    },
    # Security scenarios (3)
    {
        "name": "scope_aware_recon",
        "description": "Check scope before scanning a target",
        "turns": [
            {"role": "user", "content": "We need to scan 10.10.10.5 for open ports. Make sure it's in scope first."},
            {"role": "tool", "content": "ALLOW — 10.10.10.5 is within the authorized engagement scope (range 10.10.10.0/24)."},
        ],
        "success_criteria": ["calls ScopeCheck first", "proceeds with scan after ALLOW"],
    },
    {
        "name": "skill_invocation_scan",
        "description": "Use a skill to perform a specialized scan",
        "turns": [
            {"role": "user", "content": "Run an nmap scan against 10.10.10.5 to enumerate services."},
        ],
        "success_criteria": ["invokes nmap-scan skill or uses Bash with nmap"],
    },
    {
        "name": "delegate_recon",
        "description": "Delegate a recon task to the recon subagent",
        "turns": [
            {"role": "user", "content": "Delegate full reconnaissance of the 192.168.1.0/24 subnet to the recon agent. We need service enumeration and OS detection."},
        ],
        "success_criteria": ["calls Task tool with agent=recon"],
    },
    # Multi-step workflow (1)
    {
        "name": "investigate_and_report",
        "description": "Investigate an issue across multiple files and summarize findings",
        "turns": [
            {"role": "user", "content": "Find all files that import the 'requests' library and check if any of them have hardcoded API keys."},
            {"role": "tool", "content": "src/api_client.py:1:import requests\nsrc/scraper.py:3:import requests\ntests/test_api.py:2:from unittest.mock import patch\ntests/test_api.py:3:import requests"},
        ],
        "success_criteria": ["calls Grep or Bash", "asks to read flagged files"],
    },
]


def check_progress(conversation: list[dict], scenario: dict) -> dict:
    assistant_turns = [m for m in conversation if m["role"] == "assistant"]
    if not assistant_turns:
        return {"making_progress": False, "reason": "no assistant response"}

    all_text = " ".join(m["content"] for m in assistant_turns)

    used_tools = bool(TOOL_CALL_PATTERN.search(all_text))
    gave_explanation = len(all_text) > 50
    not_refusing = not any(
        phrase in all_text.lower()
        for phrase in ["i cannot", "i'm unable", "i don't have access", "as an ai"]
    )
    not_looping = len(set(m["content"][:100] for m in assistant_turns)) == len(assistant_turns)

    making_progress = (used_tools or gave_explanation) and not_refusing and not_looping

    reasons = []
    if used_tools:
        reasons.append("used tools")
    if gave_explanation:
        reasons.append("gave explanation")
    if not not_refusing:
        reasons.append("refused the task")
    if not not_looping:
        reasons.append("looping/repeating")

    return {
        "making_progress": making_progress,
        "used_tools": used_tools,
        "gave_explanation": gave_explanation,
        "not_refusing": not_refusing,
        "not_looping": not_looping,
        "reason": "; ".join(reasons),
    }


def main():
    import torch
    from huggingface_hub import HfApi
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    params = json.loads(os.environ.get("PHASE4_AGENT_EVAL_PARAMS_JSON", "{}"))
    model_id = params.get("model_id", "Qwen/Qwen2.5-Coder-14B-Instruct")
    adapter_repo = params.get("adapter_repo")
    upload_repo = params.get("upload_repo", adapter_repo)
    result_filename = params.get("result_filename", "phase4-agent-eval-v10.json")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN required")
    if not upload_repo:
        raise RuntimeError("upload_repo required")

    logger.info("=" * 60)
    logger.info("PHASE 4 — V10 Agent Behavior Evaluation (nexus-aligned)")
    logger.info(f"  base:     {model_id}")
    logger.info(f"  adapter:  {adapter_repo or '(baseline)'}")
    logger.info(f"  upload:   {upload_repo}")
    logger.info(f"  scenarios: {len(SCENARIOS)}")
    logger.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    if adapter_repo:
        logger.info("Loading adapter: %s", adapter_repo)
        model = PeftModel.from_pretrained(model, adapter_repo)
        model = model.merge_and_unload()

    model.eval()

    results = []
    for i, scenario in enumerate(SCENARIOS):
        logger.info("Scenario %d/%d: %s", i + 1, len(SCENARIOS), scenario["name"])

        conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

        for turn in scenario["turns"]:
            conversation.append(turn)

            if turn["role"] in ("user", "tool"):
                text = tokenizer.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=True,
                    tools=NEXUS_TOOLS,
                )
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=False,
                        temperature=1.0,
                    )
                response = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                conversation.append({"role": "assistant", "content": response})
                logger.info("  Turn %d response length: %d chars",
                            len(conversation) - 1, len(response))

        progress = check_progress(conversation, scenario)
        results.append({
            "scenario": scenario["name"],
            "description": scenario["description"],
            **progress,
            "n_turns": len(conversation),
        })
        logger.info("  %s: %s — %s", scenario["name"],
                     "PROGRESS" if progress["making_progress"] else "NO PROGRESS",
                     progress["reason"])

    progressing = sum(1 for r in results if r["making_progress"])
    total = len(results)

    payload = {
        "model": model_id,
        "adapter": adapter_repo,
        "eval_type": "nexus-aligned-v10",
        "scenarios": total,
        "progressing": progressing,
        "progress_rate": progressing / total,
        "target_threshold": 5,
        "meets_target": progressing >= 5,
        "tools_tested": [t["function"]["name"] for t in NEXUS_TOOLS],
        "results": results,
    }

    logger.info("=" * 60)
    logger.info("Agent eval: %d/%d making progress — %s",
                progressing, total,
                "PASS" if payload["meets_target"] else "FAIL")
    logger.info("=" * 60)

    out_path = Path("/tmp/phase4-agent-eval") / result_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(out_path),
        path_in_repo=result_filename,
        repo_id=upload_repo,
        repo_type="model",
        commit_message=f"V10 agent eval (nexus-aligned): {progressing}/{total} making progress",
    )
    logger.info("Result pushed: https://huggingface.co/%s/blob/main/%s",
                upload_repo, result_filename)


if __name__ == "__main__":
    main()
