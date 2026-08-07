"""
phase4_qwen_finetuning/hf_skills/tool_call_eval_entry_v10.py

Tool-calling validation for V10, aligned with the full nexus-harness tool
registry including session/meta tools (TodoWrite, Skill, Task), scope tools
(ScopeCheck), and all 8 core file/shell tools.

Scenarios cover:
  - 8 core tools: Read, Write, Edit, LS, Bash, Grep, Glob, WebFetch
  - 3 session tools: TodoWrite, Skill, Task
  - 1 scope tool: ScopeCheck
  - 1 negative case: no tool needed
  - 1 multi-tool case: sequential tool calls in one turn

Target: >= 80% pass rate (11/14 scenarios).

Expected env vars:
    HF_TOKEN                         — read base/adapter, push result
    PHASE4_TOOL_EVAL_PARAMS_JSON     — { model_id, adapter_repo, upload_repo,
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

from src.phase4_qwen_finetuning.hf_skills.nexus_tools import NEXUS_TOOLS_V10 as NEXUS_TOOLS

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

SCENARIOS = [
    # Core tools (8)
    {
        "name": "list_directory",
        "user": "What files are in the current directory?",
        "expected_tool": "LS",
    },
    {
        "name": "read_file",
        "user": "Show me the contents of /etc/hostname.",
        "expected_tool": "Read",
    },
    {
        "name": "run_command",
        "user": "Run `uname -a` and tell me what OS this is.",
        "expected_tool": "Bash",
    },
    {
        "name": "search_code",
        "user": "Search for all occurrences of 'TODO' in the src/ directory.",
        "expected_tool": "Grep",
    },
    {
        "name": "find_files",
        "user": "Find all YAML config files in this project.",
        "expected_tool": "Glob",
    },
    {
        "name": "write_file",
        "user": "Create a file called hello.py with a basic 'Hello World' script.",
        "expected_tool": "Write",
    },
    {
        "name": "edit_file",
        "user": "In main.py, replace 'debug = True' with 'debug = False'.",
        "expected_tool": "Edit",
    },
    {
        "name": "fetch_url",
        "user": "Fetch the contents of https://example.com and summarize the page.",
        "expected_tool": "WebFetch",
    },
    # Session/meta tools (3)
    {
        "name": "track_todos",
        "user": "I need to do three things: fix the login bug, update the README, and run tests. Track these as a todo list.",
        "expected_tool": "TodoWrite",
    },
    {
        "name": "invoke_skill",
        "user": "Run an nmap scan against the target 10.10.10.5.",
        "expected_tool": "Skill",
    },
    {
        "name": "delegate_to_subagent",
        "user": "Delegate reconnaissance of the target network 192.168.1.0/24 to the recon agent.",
        "expected_tool": "Task",
    },
    # Scope tool (1)
    {
        "name": "check_scope",
        "user": "Before we scan 10.10.10.5, check if it's within our authorized engagement scope.",
        "expected_tool": "ScopeCheck",
    },
    # Negative case (1)
    {
        "name": "no_tool_needed",
        "user": "What is the difference between a list and a tuple in Python?",
        "expected_tool": None,
    },
    # Multi-tool (1) — pass if ANY expected tool is the first call
    {
        "name": "run_tests",
        "user": "Run the test suite with pytest and tell me if anything fails.",
        "expected_tool": "Bash",
    },
]

TOOL_CALL_PATTERNS = [
    re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL),
    re.compile(r'✿\s*(\{.*?\})\s*', re.DOTALL),
    re.compile(r'"name"\s*:\s*"(\w+)".*?"arguments"\s*:', re.DOTALL),
]


def extract_tool_calls(response: str) -> list[dict]:
    calls = []

    hermes_matches = TOOL_CALL_PATTERNS[0].findall(response)
    for match in hermes_matches:
        try:
            parsed = json.loads(match)
            if "name" in parsed:
                calls.append(parsed)
        except json.JSONDecodeError:
            pass

    if calls:
        return calls

    for block in response.split("<tool_call>"):
        block = block.split("</tool_call>")[0].strip()
        if not block:
            continue
        if "<tool>" in block:
            name_match = re.search(r"<tool>(\w+)</tool>", block)
            if name_match:
                calls.append({"name": name_match.group(1), "arguments": {}})

    if calls:
        return calls

    json_pattern = re.compile(r'\{"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})', re.DOTALL)
    for name, args_str in json_pattern.findall(response):
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}
        calls.append({"name": name, "arguments": args})

    return calls


def evaluate_response(response: str, scenario: dict) -> dict:
    expected_tool = scenario["expected_tool"]
    calls = extract_tool_calls(response)

    if expected_tool is None:
        passed = len(calls) == 0
        return {
            "scenario": scenario["name"],
            "passed": passed,
            "reason": "correctly avoided tool call" if passed else "hallucinated a tool call",
            "response_preview": response[:300],
        }

    if not calls:
        return {
            "scenario": scenario["name"],
            "passed": False,
            "reason": "no tool call found in response",
            "response_preview": response[:300],
        }

    first_call = calls[0]
    correct_name = first_call.get("name") == expected_tool

    passed = correct_name
    reason = f"correct tool: {first_call.get('name')}" if correct_name else f"wrong tool: got {first_call.get('name')}, expected {expected_tool}"

    return {
        "scenario": scenario["name"],
        "passed": passed,
        "reason": reason,
        "tool_called": first_call.get("name"),
        "n_calls_found": len(calls),
        "response_preview": response[:300],
    }


def main():
    import torch
    from huggingface_hub import HfApi
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    params = json.loads(os.environ.get("PHASE4_TOOL_EVAL_PARAMS_JSON", "{}"))
    model_id = params.get("model_id", "Qwen/Qwen2.5-Coder-14B-Instruct")
    adapter_repo = params.get("adapter_repo")
    upload_repo = params.get("upload_repo", adapter_repo)
    result_filename = params.get("result_filename", "phase4-tool-call-eval-v10.json")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN required")
    if not upload_repo:
        raise RuntimeError("upload_repo required")

    logger.info("=" * 60)
    logger.info("PHASE 4 — V10 Tool-Calling Evaluation (nexus-aligned)")
    logger.info(f"  base:     {model_id}")
    logger.info(f"  adapter:  {adapter_repo or '(baseline)'}")
    logger.info(f"  upload:   {upload_repo}")
    logger.info(f"  scenarios: {len(SCENARIOS)}")
    logger.info(f"  tools:    {len(NEXUS_TOOLS)}")
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
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": scenario["user"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            tools=NEXUS_TOOLS,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=1.0,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )
        result = evaluate_response(response, scenario)
        results.append(result)
        logger.info("  %s: %s — %s", scenario["name"],
                     "PASS" if result["passed"] else "FAIL", result["reason"])

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    pass_rate = passed / total

    payload = {
        "model": model_id,
        "adapter": adapter_repo,
        "eval_type": "nexus-aligned-v10",
        "scenarios": total,
        "passed": passed,
        "pass_rate": pass_rate,
        "target_pass_rate": 0.80,
        "meets_target": pass_rate >= 0.80,
        "tools_tested": [t["function"]["name"] for t in NEXUS_TOOLS],
        "results": results,
    }

    logger.info("=" * 60)
    logger.info("Tool-calling eval: %d/%d passed (%.0f%%) — %s",
                passed, total, pass_rate * 100,
                "PASS" if payload["meets_target"] else "FAIL")
    logger.info("=" * 60)

    out_path = Path("/tmp/phase4-tool-eval") / result_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(out_path),
        path_in_repo=result_filename,
        repo_id=upload_repo,
        repo_type="model",
        commit_message=f"V10 tool-calling eval (nexus-aligned): {passed}/{total} ({pass_rate:.0%})",
    )
    logger.info("Result pushed: https://huggingface.co/%s/blob/main/%s",
                upload_repo, result_filename)


if __name__ == "__main__":
    main()
