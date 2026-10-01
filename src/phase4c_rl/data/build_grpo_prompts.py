"""
phase4c_rl/data/build_grpo_prompts.py

Extract prompts from V10 eval scenarios and V9 training data for GRPO
training. Prompts are tool-use requests that the model should respond to
with proper <tool_call> formatted tool invocations.

Usage:
    python -m src.phase4c_rl.data.build_grpo_prompts \
        --config src/config/config.yaml \
        --output-dir data/grpo_prompts

    # Push to Hub:
    python -m src.phase4c_rl.data.build_grpo_prompts \
        --config src/config/config.yaml \
        --output-dir data/grpo_prompts \
        --push-to-hub \
        --max-prompts 1000
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.config.settings import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _extract_v10_eval_prompts() -> list[dict]:
    """Extract prompts from V10 tool-call eval scenarios."""
    from src.phase4_qwen_finetuning.hf_skills.tool_call_eval_entry_v10 import (
        SCENARIOS,
    )

    prompts = []
    for scenario in SCENARIOS:
        if scenario.get("expected_tool") is not None:
            prompts.append({
                "prompt": scenario["user"],
                "source": "v10_eval",
                "expected_tool": scenario["expected_tool"],
                "scenario_name": scenario["name"],
            })
    logger.info("Extracted %d prompts from V10 eval scenarios", len(prompts))
    return prompts


def _extract_v9_training_prompts(config: dict, max_rows: int = 5000) -> list[dict]:
    """Extract user prompts from the V9 training dataset that involve tool use."""
    from datasets import load_dataset

    v9_cfg = config.get("v9_mixed", {})
    dataset_id = v9_cfg.get("dataset_name", "cmndcntrlcyber/code-trainer-v9-mixed")

    logger.info("Loading V9 dataset: %s", dataset_id)
    try:
        ds = load_dataset(dataset_id, split="train")
    except Exception as e:
        logger.warning("Could not load V9 dataset: %s", e)
        return []

    prompts = []
    for i, row in enumerate(ds):
        if i >= max_rows:
            break

        messages = row.get("messages", [])
        if not messages:
            continue

        # Find user messages that are followed by assistant tool calls.
        for j, msg in enumerate(messages):
            if msg.get("role") != "user":
                continue
            # Check if the next assistant message has a tool call.
            if j + 1 < len(messages):
                next_msg = messages[j + 1]
                if next_msg.get("role") == "assistant":
                    content = next_msg.get("content", "")
                    if "<tool_call>" in content or next_msg.get("tool_calls"):
                        prompts.append({
                            "prompt": msg["content"],
                            "source": "v9_training",
                        })
                        break  # One prompt per conversation.

    logger.info("Extracted %d tool-use prompts from V9 dataset", len(prompts))
    return prompts


def _generate_synthetic_prompts() -> list[dict]:
    """Generate synthetic tool-use prompts covering all NEXUS_TOOLS_V10."""
    from src.phase4_qwen_finetuning.hf_skills.nexus_tools import NEXUS_TOOLS_V10

    templates = {
        "Read": [
            "Show me the contents of {path}.",
            "What's in the file {path}?",
            "Read {path} and summarize what it does.",
            "Open {path} so I can review it.",
        ],
        "Write": [
            "Create a file called {path} with a basic Python script that {task}.",
            "Write a config file at {path} with {task}.",
            "Save the following to {path}: {task}.",
        ],
        "Edit": [
            "In {path}, replace '{old}' with '{new}'.",
            "Fix the typo in {path}: change '{old}' to '{new}'.",
            "Update {path} to use '{new}' instead of '{old}'.",
        ],
        "LS": [
            "What files are in the {path} directory?",
            "List the contents of {path}.",
            "Show me the directory listing for {path}.",
        ],
        "Bash": [
            "Run `{cmd}` and show me the output.",
            "Execute the command: {cmd}",
            "Check {task} by running the appropriate command.",
            "Install {task} using the package manager.",
        ],
        "Grep": [
            "Search for '{pattern}' in the {path} directory.",
            "Find all occurrences of '{pattern}' in the codebase.",
            "Where is '{pattern}' defined in the source code?",
        ],
        "Glob": [
            "Find all {ext} files in this project.",
            "List every {ext} file under {path}.",
            "What configuration files exist in the project?",
        ],
        "WebFetch": [
            "Fetch the contents of {url} and summarize it.",
            "Download the page at {url}.",
            "What does {url} say?",
        ],
        "TodoWrite": [
            "Track these tasks: {task}.",
            "Add to my todo list: {task}.",
            "Create a todo list with the following items: {task}.",
        ],
        "Skill": [
            "Run an nmap scan against {target}.",
            "Do a code review of the current changes.",
            "Scan {target} for vulnerabilities with nuclei.",
        ],
        "Task": [
            "Delegate reconnaissance of {target} to the recon agent.",
            "Have the web-hunter agent test {target} for XSS.",
            "Ask the reporter agent to write up findings for {target}.",
        ],
        "ScopeCheck": [
            "Check if {target} is within our engagement scope.",
            "Before scanning {target}, verify it's authorized.",
            "Is {target} in scope for this assessment?",
        ],
    }

    fill_values = {
        "path": ["src/main.py", "/etc/hosts", "config.yaml", "README.md",
                 "tests/test_api.py", "data/output.json"],
        "task": ["parse command-line arguments", "database settings",
                 "fix the login bug, update docs, and run tests",
                 "numpy and pandas"],
        "old": ["debug = True", "localhost", "v1.0"],
        "new": ["debug = False", "0.0.0.0", "v2.0"],
        "cmd": ["uname -a", "python -m pytest", "docker ps", "netstat -tlnp",
                "pip list | grep torch"],
        "pattern": ["TODO", "import os", "def main", "password", "API_KEY"],
        "ext": ["*.py", "*.yaml", "*.rs", "*.toml"],
        "url": ["https://example.com", "https://httpbin.org/get"],
        "target": ["10.10.10.5", "192.168.1.0/24", "target.htb"],
    }

    import random
    random.seed(42)
    prompts = []

    for tool_name, template_list in templates.items():
        for template in template_list:
            # Fill in template variables.
            filled = template
            for key, values in fill_values.items():
                placeholder = "{" + key + "}"
                if placeholder in filled:
                    filled = filled.replace(placeholder, random.choice(values))
            prompts.append({
                "prompt": filled,
                "source": "synthetic",
                "expected_tool": tool_name,
            })

    logger.info("Generated %d synthetic prompts", len(prompts))
    return prompts


def _generate_tool_selection_exercises() -> list[dict]:
    """TTCA P0: 150 prompts with clear 'right tool' answers."""
    exercises = [
        ("Read the file /etc/passwd and check for suspicious accounts.", "Read"),
        ("Search the codebase for hardcoded API keys.", "Grep"),
        ("Find all YAML config files in the project.", "Glob"),
        ("List what's in the /var/log directory.", "LS"),
        ("Edit config.yaml to change the port from 8080 to 9090.", "Edit"),
        ("Create a Python script that scans for open ports.", "Write"),
        ("Run nmap against the target 10.10.10.5.", "Bash"),
        ("Fetch the robots.txt from https://target.htb.", "WebFetch"),
        ("Check if 192.168.1.100 is in our engagement scope.", "ScopeCheck"),
        ("Delegate subdomain enumeration to the recon agent.", "Task"),
        ("Read the SSH config at ~/.ssh/config.", "Read"),
        ("Search for 'password' in all Python files.", "Grep"),
        ("Find all shell scripts in the project.", "Glob"),
        ("What files are in /tmp?", "LS"),
        ("Fix the typo in main.py: change 'recieve' to 'receive'.", "Edit"),
        ("Write a reverse shell payload to /tmp/shell.py.", "Write"),
        ("Run a directory brute-force with gobuster.", "Bash"),
        ("Download the page at https://target.htb/admin.", "WebFetch"),
        ("Verify 10.10.10.0/24 is authorized for scanning.", "ScopeCheck"),
        ("Have the exploiter agent test the SQL injection.", "Task"),
    ]
    import random
    rng = random.Random(42)

    templates_per_tool = {
        "Read": [
            "Show me {path}.", "What does {path} contain?",
            "Open and review {path}.", "Display {path}.",
        ],
        "Grep": [
            "Search for '{pat}' in {path}.", "Find '{pat}' across the codebase.",
            "Where is '{pat}' used?",
        ],
        "Glob": [
            "Find all {ext} files.", "List {ext} files under {path}.",
        ],
        "LS": [
            "What's in {path}?", "Show directory {path}.",
        ],
        "Edit": [
            "In {path}, change '{old}' to '{new}'.",
            "Update {path}: replace '{old}' with '{new}'.",
        ],
        "Bash": [
            "Run {cmd}.", "Execute: {cmd}.",
        ],
        "ScopeCheck": [
            "Is {target} in scope?", "Check scope for {target}.",
        ],
    }
    fills = {
        "path": ["/etc/shadow", "src/main.py", "~/.bashrc", "data/creds.txt"],
        "pat": ["TODO", "secret", "api_key", "eval(", "exec("],
        "ext": ["*.py", "*.sh", "*.conf", "*.yml"],
        "old": ["debug=True", "0.0.0.0", "v1"],
        "new": ["debug=False", "127.0.0.1", "v2"],
        "cmd": ["netstat -tlnp", "ss -tuln", "id", "whoami", "curl -s ifconfig.me"],
        "target": ["10.10.10.5", "target.htb", "192.168.1.0/24"],
    }

    prompts = []
    for prompt_text, tool in exercises:
        prompts.append({
            "prompt": prompt_text,
            "source": "tool_selection_exercise",
            "expected_tool": tool,
        })

    for tool, tmpls in templates_per_tool.items():
        for tmpl in tmpls:
            filled = tmpl
            for key, vals in fills.items():
                ph = "{" + key + "}"
                if ph in filled:
                    filled = filled.replace(ph, rng.choice(vals))
            prompts.append({
                "prompt": filled,
                "source": "tool_selection_exercise",
                "expected_tool": tool,
            })

    rng.shuffle(prompts)
    prompts = prompts[:150]
    logger.info("Generated %d tool-selection exercise prompts", len(prompts))
    return prompts


def _generate_role_prompts(role: str) -> list[dict]:
    """Generate role-specific swarm prompts."""
    prompts = []
    if role == "orchestrator":
        delegation_prompts = [
            "Perform a full reconnaissance of 10.10.10.5 including port scanning, service enumeration, and vulnerability assessment.",
            "Audit the web application at https://target.htb for OWASP Top 10 vulnerabilities and generate a report.",
            "Enumerate all subdomains of target.htb, scan each for open ports, and identify potential entry points.",
            "Check all files in /var/www/html for hardcoded credentials, then test any found credentials against the SSH service.",
            "Scan the network 192.168.1.0/24 for live hosts, identify services, and prioritize targets by attack surface.",
            "Review the source code in src/ for security vulnerabilities, then write exploits for any critical findings.",
            "Perform a privilege escalation assessment: enumerate SUID binaries, cron jobs, and writable paths.",
            "Map the attack surface of the target: DNS records, open ports, web technologies, and potential entry points.",
            "Coordinate a full engagement: scope check, recon, vulnerability scan, exploitation, and report generation.",
            "Investigate the suspicious process on port 4444: identify it, check for persistence mechanisms, and document findings.",
        ]
        for p in delegation_prompts:
            prompts.append({
                "prompt": p,
                "source": f"role_{role}",
                "expected_tool": "Task",
            })
    elif role == "worker":
        execution_prompts = [
            "Parse this nmap output and extract all open ports with service versions.",
            "Run gobuster against https://target.htb with the common.txt wordlist.",
            "Read /etc/crontab and identify any suspicious scheduled tasks.",
            "Execute 'find / -perm -4000 -type f 2>/dev/null' and format the SUID binaries found.",
            "Search the codebase for SQL injection patterns: string concatenation in queries.",
            "Run nuclei with the cves/ template directory against 10.10.10.5.",
            "Check if the SSH service on port 22 accepts password authentication.",
            "Read the Apache access log and summarize the last 50 requests.",
            "Execute linpeas.sh and return a summary of privilege escalation vectors found.",
            "Grep for 'password', 'secret', and 'token' across all config files.",
        ]
        for p in execution_prompts:
            prompts.append({
                "prompt": p,
                "source": f"role_{role}",
            })
    logger.info("Generated %d %s role prompts", len(prompts), role)
    return prompts


def main():
    parser = argparse.ArgumentParser(
        description="Build GRPO prompt dataset from V10 evals and V9 training"
    )
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--output-dir", default="data/grpo_prompts")
    parser.add_argument("--max-prompts", type=int, default=1000)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--role", default=None,
                        choices=["orchestrator", "worker"],
                        help="Swarm role — adds role-specific prompts")
    parser.add_argument("--include-tool-selection-exercises", action="store_true",
                        help="Add 150 TTCA P0 tool-selection exercise prompts")
    args = parser.parse_args()

    import random
    random.seed(args.seed)

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect prompts from all sources.
    all_prompts = []

    # 1. V10 eval scenarios.
    all_prompts.extend(_extract_v10_eval_prompts())

    # 2. V9 training data (tool-use subset).
    all_prompts.extend(_extract_v9_training_prompts(config))

    # 3. Synthetic prompts.
    all_prompts.extend(_generate_synthetic_prompts())

    # 4. Tool-selection exercises (TTCA P0, opt-in).
    if args.include_tool_selection_exercises:
        all_prompts.extend(_generate_tool_selection_exercises())

    # 5. Role-specific prompts (swarm, opt-in).
    if args.role:
        all_prompts.extend(_generate_role_prompts(args.role))

    # Deduplicate by prompt text.
    seen = set()
    unique_prompts = []
    for p in all_prompts:
        key = p["prompt"].strip().lower()
        if key not in seen:
            seen.add(key)
            unique_prompts.append(p)

    random.shuffle(unique_prompts)
    unique_prompts = unique_prompts[: args.max_prompts]

    logger.info("Final prompt count: %d (from %d total)", len(unique_prompts), len(all_prompts))

    # Save as JSONL.
    out_path = output_dir / "train.jsonl"
    with open(out_path, "w") as f:
        for p in unique_prompts:
            f.write(json.dumps(p) + "\n")

    stats = {
        "total_prompts": len(unique_prompts),
        "sources": {},
    }
    for p in unique_prompts:
        src = p.get("source", "unknown")
        stats["sources"][src] = stats["sources"].get(src, 0) + 1
    (output_dir / "prompt_stats.json").write_text(json.dumps(stats, indent=2))

    logger.info("Saved %d prompts to %s", len(unique_prompts), out_path)

    # Push to Hub.
    if args.push_to_hub:
        from datasets import load_dataset as ld

        rl_cfg = config.get("rl_training", {})
        grpo_cfg = rl_cfg.get("grpo", {})
        ds_name = grpo_cfg.get(
            "prompt_dataset",
            f"{os.environ.get('HF_USERNAME', 'cmndcntrlcyber')}/code-trainer-v10-grpo-prompts",
        )

        logger.info("Pushing to Hub: %s", ds_name)
        ds = ld("json", data_files={"train": str(out_path)})
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        ds.push_to_hub(ds_name, token=token, private=False)
        logger.info("Pushed: https://huggingface.co/datasets/%s", ds_name)


if __name__ == "__main__":
    main()
