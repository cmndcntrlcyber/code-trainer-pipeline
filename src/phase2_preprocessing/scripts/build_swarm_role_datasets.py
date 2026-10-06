"""
phase2_preprocessing/scripts/build_swarm_role_datasets.py

Build role-specific SFT training datasets for a distributed inference swarm.
Each swarm role (orchestrator, worker, explore, triage) gets a differently
composed dataset from shared data sources.

Dataset composition per role:
  Orchestrator (~12K): 4K code-gen, 6K tool-calling (multi-step), 3K agent
      traces, 2K instruction, 400 identity, 500 synthetic delegation
  Worker (~8K): 2K code-gen, 5K tool-calling (single-call), 1K short agent
      traces, 1K instruction, 200 identity
  Explore (~5K): 1K code-gen, 3K tool-calling (read-only tools), 1K
      instruction, 150 identity
  Triage (~3K): 2K synthetic routing, 100 identity, 500 edge-case routing

Usage:
    python -m src.phase2_preprocessing.scripts.build_swarm_role_datasets \\
        --config src/config/pipeline-swarm-v4a.yml --role orchestrator

    python -m src.phase2_preprocessing.scripts.build_swarm_role_datasets \\
        --config src/config/pipeline-swarm-v4a.yml --role all
"""
import argparse
import json
import logging
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from datasets import Dataset, DatasetDict, load_dataset

from src.config.nexus_role_identities import ROLE_IDENTITIES
from src.config.settings import ensure_cwd, load_config
from src.phase2_preprocessing.converters.tool_format_converter import (
    convert_fable5_messages_to_hermes,
    detect_tools_in_messages,
    validate_messages,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ALL_ROLES = ("orchestrator", "worker", "explore", "triage")

NON_ASCII_THRESHOLD = 0.05

# Read-only tools the explore agent is limited to.
EXPLORE_READ_ONLY_TOOLS = frozenset({
    "Read", "Grep", "Glob", "LS", "WebFetch", "ScopeCheck",
})

# ── Glaive parsing (shared with build_v9_mixed_dataset.py) ──────────────────

QWEN_TOOLS_TEMPLATE = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_json_lines}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""

OPENHERMES_ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system"}
EXCLUDED_CATEGORIES = {"roleplay", "song", "poem", "story"}


# ── Helpers ──────────────────────────────────────────────────────────────────


def is_english(text: str) -> bool:
    if not text:
        return True
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return (non_ascii / len(text)) < NON_ASCII_THRESHOLD


def messages_are_english(messages: list[dict]) -> bool:
    for msg in messages:
        content = msg.get("content", "") or ""
        if len(content) > 20 and not is_english(content):
            return False
    return True


def parse_glaive_system(system_text: str) -> tuple[str, list[dict]]:
    """Extract base prompt and tool definitions from a Glaive system message."""
    system_text = system_text.replace("SYSTEM: ", "", 1).strip()
    tools = []
    depth = 0
    start = None
    for ci, ch in enumerate(system_text):
        if ch == "{":
            if depth == 0:
                start = ci
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(system_text[start : ci + 1])
                    if "name" in obj and "parameters" in obj:
                        tools.append({"type": "function", "function": obj})
                except json.JSONDecodeError:
                    pass
                start = None
    base_prompt = system_text
    if tools:
        first_brace = system_text.find("{")
        if first_brace > 0:
            base_prompt = system_text[:first_brace].strip().rstrip("-").strip()
    return base_prompt, tools


def build_qwen_system_with_tools(base_prompt: str, tools: list[dict]) -> str:
    tool_lines = "\n".join(json.dumps(t) for t in tools)
    tools_block = QWEN_TOOLS_TEMPLATE.format(tool_json_lines=tool_lines)
    if base_prompt:
        return f"{base_prompt}\n\n{tools_block}"
    return tools_block


def parse_glaive_chat(chat_text: str) -> list[dict]:
    """Parse a Glaive chat string into message dicts."""
    messages = []
    parts = re.split(r"\n(?=USER:|FUNCTION RESPONSE:|ASSISTANT:)", chat_text.strip())
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("USER:"):
            content = part[5:].strip()
            if content:
                messages.append({"role": "user", "content": content})
        elif part.startswith("ASSISTANT:"):
            content = part[10:].strip().replace("<|endoftext|>", "").strip()
            if not content:
                continue
            fc_match = re.search(r"<functioncall>\s*(\{.*\})", content, re.DOTALL)
            if fc_match:
                raw = fc_match.group(1)
                raw = re.sub(
                    r"'(\{.*?\})'",
                    lambda m: '"' + m.group(1).replace('"', '\\"') + '"',
                    raw,
                )
                try:
                    call = json.loads(raw)
                except json.JSONDecodeError:
                    try:
                        call = json.loads(raw.replace("'", '"'))
                    except json.JSONDecodeError:
                        messages.append({"role": "assistant", "content": content})
                        continue
                name = call.get("name", "")
                args_raw = call.get("arguments", {})
                if isinstance(args_raw, str):
                    try:
                        args_raw = json.loads(args_raw)
                    except json.JSONDecodeError:
                        args_raw = {}
                tool_call = {"name": name, "arguments": args_raw}
                messages.append({
                    "role": "assistant",
                    "content": f"<tool_call>\n{json.dumps(tool_call)}\n</tool_call>",
                })
            else:
                messages.append({"role": "assistant", "content": content})
        elif part.startswith("FUNCTION RESPONSE:"):
            content = part[len("FUNCTION RESPONSE:") :].strip()
            messages.append({
                "role": "user",
                "content": f"<tool_response>\n{content}\n</tool_response>",
            })
    return messages


def count_tool_calls_in_messages(messages: list[dict]) -> int:
    """Count distinct <tool_call> blocks across all assistant turns."""
    count = 0
    for msg in messages:
        if msg["role"] == "assistant":
            count += msg.get("content", "").count("<tool_call>")
    return count


def _tool_names_in_messages(messages: list[dict]) -> set[str]:
    """Extract tool names referenced in <tool_call> blocks."""
    names = set()
    for msg in messages:
        if msg["role"] == "assistant":
            for m in re.finditer(
                r'<tool_call>\s*\{[^}]*"name"\s*:\s*"([^"]+)"', msg.get("content", "")
            ):
                names.add(m.group(1))
    return names


# ── Slice loaders ────────────────────────────────────────────────────────────


def load_slice_a(
    dataset_id: str = "cmndcntrlcyber/code-trainer-offsec-dataset",
    subsample_size: int = 4000,
    seed: int = 42,
) -> list[dict]:
    """Slice A: Code generation from the offsec dataset."""
    logger.info("Slice A: %s (subsample %d)", dataset_id, subsample_size)
    ds = load_dataset(dataset_id, split="train")
    logger.info("  Loaded %d rows", len(ds))
    ds = ds.shuffle(seed=seed)
    n = min(subsample_size, len(ds))
    ds = ds.select(range(n))

    records = []
    skipped = 0
    for row in ds:
        messages = row.get("messages")
        if not messages or not validate_messages(messages):
            skipped += 1
            continue
        if not messages_are_english(messages):
            skipped += 1
            continue
        records.append({
            "messages": messages,
            "slice": "code_gen",
            "source": dataset_id,
            "category": row.get("language", "unknown"),
            "has_tools": False,
            "n_turns": len(messages),
        })
    logger.info("  Slice A: %d valid records (%d skipped)", len(records), skipped)
    return records


def load_slice_b(
    dataset_id: str = "glaiveai/glaive-function-calling-v2",
    max_rows: int = 6000,
    seed: int = 42,
    filter_mode: str = "multi_step",
) -> list[dict]:
    """Slice B: Tool-calling from Glaive v2.

    filter_mode controls which records pass:
        "multi_step"  — prefer records with 2+ tool_call blocks (orchestrator)
        "single_call" — keep only records with exactly 1 tool_call per turn (worker)
        "read_only"   — keep only records whose tool names are in EXPLORE_READ_ONLY_TOOLS
        "all"         — no filtering beyond having at least one tool call
    """
    logger.info("Slice B: %s (max %d, filter=%s)", dataset_id, max_rows, filter_mode)
    ds = load_dataset(dataset_id, split="train")
    logger.info("  Loaded %d total rows", len(ds))
    ds = ds.shuffle(seed=seed)

    records = []
    skipped_no_call = 0
    skipped_parse = 0
    skipped_lang = 0
    skipped_filter = 0

    for row in ds:
        if len(records) >= max_rows:
            break

        chat = row.get("chat", "")
        if "<functioncall>" not in chat:
            skipped_no_call += 1
            continue

        system_text = row.get("system", "")
        base_prompt, tools = parse_glaive_system(system_text)
        if not tools:
            skipped_parse += 1
            continue

        messages = parse_glaive_chat(chat)
        if not messages:
            skipped_parse += 1
            continue

        system_content = build_qwen_system_with_tools(base_prompt, tools)
        if not any(m["role"] == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": system_content})

        clean_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

        if not validate_messages(clean_messages):
            skipped_parse += 1
            continue
        if not messages_are_english(clean_messages):
            skipped_lang += 1
            continue

        # Apply role-specific filtering
        tc_count = count_tool_calls_in_messages(clean_messages)

        if filter_mode == "multi_step" and tc_count < 2:
            # For orchestrator, strongly prefer multi-step but accept single
            # to fill the quota if needed — keep going and mark for later sort
            pass
        elif filter_mode == "single_call":
            # Worker: exactly one tool_call per assistant turn
            has_multi = False
            for msg in clean_messages:
                if msg["role"] == "assistant" and msg["content"].count("<tool_call>") > 1:
                    has_multi = True
                    break
            if has_multi:
                skipped_filter += 1
                continue
        elif filter_mode == "read_only":
            # Explore: only keep records whose tool names are all read-only
            tool_names = _tool_names_in_messages(clean_messages)
            if tool_names and not tool_names.issubset(EXPLORE_READ_ONLY_TOOLS):
                skipped_filter += 1
                continue

        records.append({
            "messages": clean_messages,
            "slice": "tool_calling",
            "source": dataset_id,
            "category": "function_calling",
            "has_tools": True,
            "n_turns": len(clean_messages),
            "_tc_count": tc_count,
        })

    # For orchestrator mode, sort by tool-call count descending to prefer multi-step
    if filter_mode == "multi_step":
        records.sort(key=lambda r: r["_tc_count"], reverse=True)
        records = records[:max_rows]

    # Clean up internal key
    for r in records:
        r.pop("_tc_count", None)

    logger.info(
        "  Slice B: %d records (skipped: %d no-call, %d parse, %d lang, %d filter)",
        len(records), skipped_no_call, skipped_parse, skipped_lang, skipped_filter,
    )
    return records


def load_slice_c(
    dataset_id: str = "greghavens/kimi-k3-coding-and-debugging-traces",
    max_rows: int = 3000,
    seed: int = 42,
    short_only: bool = False,
    max_turns_short: int = 8,
) -> list[dict]:
    """Slice C: Agent traces from Fable 5.

    When short_only is True (worker role), keep only traces with <= max_turns_short turns.
    """
    logger.info("Slice C: %s (max %d, short_only=%s)", dataset_id, max_rows, short_only)
    ds = load_dataset(dataset_id, split="train")
    logger.info("  Loaded %d rows", len(ds))
    ds = ds.shuffle(seed=seed)

    records = []
    skipped = 0

    for row in ds:
        if len(records) >= max_rows:
            break
        raw_messages = row.get("messages")
        if not raw_messages:
            skipped += 1
            continue
        messages = convert_fable5_messages_to_hermes(raw_messages)
        if not validate_messages(messages):
            skipped += 1
            continue
        if not messages_are_english(messages):
            skipped += 1
            continue
        if short_only and len(messages) > max_turns_short:
            skipped += 1
            continue
        records.append({
            "messages": messages,
            "slice": "agentic",
            "source": dataset_id,
            "category": row.get("category", "coding"),
            "has_tools": detect_tools_in_messages(messages),
            "n_turns": len(messages),
        })

    logger.info("  Slice C: %d records (%d skipped)", len(records), skipped)
    return records


def load_slice_d(
    dataset_id: str = "teknium/OpenHermes-2.5",
    max_rows: int = 2000,
    seed: int = 42,
) -> list[dict]:
    """Slice D: English instruction following from OpenHermes-2.5."""
    logger.info("Slice D: %s (max %d)", dataset_id, max_rows)
    ds = load_dataset(dataset_id, split="train")
    logger.info("  Loaded %d total rows", len(ds))

    indices = list(range(len(ds)))
    random.seed(seed)
    random.shuffle(indices)

    records = []
    skipped = 0

    for idx in indices:
        if len(records) >= max_rows:
            break
        row = ds[idx]
        category = row.get("category", "")
        if category in EXCLUDED_CATEGORIES:
            skipped += 1
            continue
        convs = row.get("conversations", [])
        if not convs:
            skipped += 1
            continue

        messages = []
        sys_prompt = row.get("system_prompt")
        if sys_prompt and isinstance(sys_prompt, str) and sys_prompt.strip():
            messages.append({"role": "system", "content": sys_prompt.strip()})
        for turn in convs:
            role = OPENHERMES_ROLE_MAP.get(turn.get("from", ""), turn.get("from", ""))
            content = turn.get("value", "")
            if not content:
                continue
            messages.append({"role": role, "content": content})
        if not validate_messages(messages):
            skipped += 1
            continue
        if not messages_are_english(messages):
            skipped += 1
            continue
        records.append({
            "messages": messages,
            "slice": "instruction",
            "source": dataset_id,
            "category": category or "general",
            "has_tools": False,
            "n_turns": len(messages),
        })

    logger.info("  Slice D: %d records (%d skipped)", len(records), skipped)
    return records


def load_slice_e(
    identity_path: str,
    max_rows: int = 400,
    seed: int = 42,
) -> list[dict]:
    """Slice E: Pre-generated identity/persona examples from JSONL."""
    path = Path(identity_path)
    if not path.exists():
        logger.warning("Identity examples not found at %s; skipping Slice E", path)
        return []

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not validate_messages(rec.get("messages", [])):
                continue
            records.append(rec)

    if len(records) > max_rows:
        rng = random.Random(seed)
        rng.shuffle(records)
        records = records[:max_rows]

    logger.info("  Slice E: %d identity examples from %s", len(records), path)
    return records


# ── Synthetic data generators ────────────────────────────────────────────────


# Templates for orchestrator delegation examples (Slice F)
DELEGATION_TASK_TEMPLATES = [
    (
        "Conduct a full penetration test against {target}. Start with recon, "
        "identify vulnerabilities, attempt exploitation, and document findings."
    ),
    (
        "Enumerate the attack surface of {target}. Map all exposed services, "
        "check for default credentials, and identify the top three attack vectors."
    ),
    (
        "We need a security audit of the web application on {target}. Test for "
        "OWASP Top 10 vulnerabilities and write a findings summary."
    ),
    (
        "Investigate the compromised host {target}. Determine initial access vector, "
        "identify persistence mechanisms, and document the attack timeline."
    ),
    (
        "Perform privilege escalation on {target} from the www-data user. "
        "Check SUID binaries, cron jobs, and kernel vulnerabilities."
    ),
    (
        "Conduct OSINT on the target organization. Enumerate subdomains, harvest "
        "email addresses, and identify exposed services for {target}."
    ),
    (
        "Test the API endpoints on {target} for authentication bypass, IDOR, "
        "and injection vulnerabilities."
    ),
    (
        "Map the internal network reachable from {target}. Identify other live hosts, "
        "shared services, and potential lateral movement paths."
    ),
    (
        "Analyze the source code in /var/www/html on {target} for hardcoded "
        "credentials, SQL injection, and insecure deserialization."
    ),
    (
        "Conduct a password audit: spray common credentials against SSH, SMB, "
        "and the web login on {target}."
    ),
]

DELEGATION_SUBTASK_POOL = [
    ("worker", "Run nmap -sC -sV on {target} and return the parsed results."),
    ("worker", "Run gobuster dir against http://{target}/ with the common wordlist."),
    ("worker", "Run nikto against http://{target}/ and summarize the findings."),
    ("worker", "Execute linpeas.sh on {target} and extract privilege escalation vectors."),
    ("worker", "Run sqlmap against the login form on http://{target}/login."),
    ("worker", "Run nuclei with the cves template set against {target}."),
    ("worker", "Check for default credentials on SMB with crackmapexec."),
    ("worker", "Parse the nmap XML output and list services by risk priority."),
    ("explore", "Search for .env and config files in /var/www/ on {target}."),
    ("explore", "Read /etc/passwd and identify non-system user accounts."),
    ("explore", "Grep the web application source for hardcoded secrets."),
    ("explore", "Check the .git directory on {target} for exposed commit history."),
    ("explore", "List SUID binaries on {target} and classify each by GTFOBins status."),
    ("explore", "Enumerate subdomains of {target} using passive DNS sources."),
    ("explore", "Fetch robots.txt and sitemap.xml from http://{target}/."),
    ("explore", "Read the Apache/nginx config for virtual host entries."),
]

DELEGATION_TARGETS = [
    "10.10.10.5", "10.10.10.42", "10.129.45.67", "192.168.1.100",
    "target.htb", "app.example.com", "172.16.0.10", "vulnerable.local",
    "10.10.11.13", "192.168.56.101",
]


def generate_delegation_examples(
    rng: random.Random,
    count: int = 500,
    role_identity: str = "",
) -> list[dict]:
    """Slice F: Synthetic delegation examples for the orchestrator.

    Each example shows the orchestrator receiving a complex task and
    decomposing it into delegated subtasks for worker/explore agents.
    """
    logger.info("Slice F: generating %d synthetic delegation examples", count)
    records = []

    for i in range(count):
        target = rng.choice(DELEGATION_TARGETS)
        task_template = rng.choice(DELEGATION_TASK_TEMPLATES)
        user_request = task_template.format(target=target)

        # Pick 2-4 subtasks to delegate
        n_subtasks = rng.randint(2, 4)
        subtasks = rng.sample(DELEGATION_SUBTASK_POOL, min(n_subtasks, len(DELEGATION_SUBTASK_POOL)))

        # Build the orchestrator's decomposition response
        lines = [
            "I'll decompose this into targeted subtasks and delegate to the appropriate agents.\n",
            "**Task decomposition:**\n",
        ]
        for j, (agent, desc) in enumerate(subtasks, 1):
            filled = desc.format(target=target)
            lines.append(f"{j}. **{agent}**: {filled}")

        lines.append(
            f"\nStarting with step 1. Delegating to the {subtasks[0][0]} agent."
        )

        # Build a tool_call for the first delegation
        first_agent, first_desc = subtasks[0]
        delegation_call = {
            "name": "Task",
            "arguments": {
                "profile": first_agent,
                "prompt": first_desc.format(target=target),
            },
        }
        lines.append(
            f"\n<tool_call>\n{json.dumps(delegation_call)}\n</tool_call>"
        )

        assistant_content = "\n".join(lines)

        messages = [
            {"role": "user", "content": user_request},
            {"role": "assistant", "content": assistant_content},
        ]
        if role_identity:
            messages.insert(0, {"role": "system", "content": role_identity})

        records.append({
            "messages": messages,
            "slice": "delegation",
            "source": "synthetic_delegation",
            "category": "orchestrator_delegation",
            "has_tools": True,
            "n_turns": len(messages),
        })

    logger.info("  Slice F: %d delegation examples generated", len(records))
    return records


# Templates for triage routing examples (Slices H and I)

TRIAGE_CATEGORIES = {
    "recon": {
        "route": "nexus-explore",
        "templates": [
            "Scan {target} for open ports.",
            "Enumerate subdomains of {domain}.",
            "What services are running on {target}?",
            "Run nmap against {target}.",
            "Find all web servers on the 10.10.10.0/24 subnet.",
            "Check if {target} has any exposed .git directories.",
            "Do passive recon on {domain} and list DNS records.",
            "What technology stack is {target} running?",
            "Fingerprint the web application on {target}:8080.",
            "List all open ports and services on {target}.",
            "Run a UDP scan on {target}.",
            "Enumerate SMB shares on {target}.",
            "Check for DNS zone transfer on {domain}.",
            "Map the attack surface of {target}.",
            "Identify the CMS running on http://{target}/.",
        ],
    },
    "exploit": {
        "route": "nexus-orchestrator",
        "templates": [
            "Exploit the SQL injection on {target}/login.",
            "I found CVE-2021-41773 on {target}. Get a shell.",
            "The file upload on {target} is unrestricted — exploit it.",
            "Exploit the SUID binary on {target} to get root.",
            "Chain the SSRF and LFI on {target} for RCE.",
            "Use the credentials admin:password123 to compromise {target}.",
            "Exploit the deserialization vulnerability on {target}:8443.",
            "The kernel is 4.4.0 — find and run a privesc exploit on {target}.",
            "Exploit the log4j vulnerability on {target}:8080.",
            "There's a buffer overflow in the custom service on {target}:9999.",
        ],
    },
    "code-review": {
        "route": "nexus-explore",
        "templates": [
            "Review the source code in /var/www/html for vulnerabilities.",
            "Audit {path} for SQL injection patterns.",
            "Check {path} for hardcoded credentials.",
            "Is this PHP code vulnerable to command injection?",
            "Review the authentication logic in {path}.",
            "Check this Python script for insecure deserialization.",
            "Audit the API route handlers for IDOR vulnerabilities.",
            "Does this JavaScript code have any XSS sinks?",
            "Review the sudo configuration for privilege escalation risks.",
            "Check the Dockerfile for security misconfigurations.",
        ],
    },
    "report": {
        "route": "nexus-orchestrator",
        "templates": [
            "Write the penetration test report for the {target} engagement.",
            "Summarize all findings from the assessment.",
            "Generate an executive summary of the security audit.",
            "Document the attack path from initial access to domain admin.",
            "Create a remediation priority list for the findings.",
            "Write up the SQL injection finding with evidence.",
            "Prepare the final deliverable for the client.",
            "Compile the engagement notes into a formal report.",
        ],
    },
    "admin": {
        "route": "nexus-worker",
        "templates": [
            "Set up a reverse shell listener on port 4444.",
            "Transfer linpeas.sh to {target}.",
            "Start a Python HTTP server on port 8000.",
            "Create a new engagement directory for {target}.",
            "Save these credentials to the loot file.",
            "Update the engagement notes with the new findings.",
            "Compress and exfiltrate the collected data.",
            "Set up port forwarding through the compromised host.",
            "Configure proxychains for pivoting through {target}.",
            "Install the required tools on the attack machine.",
        ],
    },
    "unknown": {
        "route": "nexus-orchestrator",
        "templates": [
            "What's the weather like today?",
            "Tell me a joke.",
            "How do I cook pasta?",
            "What's the capital of France?",
            "Explain quantum computing.",
            "Help me write a poem.",
            "What is machine learning?",
            "Can you recommend a good book?",
        ],
    },
}

TRIAGE_TARGETS = [
    "10.10.10.5", "10.10.10.42", "192.168.1.100", "target.htb",
    "10.129.45.67", "172.16.0.10", "app.example.com", "10.10.11.13",
]
TRIAGE_DOMAINS = [
    "example.com", "target.htb", "corp.local", "vulnerable.org",
    "testfire.net", "hackthebox.eu", "tryhackme.com",
]
TRIAGE_PATHS = [
    "/var/www/html/index.php",
    "/opt/app/server.py",
    "/home/user/scripts/backup.sh",
    "/etc/nginx/nginx.conf",
    "src/auth/login.js",
    "app/models/user.rb",
    "api/routes/upload.py",
]


def generate_routing_examples(
    rng: random.Random,
    count: int = 2000,
    role_identity: str = "",
) -> list[dict]:
    """Slice H: Synthetic routing/classification examples for triage.

    Each example has a user request and the triage agent's JSON classification.
    """
    logger.info("Slice H: generating %d synthetic routing examples", count)
    records = []

    # Distribute roughly evenly across categories, with more recon/exploit
    weights = {
        "recon": 0.25, "exploit": 0.25, "code-review": 0.15,
        "report": 0.10, "admin": 0.15, "unknown": 0.10,
    }
    category_counts = {
        cat: max(1, int(count * w)) for cat, w in weights.items()
    }
    # Adjust to hit target exactly
    diff = count - sum(category_counts.values())
    if diff > 0:
        category_counts["recon"] += diff
    elif diff < 0:
        category_counts["unknown"] = max(1, category_counts["unknown"] + diff)

    for category, cat_info in TRIAGE_CATEGORIES.items():
        n = category_counts.get(category, 0)
        templates = cat_info["templates"]
        route = cat_info["route"]

        for _ in range(n):
            template = rng.choice(templates)
            target = rng.choice(TRIAGE_TARGETS)
            domain = rng.choice(TRIAGE_DOMAINS)
            path = rng.choice(TRIAGE_PATHS)
            user_text = template.format(target=target, domain=domain, path=path)

            confidence = round(rng.uniform(0.75, 0.99), 2)
            response_obj = {
                "category": category,
                "route": route,
                "confidence": confidence,
            }

            messages = [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": json.dumps(response_obj)},
            ]
            if role_identity:
                messages.insert(0, {"role": "system", "content": role_identity})

            records.append({
                "messages": messages,
                "slice": "routing",
                "source": "synthetic_routing",
                "category": f"triage_{category}",
                "has_tools": False,
                "n_turns": len(messages),
            })

    rng.shuffle(records)
    logger.info("  Slice H: %d routing examples generated", len(records))
    return records


# Edge-case templates for triage (Slice I)
EDGE_CASE_TEMPLATES = [
    # Ambiguous: could be recon or exploit
    {
        "request": "Check if {target} is vulnerable to CVE-2021-44228.",
        "category": "recon",
        "route": "nexus-explore",
        "reason": "Checking vulnerability presence is reconnaissance, not exploitation.",
    },
    {
        "request": "Test the login form on {target} for weak credentials.",
        "category": "exploit",
        "route": "nexus-orchestrator",
        "reason": "Credential testing modifies state and is multi-step.",
    },
    # Ambiguous: admin or exploit
    {
        "request": "Set up a pivot through {target} to reach the internal network.",
        "category": "admin",
        "route": "nexus-worker",
        "reason": "Setting up a pivot is infrastructure work, not exploitation.",
    },
    # Ambiguous: report or code-review
    {
        "request": "Document the XSS vulnerability I found on {target} and suggest a fix.",
        "category": "report",
        "route": "nexus-orchestrator",
        "reason": "Documentation with remediation requires orchestrator synthesis.",
    },
    # Multi-category requests
    {
        "request": "Scan {target}, exploit any SQLi you find, and write it up.",
        "category": "exploit",
        "route": "nexus-orchestrator",
        "reason": "Multi-step request with scan+exploit+report requires orchestrator.",
    },
    {
        "request": "Read the source code at {path} and find a way to exploit it.",
        "category": "exploit",
        "route": "nexus-orchestrator",
        "reason": "Code review leading to exploitation is multi-step.",
    },
    # Unclear phrasing
    {
        "request": "Take a look at {target}.",
        "category": "recon",
        "route": "nexus-explore",
        "reason": "Vague request defaults to reconnaissance.",
    },
    {
        "request": "Do your thing on {target}.",
        "category": "recon",
        "route": "nexus-orchestrator",
        "reason": "Completely ambiguous — route to orchestrator for planning.",
    },
    {
        "request": "What can we do with {target}?",
        "category": "recon",
        "route": "nexus-explore",
        "reason": "Exploratory question maps to reconnaissance.",
    },
    {
        "request": "Help with {target}.",
        "category": "unknown",
        "route": "nexus-orchestrator",
        "reason": "Insufficient context — route to orchestrator for clarification.",
    },
    # Off-topic edge cases
    {
        "request": "Can you help me with my homework about {domain}?",
        "category": "unknown",
        "route": "nexus-orchestrator",
        "reason": "Non-security request.",
    },
    {
        "request": "What is your purpose?",
        "category": "unknown",
        "route": "nexus-orchestrator",
        "reason": "Identity question — not a security task.",
    },
    # Chained requests
    {
        "request": "First scan {target}, then enumerate SMB, then try pass-the-hash.",
        "category": "exploit",
        "route": "nexus-orchestrator",
        "reason": "Explicitly chained multi-step task requires orchestrator.",
    },
    {
        "request": "Run nmap and gobuster in parallel on {target}.",
        "category": "recon",
        "route": "nexus-orchestrator",
        "reason": "Parallel task coordination requires orchestrator.",
    },
    # Single-tool tasks phrased as complex
    {
        "request": "I need a comprehensive port scan of {target} with all scripts.",
        "category": "recon",
        "route": "nexus-worker",
        "reason": "Despite verbose phrasing, this is a single nmap execution.",
    },
    {
        "request": "Just cat /etc/passwd on {target}.",
        "category": "recon",
        "route": "nexus-explore",
        "reason": "Single read-only file operation.",
    },
]


def generate_edge_case_routing(
    rng: random.Random,
    count: int = 500,
    role_identity: str = "",
) -> list[dict]:
    """Slice I: Edge-case routing examples for triage.

    These are harder classification cases with ambiguous or multi-category requests.
    Each example includes a reasoning field in the response to teach the triage
    model when to route to orchestrator vs. worker vs. explore.
    """
    logger.info("Slice I: generating %d edge-case routing examples", count)
    records = []

    for i in range(count):
        template = rng.choice(EDGE_CASE_TEMPLATES)
        target = rng.choice(TRIAGE_TARGETS)
        domain = rng.choice(TRIAGE_DOMAINS)
        path = rng.choice(TRIAGE_PATHS)

        request_text = template["request"].format(
            target=target, domain=domain, path=path,
        )

        confidence = round(rng.uniform(0.55, 0.90), 2)
        response_obj = {
            "category": template["category"],
            "route": template["route"],
            "confidence": confidence,
            "reasoning": template["reason"],
        }

        messages = [
            {"role": "user", "content": request_text},
            {"role": "assistant", "content": json.dumps(response_obj)},
        ]
        if role_identity:
            messages.insert(0, {"role": "system", "content": role_identity})

        records.append({
            "messages": messages,
            "slice": "routing_edge",
            "source": "synthetic_routing_edge",
            "category": f"triage_edge_{template['category']}",
            "has_tools": False,
            "n_turns": len(messages),
        })

    logger.info("  Slice I: %d edge-case routing examples generated", len(records))
    return records


# ── System prompt injection ──────────────────────────────────────────────────


def inject_role_system_prompt(
    records: list[dict],
    role_identity: str,
    preserve_tools: bool = True,
) -> list[dict]:
    """Replace or prepend the role-specific identity into every record's system message.

    When preserve_tools is True and the existing system message contains a
    <tools> block, the role identity is prepended before the tools block.
    """
    injected = 0

    for record in records:
        messages = record.get("messages", [])
        if not messages:
            continue

        sys_idx = None
        for i, msg in enumerate(messages):
            if msg["role"] == "system":
                sys_idx = i
                break

        if sys_idx is not None:
            old_content = messages[sys_idx]["content"]
            if preserve_tools and "<tools>" in old_content:
                header_idx = old_content.rfind("# Tools", 0, old_content.index("<tools>"))
                if header_idx > 0:
                    tools_section = old_content[header_idx:]
                else:
                    tools_section = old_content[old_content.index("<tools>"):]
                messages[sys_idx]["content"] = f"{role_identity}\n\n{tools_section}"
            else:
                messages[sys_idx]["content"] = role_identity
            injected += 1
        else:
            messages.insert(0, {"role": "system", "content": role_identity})
            record["n_turns"] = len(messages)
            injected += 1

    logger.info("  System prompt injection: %d records updated", injected)
    return records


# ── Post-processing ──────────────────────────────────────────────────────────


def strip_after_tool_call_close(content: str) -> str:
    last_close = content.rfind("</tool_call>")
    if last_close == -1:
        return content
    return content[: last_close + len("</tool_call>")]


def enforce_stop_after_tag(records: list[dict]) -> list[dict]:
    stripped = 0
    for record in records:
        for msg in record["messages"]:
            if msg["role"] == "assistant" and "</tool_call>" in msg["content"]:
                cleaned = strip_after_tool_call_close(msg["content"])
                if cleaned != msg["content"]:
                    stripped += 1
                msg["content"] = cleaned
    logger.info("  Stop-after-tag: stripped trailing text from %d assistant turns", stripped)
    return records


def validate_tag_completeness(records: list[dict]) -> list[dict]:
    valid = []
    dropped = 0
    for record in records:
        ok = True
        for msg in record["messages"]:
            if msg["role"] == "assistant":
                opens = msg["content"].count("<tool_call>")
                closes = msg["content"].count("</tool_call>")
                if opens != closes:
                    ok = False
                    break
        if ok:
            valid.append(record)
        else:
            dropped += 1
    logger.info("  Tag validation: dropped %d records with unmatched tags", dropped)
    return valid


# ── Dataset assembly ─────────────────────────────────────────────────────────


def build_dataset_dict(
    records: list[dict],
    seed: int = 42,
    val_ratio: float = 0.1,
) -> DatasetDict:
    random.seed(seed)
    random.shuffle(records)

    val_size = int(len(records) * val_ratio)
    train_records = records[val_size:]
    val_records = records[:val_size]

    logger.info("Split: %d train, %d validation", len(train_records), len(val_records))

    return DatasetDict({
        "train": Dataset.from_list(train_records),
        "validation": Dataset.from_list(val_records),
    })


def compute_statistics(dataset_dict: DatasetDict) -> dict:
    stats = {}
    for split_name, ds in dataset_dict.items():
        slice_counts = Counter(ds["slice"])
        tool_counts = Counter(str(x) for x in ds["has_tools"])
        n_turns = ds["n_turns"]
        category_counts = Counter(ds["category"])
        stats[split_name] = {
            "total": len(ds),
            "by_slice": dict(slice_counts),
            "has_tools": dict(tool_counts),
            "n_turns": {
                "min": min(n_turns),
                "max": max(n_turns),
                "mean": round(sum(n_turns) / len(n_turns), 1),
            },
            "top_categories": dict(category_counts.most_common(20)),
        }
    return stats


# ── Role-specific build functions ────────────────────────────────────────────


def build_orchestrator(config: dict, args) -> list[dict]:
    """~12K rows: 4K code-gen, 6K tool-calling (multi-step), 3K agent traces,
    2K instruction, 400 identity, 500 delegation."""
    logger.info("Building ORCHESTRATOR dataset")
    swarm_cfg = config.get("swarm_roles", {}).get("orchestrator", {})
    seed = args.seed

    records_a = load_slice_a(
        swarm_cfg.get("slice_a_source", "cmndcntrlcyber/code-trainer-offsec-dataset"),
        swarm_cfg.get("slice_a_size", 4000), seed,
    )
    records_b = load_slice_b(
        swarm_cfg.get("slice_b_source", "glaiveai/glaive-function-calling-v2"),
        swarm_cfg.get("slice_b_size", 6000), seed, filter_mode="multi_step",
    )
    records_c = load_slice_c(
        swarm_cfg.get("slice_c_source", "greghavens/kimi-k3-coding-and-debugging-traces"),
        swarm_cfg.get("slice_c_size", 3000), seed,
    )
    records_d = load_slice_d(
        swarm_cfg.get("slice_d_source", "teknium/OpenHermes-2.5"),
        swarm_cfg.get("slice_d_size", 2000), seed,
    )

    identity_path = args.identity_examples or swarm_cfg.get(
        "identity_path", "data/identity_examples/nexus_identity.jsonl",
    )
    records_e = load_slice_e(identity_path, swarm_cfg.get("slice_e_size", 400), seed)

    rng = random.Random(seed)
    role_identity = ROLE_IDENTITIES["orchestrator"] if args.inject_system_prompt else ""
    records_f = generate_delegation_examples(
        rng, swarm_cfg.get("slice_f_size", 500), role_identity,
    )

    all_records = records_a + records_b + records_c + records_d + records_e + records_f
    logger.info(
        "Orchestrator total: %d (A=%d B=%d C=%d D=%d E=%d F=%d)",
        len(all_records), len(records_a), len(records_b), len(records_c),
        len(records_d), len(records_e), len(records_f),
    )
    return all_records


def build_worker(config: dict, args) -> list[dict]:
    """~8K rows: 2K code-gen, 5K tool-calling (single-call), 1K short agent traces,
    1K instruction, 200 identity."""
    logger.info("Building WORKER dataset")
    swarm_cfg = config.get("swarm_roles", {}).get("worker", {})
    seed = args.seed

    records_a = load_slice_a(
        swarm_cfg.get("slice_a_source", "cmndcntrlcyber/code-trainer-offsec-dataset"),
        swarm_cfg.get("slice_a_size", 2000), seed,
    )
    records_b = load_slice_b(
        swarm_cfg.get("slice_b_source", "glaiveai/glaive-function-calling-v2"),
        swarm_cfg.get("slice_b_size", 5000), seed, filter_mode="single_call",
    )
    records_c = load_slice_c(
        swarm_cfg.get("slice_c_source", "greghavens/kimi-k3-coding-and-debugging-traces"),
        swarm_cfg.get("slice_c_size", 1000), seed, short_only=True,
    )
    records_d = load_slice_d(
        swarm_cfg.get("slice_d_source", "teknium/OpenHermes-2.5"),
        swarm_cfg.get("slice_d_size", 1000), seed,
    )

    identity_path = args.identity_examples or swarm_cfg.get(
        "identity_path", "data/identity_examples/nexus_identity.jsonl",
    )
    records_e = load_slice_e(identity_path, swarm_cfg.get("slice_e_size", 200), seed)

    all_records = records_a + records_b + records_c + records_d + records_e
    logger.info(
        "Worker total: %d (A=%d B=%d C=%d D=%d E=%d)",
        len(all_records), len(records_a), len(records_b), len(records_c),
        len(records_d), len(records_e),
    )
    return all_records


def build_explore(config: dict, args) -> list[dict]:
    """~5K rows: 1K code-gen, 3K tool-calling (read-only), 1K instruction,
    150 identity."""
    logger.info("Building EXPLORE dataset")
    swarm_cfg = config.get("swarm_roles", {}).get("explore", {})
    seed = args.seed

    records_a = load_slice_a(
        swarm_cfg.get("slice_a_source", "cmndcntrlcyber/code-trainer-offsec-dataset"),
        swarm_cfg.get("slice_a_size", 1000), seed,
    )
    records_b = load_slice_b(
        swarm_cfg.get("slice_b_source", "glaiveai/glaive-function-calling-v2"),
        swarm_cfg.get("slice_b_size", 3000), seed, filter_mode="read_only",
    )
    records_d = load_slice_d(
        swarm_cfg.get("slice_d_source", "teknium/OpenHermes-2.5"),
        swarm_cfg.get("slice_d_size", 1000), seed,
    )

    identity_path = args.identity_examples or swarm_cfg.get(
        "identity_path", "data/identity_examples/nexus_identity.jsonl",
    )
    records_e = load_slice_e(identity_path, swarm_cfg.get("slice_e_size", 150), seed)

    all_records = records_a + records_b + records_d + records_e
    logger.info(
        "Explore total: %d (A=%d B=%d D=%d E=%d)",
        len(all_records), len(records_a), len(records_b),
        len(records_d), len(records_e),
    )
    return all_records


def build_triage(config: dict, args) -> list[dict]:
    """~3K rows: 2K routing, 100 identity, 500 edge-case routing."""
    logger.info("Building TRIAGE dataset")
    swarm_cfg = config.get("swarm_roles", {}).get("triage", {})
    seed = args.seed

    rng = random.Random(seed)
    role_identity = ROLE_IDENTITIES["triage"] if args.inject_system_prompt else ""

    records_h = generate_routing_examples(
        rng, swarm_cfg.get("slice_h_size", 2000), role_identity,
    )

    identity_path = args.identity_examples or swarm_cfg.get(
        "identity_path", "data/identity_examples/nexus_identity.jsonl",
    )
    records_e = load_slice_e(identity_path, swarm_cfg.get("slice_e_size", 100), seed)

    records_i = generate_edge_case_routing(
        rng, swarm_cfg.get("slice_i_size", 500), role_identity,
    )

    all_records = records_h + records_e + records_i
    logger.info(
        "Triage total: %d (H=%d E=%d I=%d)",
        len(all_records), len(records_h), len(records_e), len(records_i),
    )
    return all_records


ROLE_BUILDERS = {
    "orchestrator": build_orchestrator,
    "worker": build_worker,
    "explore": build_explore,
    "triage": build_triage,
}


# ── Main ─────────────────────────────────────────────────────────────────────


def build_and_save_role(role: str, config: dict, args):
    """Build, post-process, and save one role's dataset."""
    logger.info("=" * 60)
    logger.info("Building dataset for role: %s", role)
    logger.info("=" * 60)

    records = ROLE_BUILDERS[role](config, args)

    # Post-processing: stop-after-tag + tag validation
    records = enforce_stop_after_tag(records)
    records = validate_tag_completeness(records)
    logger.info("After post-processing: %d records", len(records))

    # System prompt injection (skip for records that already have role identity,
    # e.g. synthetic delegation/routing records)
    if args.inject_system_prompt:
        role_identity = ROLE_IDENTITIES[role]
        records = inject_role_system_prompt(records, role_identity, preserve_tools=True)

    # Build HF dataset
    val_ratio = float(config.get("swarm_roles", {}).get("val_split", 0.1))
    dataset_dict = build_dataset_dict(records, args.seed, val_ratio)

    # Statistics
    stats = compute_statistics(dataset_dict)
    for split_name, s in stats.items():
        logger.info(
            "  %s: %d rows | slices: %s | turns: %s",
            split_name, s["total"], s["by_slice"], s["n_turns"],
        )

    # Save locally
    project_root = Path(__file__).resolve().parents[3]
    output_dir = Path(args.output_dir) / role
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
        logger.info("Cleared existing output dir: %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving to %s", output_dir)
    ensure_cwd()
    dataset_dict.save_to_disk(str(output_dir))

    stats_path = output_dir / "statistics.json"
    stats_path.write_text(json.dumps(stats, indent=2))

    # Push to Hub
    if args.push_to_hub:
        from huggingface_hub import HfApi, create_repo

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not token:
            logger.error("HF_TOKEN not set; skipping Hub push for %s.", role)
            return

        hf_username = os.environ.get("HF_USERNAME", "atlas-institute")
        hub_repo = (
            config.get("swarm_roles", {}).get(role, {}).get("hub_repo")
            or f"{hf_username}/code-trainer-v11-{role}"
        )

        logger.info("Pushing %s to %s", role, hub_repo)
        create_repo(hub_repo, token=token, repo_type="dataset", private=False, exist_ok=True)
        dataset_dict.push_to_hub(
            hub_repo, token=token,
            commit_message=f"V11 swarm role dataset: {role}",
        )

        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(stats_path),
            path_in_repo="statistics.json",
            repo_id=hub_repo,
            repo_type="dataset",
            commit_message=f"Upload {role} statistics",
        )
        logger.info("Pushed: https://huggingface.co/datasets/%s", hub_repo)


def main():
    parser = argparse.ArgumentParser(
        description="Build role-specific SFT datasets for the Nexus inference swarm"
    )
    parser.add_argument("--config", required=True,
                        help="YAML config file (e.g. src/config/pipeline-swarm-v4a.yml)")
    parser.add_argument("--role", required=True,
                        choices=list(ALL_ROLES) + ["all"],
                        help="Swarm role to build (or 'all' for all roles)")
    parser.add_argument("--output-dir", default="data/swarm_role_datasets",
                        help="Base output directory (role name appended)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--identity-examples", default=None,
                        help="JSONL file of identity/persona examples (Slice E)")
    parser.add_argument("--inject-system-prompt", action="store_true", default=True,
                        help="Replace system prompts with role-specific identity (default: True)")
    parser.add_argument("--no-inject-system-prompt", dest="inject_system_prompt",
                        action="store_false",
                        help="Disable system prompt injection")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push to HuggingFace Hub as ${HF_USERNAME}/code-trainer-v11-{role}")
    args = parser.parse_args()

    config = load_config(args.config)

    roles = list(ALL_ROLES) if args.role == "all" else [args.role]

    logger.info("Swarm Role Dataset Builder")
    logger.info("  Config: %s", args.config)
    logger.info("  Roles: %s", ", ".join(roles))
    logger.info("  System prompt injection: %s", args.inject_system_prompt)
    logger.info("  Push to Hub: %s", args.push_to_hub)

    for role in roles:
        build_and_save_role(role, config, args)

    logger.info("Done. Built %d role dataset(s).", len(roles))


if __name__ == "__main__":
    main()
