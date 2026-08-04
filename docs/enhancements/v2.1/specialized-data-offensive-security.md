# v2.1 Enhancement: Specialized Data — Offensive Security

## Summary

Extend the Phase 1 data collection pipeline to target **offensive security repositories** specifically, producing a curated dataset of exploit code, red team tooling, C2 frameworks, post-exploitation scripts, and security research code. This enhancement adds a specialized scraper profile that replaces the general-purpose 8-language sweep with security-focused repository discovery, keyword filtering, and domain-aware quality scoring.

## Motivation

The base Code-Trainer pipeline collects general-purpose code across 8 languages. For the rtpi project's security operations use case, a specialized dataset of offensive security code will produce significantly better results when fine-tuning Qwen models for:

- Generating exploit proof-of-concepts
- Writing red team automation scripts
- Understanding C2 framework patterns
- Producing post-exploitation tooling
- Analyzing vulnerability patterns across codebases

A model trained on curated offensive security code will outperform a general-purpose code model for these tasks, even at smaller parameter counts.

## Data Sources

### Primary: cmndcntrlcyber Repositories

The `cmndcntrlcyber` GitHub organization serves as the seed source. A dedicated scraper enumerates all repositories under this account and classifies them by offensive security subdomain.

### Secondary: Curated Offensive Security Organizations

Expand collection beyond the seed source to high-quality public offensive security repositories:

| Organization / User | Focus Area | Example Repos |
|---|---|---|
| `cmndcntrlcyber` | Red team operations, C2, automation | Seed source — include all |
| `BC-SECURITY` | Empire C2 framework | Empire, Starkiller |
| `BishopFox` | Offensive tools | Sliver, Cloudfox |
| `fortra` (Cobaltstrike) | C2 references | Open-source C2 tooling |
| `projectdiscovery` | Recon & scanning | Nuclei, httpx, subfinder |
| `rapid7` | Exploitation frameworks | Metasploit-framework |
| `offensive-security` | Exploit DB, tools | exploitdb, kali-linux |
| `swisskyrepo` | Payload references | PayloadsAllTheThings |
| `The-Z-Labs` | Exploit development | bof-launcher |
| `gentilkiwi` | Credential tools | Mimikatz |
| `GhostPack` | .NET offensive tools | Seatbelt, Rubeus, SharpUp |
| `impacket` | Network protocols | Impacket |
| `SecureAuthCorp` | Auth exploitation | Impacket (legacy) |
| `S3cur3Th1sSh1t` | Red team scripts | PowerSharpPack, WinPwn |

### Tertiary: Topic-Based Discovery

Search GitHub by topic tags to discover repos not in the curated list. Topics are organized by offensive security kill chain phase and technique category:

```
# --- Exploitation & Vulnerability Research ---
exploit, exploit-development, exploitdb, vulnerability, vulnerability-research,
0day, zero-day, cve, poc, proof-of-concept, rce, remote-code-execution,
buffer-overflow, heap-overflow, stack-overflow, use-after-free, type-confusion,
integer-overflow, format-string, race-condition, deserialization,
sql-injection, sqli, xss, cross-site-scripting, ssrf, lfi, rfi,
command-injection, xxe, ssti, idor, insecure-deserialization,
kernel-exploit, browser-exploit, ios-exploit, android-exploit,
windows-exploit, linux-exploit, macos-exploit,

# --- Command & Control (C2) ---
c2, c2-framework, command-and-control, rat, remote-access-trojan,
beacon, implant, agent, listener, stager, teamserver,
cobalt-strike, metasploit, empire, sliver, havoc, mythic, covenant,
brute-ratel, nighthawk, poshc2, merlin-c2, deimosC2,
dns-c2, http-c2, smb-c2, icmp-c2, websocket-c2,
malleable-profile, redirector, infrastructure,

# --- Reconnaissance & OSINT ---
recon, reconnaissance, enumeration, scanner, discovery,
osint, open-source-intelligence, information-gathering,
subdomain-enumeration, subdomain-discovery, subdomain-takeover,
port-scan, port-scanner, service-enumeration,
fingerprint, fingerprinting, banner-grabbing,
asset-discovery, attack-surface, external-recon,
dns-enumeration, dns-recon, zone-transfer,
web-recon, directory-bruteforce, content-discovery,
google-dork, shodan, censys, zoomeye, fofa,
email-harvesting, username-enumeration, social-engineering-recon,
network-mapping, arp-scan, host-discovery,

# --- Initial Access & Phishing ---
initial-access, phishing, spearphishing, social-engineering,
payload-delivery, dropper, downloader, macro, vba-macro,
hta-payload, iso-payload, lnk-payload, msi-payload,
html-smuggling, qr-phishing, usb-attack, rubber-ducky,
watering-hole, drive-by-download, supply-chain-attack,
typosquatting, dependency-confusion,

# --- Post-Exploitation ---
post-exploitation, post-exploit, lateral-movement, pivoting,
persistence, backdoor, rootkit, bootkit, webshell,
privilege-escalation, privesc, lpe, local-privilege-escalation,
credential-access, credential-dumping, credential-harvesting,
loot, data-exfiltration, exfiltration, staging,
token-manipulation, token-impersonation, token-theft,
process-injection, dll-injection, process-hollowing,
thread-hijacking, apc-injection, early-bird-injection,
named-pipe-impersonation, service-exploitation,

# --- Active Directory & Identity ---
active-directory, ad-attack, ad-exploitation, ad-enumeration,
kerberos, kerberoasting, as-rep-roasting, golden-ticket, silver-ticket,
diamond-ticket, sapphire-ticket, pass-the-ticket,
ntlm, pass-the-hash, ntlm-relay, relay-attack,
dcsync, dcshadow, sid-history, acl-abuse, dacl-abuse,
bloodhound, sharphound, ldap-enumeration, group-policy-abuse,
constrained-delegation, unconstrained-delegation, rbcd,
resource-based-constrained-delegation,
certificate-abuse, adcs, certipy, shadow-credentials,
azure-ad, entra-id, cloud-identity, oauth-abuse,

# --- Evasion & Defense Bypass ---
evasion, defense-evasion, bypass, av-bypass, antivirus-bypass,
edr-bypass, edr-evasion, endpoint-detection,
amsi, amsi-bypass, etw, etw-bypass, etw-patching,
unhooking, syscall, direct-syscall, indirect-syscall,
obfuscation, code-obfuscation, string-obfuscation,
packer, crypter, protector, metamorphic, polymorphic,
anti-analysis, anti-debugging, anti-vm, anti-sandbox,
sandbox-evasion, sandbox-detection, vm-detection,
timestomping, log-evasion, event-log-tampering,
ppid-spoofing, argument-spoofing, command-line-spoofing,
opsec, operational-security, covert-channel,
living-off-the-land, lolbas, lolbins, gtfobins,
fileless, fileless-malware, in-memory-execution,
code-signing, signature-evasion, entropy-reduction,

# --- Payload & Shellcode Development ---
payload, payload-generation, payload-framework,
shellcode, shellcode-loader, shellcode-runner, shellcode-encoder,
stager, stageless, staged-payload,
bof, beacon-object-file, coff-loader,
donut, pe-to-shellcode, reflective-dll, reflective-loading,
position-independent-code, pic, egg-hunter,
msfvenom, venom, payload-obfuscation,
dll-sideloading, dll-hijacking, dll-proxying,
binary-patching, pe-manipulation, elf-manipulation,

# --- Credential & Authentication Attacks ---
credential, credential-theft, credential-stuffing,
password, password-cracking, password-spraying, password-attack,
hash, hash-cracking, hashcat, john-the-ripper,
brute-force, bruteforce, dictionary-attack, wordlist,
mimikatz, lsass, sam-dump, secretsdump,
keylogger, keylogging, clipboard-hijacking,
cookie-theft, session-hijacking, jwt-attack,
mfa-bypass, 2fa-bypass, otp-bypass, push-fatigue,
phishing-kit, evilginx, modlishka, gophish,

# --- Network & Protocol Attacks ---
network-attack, network-security, packet-crafting,
man-in-the-middle, mitm, arp-spoofing, dns-spoofing,
responder, llmnr, nbns, mdns, wpad,
smb-attack, smb-relay, rpc-attack, dcom-attack,
wifi-attack, wireless-security, evil-twin, wpa-crack,
bluetooth-attack, ble-attack, zigbee, rfid,
sniffing, packet-capture, traffic-analysis,
vpn-attack, ssl-stripping, tls-attack,
coap, mqtt, modbus, scada, ics-security, ot-security,

# --- Cloud & Container Security ---
cloud-security, cloud-pentest, cloud-exploitation,
aws-security, aws-exploitation, pacu, prowler,
azure-security, azure-exploitation, azurehound, roadtools,
gcp-security, gcp-exploitation,
kubernetes-security, k8s-pentest, container-escape,
docker-escape, docker-security,
serverless-security, lambda-exploitation,
s3-bucket, storage-enumeration, cloud-enumeration,
terraform-abuse, iam-exploitation, iam-escalation,
metadata-service, imds, ssrf-to-rce,

# --- Web Application & API Attacks ---
web-security, web-exploitation, web-pentest,
owasp, owasp-top-10, burp-suite, burp-extension,
api-security, api-pentest, api-fuzzing,
fuzzing, fuzzer, web-fuzzer, protocol-fuzzer,
cors-misconfiguration, csp-bypass, cache-poisoning,
jwt-attack, graphql-attack, websocket-attack,
waf-bypass, firewall-bypass, ips-evasion,
subdomain-takeover, dangling-dns,
cms-exploit, wordpress-exploit, joomla-exploit,

# --- Mobile & IoT ---
mobile-security, mobile-pentest, android-security, ios-security,
apk-analysis, ipa-analysis, frida, objection,
iot-security, iot-pentest, firmware-analysis, firmware-extraction,
embedded-security, hardware-hacking, jtag, uart, spi,

# --- Malware Development & Analysis ---
malware, malware-development, malware-analysis, malware-research,
reverse-engineering, reversing, binary-analysis,
disassembler, decompiler, ida, ghidra, binary-ninja,
unpacking, deobfuscation, config-extraction,
ransomware, worm, trojan, spyware, infostealer,
botnet, bot-framework,

# --- Red Team Operations & Frameworks ---
red-team, redteam, red-teaming, adversary-simulation,
purple-team, breach-simulation, attack-simulation,
penetration-testing, pentest, pentesting, ethical-hacking,
bug-bounty, bug-hunting, vulnerability-assessment,
attack-framework, offensive-framework, offensive-tool,
offensive-security, offensive-development,
ttps, mitre-attack, attack-technique,
threat-emulation, threat-simulation,

# --- Cryptography Attacks ---
cryptanalysis, crypto-attack, padding-oracle, bleichenbacher,
hash-collision, birthday-attack, length-extension,
weak-crypto, deprecated-cipher, rsa-attack, aes-attack,
side-channel, timing-attack, power-analysis,

# --- Windows Internals & Techniques ---
windows-internals, win32-api, nt-api, native-api,
pe-format, coff, com-hijacking, dcom,
wmi, wmi-persistence, wmi-lateral-movement,
scheduled-task, registry-persistence, startup-persistence,
service-creation, driver-exploitation, kernel-driver,
minifilter, callback, ssdt, idt,
etw-provider, tracelogging, windows-event-log,

# --- Linux & Unix Techniques ---
linux-privesc, linux-exploit, linux-persistence,
capabilities-abuse, suid-exploitation, cron-exploitation,
container-breakout, namespace-escape, seccomp-bypass,
ebpf, ebpf-rootkit, kernel-module, lkm-rootkit,
pam-backdoor, ssh-backdoor, ld-preload
```

## Architecture Changes

### New: Offensive Security Scraper Profile

Add an alternative scraper configuration that replaces the general-purpose GitHub scraper with security-focused discovery.

#### File: `phase1_data_collection/scrapers/offsec_scraper.py`

```
OffSecScraper
├── enumerate_org_repos(org: str) -> List[RepoMetadata]
│   └── Uses GitHub API: GET /orgs/{org}/repos or /users/{user}/repos
├── search_by_topics(topics: List[str]) -> List[RepoMetadata]
│   └── Uses GitHub Search API with topic: qualifiers
├── classify_offsec_domain(repo: RepoMetadata) -> str
│   └── Returns: exploit, c2, recon, post_exploit, evasion, credential, payload,
│                active_directory, cloud, network, malware, web_attack, mobile_iot, general
├── score_offsec_quality(repo: RepoMetadata) -> float
│   └── Security-specific scoring (see below)
└── collect_all() -> Generator[Path, None, None]
    └── Orchestrates all discovery methods, deduplicates, clones
```

#### Security-Specific Quality Scoring

Replace the general `QualityScorer` with domain-aware scoring for offensive security repos:

| Component | Weight | Criteria |
|---|---|---|
| **Relevance** | 30 pts | Keyword density in description, topics, README. Bonus for CVE references, MITRE ATT&CK TIDs |
| **Code Quality** | 20 pts | Has requirements/setup, structured project, not just scripts dumped in root |
| **Recency** | 20 pts | Updated within 1 year; actively maintained tools score highest |
| **Community** | 15 pts | Stars, forks (log scale). Widely-used tools produce better training signal |
| **Completeness** | 15 pts | Has README, examples, documentation. Models learn better from well-documented code |

#### Offensive Security Domain Keywords

```python
OFFSEC_KEYWORDS = {
    "exploit": [
        # Core exploit terms
        "exploit", "exploit-development", "exploitdb", "cve-", "vulnerability",
        "vulnerability-research", "zero-day", "0day", "poc", "proof-of-concept",
        # Memory corruption
        "buffer-overflow", "heap-overflow", "heap-spray", "stack-overflow",
        "use-after-free", "type-confusion", "integer-overflow", "format-string",
        "double-free", "out-of-bounds", "race-condition", "rop-chain", "rop-gadget",
        # Execution types
        "rce", "remote-code-execution", "lpe", "local-privilege-escalation",
        "arbitrary-code-execution", "code-execution",
        # Web exploitation
        "sql-injection", "sqli", "xss", "cross-site-scripting", "ssrf",
        "lfi", "rfi", "command-injection", "xxe", "ssti", "idor",
        "insecure-deserialization", "deserialization", "prototype-pollution",
        "path-traversal", "file-inclusion", "template-injection",
        # Platform-specific
        "kernel-exploit", "browser-exploit", "windows-exploit", "linux-exploit",
        "ios-exploit", "android-exploit", "macos-exploit", "chrome-exploit",
        "v8-exploit", "webkit-exploit", "firmware-exploit",
    ],
    "c2": [
        # Core C2 terms
        "c2", "c2-framework", "command-and-control", "cnc",
        "rat", "remote-access-trojan", "remote-access-tool",
        # C2 components
        "beacon", "implant", "agent", "listener", "stager", "stageless",
        "teamserver", "operator", "handler", "callback", "check-in",
        "payload-delivery", "tasking", "job-queue",
        # Named C2 frameworks
        "cobalt-strike", "metasploit", "empire", "sliver", "havoc",
        "mythic", "covenant", "brute-ratel", "nighthawk", "poshc2",
        "merlin-c2", "deimos", "villain", "pupy", "koadic", "silenttrinity",
        "faction", "nuages", "ares", "trevorC2", "shad0w",
        # C2 channels
        "dns-c2", "http-c2", "https-c2", "smb-c2", "icmp-c2",
        "websocket-c2", "dns-tunnel", "dns-exfiltration",
        "covert-channel", "data-exfiltration", "exfiltration",
        "domain-fronting", "cdn-fronting", "cloud-fronting",
        # C2 infrastructure
        "malleable-profile", "redirector", "infrastructure",
        "teamserver-setup", "c2-profile", "traffic-shaping",
    ],
    "recon": [
        # Core recon terms
        "recon", "reconnaissance", "enumeration", "scanner", "discovery",
        "information-gathering", "footprinting", "profiling",
        # OSINT
        "osint", "open-source-intelligence", "social-engineering-recon",
        "email-harvesting", "username-enumeration", "people-search",
        "google-dork", "dorking", "search-engine-hacking",
        # Network recon
        "port-scan", "port-scanner", "service-enumeration",
        "host-discovery", "network-mapping", "network-scan",
        "arp-scan", "ping-sweep", "traceroute",
        "banner-grabbing", "fingerprint", "fingerprinting",
        "os-detection", "version-detection",
        # Domain/web recon
        "subdomain-enumeration", "subdomain-discovery", "subdomain-takeover",
        "dns-enumeration", "dns-recon", "zone-transfer", "dns-brute",
        "web-recon", "directory-bruteforce", "content-discovery",
        "vhost-enumeration", "parameter-discovery", "endpoint-discovery",
        "technology-detection", "wappalyzer", "whatweb",
        # Asset discovery
        "asset-discovery", "attack-surface", "external-recon",
        "cloud-enumeration", "s3-enumeration", "bucket-finder",
        "certificate-transparency", "ct-logs",
        # Search engines
        "shodan", "censys", "zoomeye", "fofa", "binaryedge",
        "spyse", "securitytrails", "virustotal",
    ],
    "post_exploit": [
        # Core post-exploitation
        "post-exploitation", "post-exploit", "post-compromise",
        # Lateral movement
        "lateral-movement", "pivoting", "port-forwarding", "tunneling",
        "socks-proxy", "ssh-tunnel", "chisel", "ligolo",
        "psexec", "wmiexec", "smbexec", "atexec", "dcomexec",
        "winrm", "evil-winrm", "rdp-hijacking",
        # Persistence
        "persistence", "backdoor", "rootkit", "bootkit", "webshell",
        "implant-persistence", "registry-persistence", "startup-persistence",
        "scheduled-task", "service-creation", "wmi-persistence",
        "cron-persistence", "ssh-backdoor", "pam-backdoor",
        "golden-ticket-persistence", "skeleton-key",
        # Privilege escalation
        "privilege-escalation", "privesc", "lpe",
        "local-privilege-escalation", "vertical-escalation",
        "uac-bypass", "potato-attack", "printspoofer",
        "suid-exploitation", "capabilities-abuse",
        "linux-privesc", "windows-privesc",
        "kernel-privesc", "service-exploitation",
        # Data collection
        "loot", "credential-access", "data-collection",
        "screenshot-capture", "keylogger", "clipboard-hijacking",
        "browser-data", "wifi-passwords", "vault-dump",
        # Process manipulation
        "process-injection", "dll-injection", "process-hollowing",
        "thread-hijacking", "apc-injection", "early-bird-injection",
        "token-manipulation", "token-impersonation", "token-theft",
        "named-pipe-impersonation", "handle-duplication",
    ],
    "evasion": [
        # Core evasion terms
        "evasion", "defense-evasion", "bypass", "av-bypass",
        "antivirus-bypass", "edr-bypass", "edr-evasion",
        "endpoint-detection", "detection-bypass",
        # Windows security bypass
        "amsi", "amsi-bypass", "amsi-patch",
        "etw", "etw-bypass", "etw-patching", "etw-blind",
        "wdac-bypass", "applocker-bypass", "clm-bypass",
        "constrained-language-mode", "script-block-logging",
        # Syscall techniques
        "unhooking", "syscall", "direct-syscall", "indirect-syscall",
        "hell-gate", "halo-gate", "tartarus-gate", "syswhispers",
        "ntdll-unhooking", "userland-hooking",
        # Code obfuscation
        "obfuscation", "code-obfuscation", "string-obfuscation",
        "string-encryption", "control-flow-obfuscation",
        "call-stack-spoofing", "return-address-spoofing",
        # Packers and protectors
        "packer", "crypter", "protector", "metamorphic", "polymorphic",
        "pe-packer", "runtime-packer",
        # Anti-analysis
        "anti-analysis", "anti-debugging", "anti-vm", "anti-sandbox",
        "sandbox-evasion", "sandbox-detection", "vm-detection",
        "timing-check", "cpuid-check",
        # Forensic evasion
        "timestomping", "log-evasion", "event-log-tampering",
        "indicator-removal", "artifact-cleanup", "track-covering",
        # Spoofing
        "ppid-spoofing", "argument-spoofing", "command-line-spoofing",
        "module-stomping", "phantom-dll",
        # Living off the land
        "living-off-the-land", "lolbas", "lolbins", "gtfobins",
        "loldriver", "byovd", "bring-your-own-vulnerable-driver",
        # Fileless techniques
        "fileless", "fileless-malware", "in-memory-execution",
        "memory-only", "reflective-execution",
        # Signature evasion
        "code-signing", "signature-evasion", "entropy-reduction",
        "binary-padding", "section-manipulation",
        # Operational security
        "opsec", "operational-security", "covert-channel",
        "traffic-blending", "jitter", "sleep-obfuscation",
    ],
    "credential": [
        # Core credential terms
        "credential", "credential-theft", "credential-stuffing",
        "credential-dumping", "credential-harvesting",
        # Password attacks
        "password", "password-cracking", "password-spraying",
        "password-attack", "password-guessing", "default-password",
        # Hash attacks
        "hash", "hash-cracking", "hashcat", "john-the-ripper",
        "rainbow-table", "hash-extraction",
        # Brute force
        "brute-force", "bruteforce", "dictionary-attack", "wordlist",
        "rule-based-attack", "mask-attack", "combinator-attack",
        # Windows credential attacks
        "mimikatz", "lsass", "lsass-dump", "sam-dump", "secretsdump",
        "dpapi", "windows-vault", "credential-manager",
        "ntds-dit", "ntds-extraction", "volume-shadow-copy",
        # Active Directory credential attacks
        "kerberos", "kerberoasting", "as-rep-roasting",
        "golden-ticket", "silver-ticket", "diamond-ticket",
        "sapphire-ticket", "pass-the-ticket",
        "ntlm", "pass-the-hash", "overpass-the-hash",
        "ntlm-relay", "relay-attack", "responder",
        "dcsync", "dcshadow",
        # Network credential attacks
        "llmnr", "nbns", "mdns", "wpad", "llmnr-poisoning",
        "man-in-the-middle", "mitm", "arp-spoofing",
        # Modern auth attacks
        "mfa-bypass", "2fa-bypass", "otp-bypass", "push-fatigue",
        "token-theft", "cookie-theft", "session-hijacking",
        "jwt-attack", "oauth-abuse", "saml-attack",
        # Phishing for credentials
        "phishing-kit", "evilginx", "evilginx2", "modlishka",
        "gophish", "credential-phishing", "adversary-in-the-middle",
        # Keylogging and capture
        "keylogger", "keylogging", "input-capture",
        "screen-capture", "clipboard-monitor",
        # Spray and dump tools
        "spray", "dump", "extract", "harvest",
    ],
    "payload": [
        # Core payload terms
        "payload", "payload-generation", "payload-framework",
        "payload-obfuscation", "payload-encryption",
        # Shellcode
        "shellcode", "shellcode-loader", "shellcode-runner",
        "shellcode-encoder", "shellcode-generator",
        "position-independent-code", "pic", "egg-hunter",
        # Stagers and loaders
        "stager", "stageless", "staged-payload",
        "dropper", "downloader", "loader", "cradle",
        "powershell-cradle", "download-cradle",
        # BOF and COFF
        "bof", "beacon-object-file", "coff-loader",
        "inline-execute", "object-file",
        # PE manipulation
        "donut", "pe-to-shellcode", "reflective-dll",
        "reflective-loading", "reflective-injection",
        "dll-sideloading", "dll-hijacking", "dll-proxying",
        "binary-patching", "pe-manipulation", "elf-manipulation",
        "section-injection", "code-cave",
        # Payload generators
        "msfvenom", "venom", "sharpshooter", "unicorn",
        "macro-pack", "evil-clippy", "phishery",
        # Initial access payloads
        "hta-payload", "iso-payload", "lnk-payload", "msi-payload",
        "vba-macro", "macro", "office-macro", "xll-payload",
        "html-smuggling", "svg-smuggling",
        # Implant development
        "implant", "implant-framework", "agent-development",
        "c2-agent", "beacon-development",
    ],
    "active_directory": [
        # AD enumeration and attack
        "active-directory", "ad-attack", "ad-exploitation", "ad-enumeration",
        "bloodhound", "sharphound", "ldap-enumeration",
        "group-policy-abuse", "gpo-abuse",
        # Delegation attacks
        "constrained-delegation", "unconstrained-delegation",
        "rbcd", "resource-based-constrained-delegation",
        # ACL attacks
        "acl-abuse", "dacl-abuse", "writedacl", "genericall",
        "genericwrite", "forcechangepassword",
        # Certificate attacks
        "certificate-abuse", "adcs", "certipy", "certify",
        "shadow-credentials", "esc1", "esc2", "esc3", "esc4",
        "esc6", "esc7", "esc8", "esc9", "esc10",
        # Trust attacks
        "domain-trust", "forest-trust", "sid-history",
        "cross-domain", "inter-forest",
        # Azure/Entra
        "azure-ad", "entra-id", "azure-exploitation",
        "azurehound", "roadtools", "azure-enumeration",
    ],
    "cloud": [
        # General cloud security
        "cloud-security", "cloud-pentest", "cloud-exploitation",
        "cloud-enumeration", "multi-cloud",
        # AWS
        "aws-security", "aws-exploitation", "aws-pentest",
        "pacu", "prowler", "scout-suite", "cloudmapper",
        "s3-bucket", "iam-exploitation", "iam-escalation",
        "lambda-exploitation", "ec2-exploitation",
        "metadata-service", "imds", "ssrf-to-rce",
        # Azure
        "azure-security", "azure-exploitation", "azure-pentest",
        "microburst", "stormspotter", "azurite",
        # GCP
        "gcp-security", "gcp-exploitation", "gcp-pentest",
        # Container
        "kubernetes-security", "k8s-pentest", "container-escape",
        "docker-escape", "docker-security", "pod-escape",
        "container-breakout", "namespace-escape",
        # Serverless
        "serverless-security", "lambda-exploitation",
        "function-exploitation",
        # Infrastructure as Code
        "terraform-abuse", "cloudformation-abuse",
        "misconfiguration", "cloud-misconfiguration",
    ],
    "network": [
        # Network attacks
        "network-attack", "network-security", "packet-crafting",
        "man-in-the-middle", "mitm", "arp-spoofing", "dns-spoofing",
        "dns-poisoning", "bgp-hijacking",
        # Protocol attacks
        "smb-attack", "smb-relay", "rpc-attack", "dcom-attack",
        "ldap-attack", "mssql-attack", "mysql-attack",
        "snmp-attack", "ftp-attack", "telnet-attack",
        # Wireless
        "wifi-attack", "wireless-security", "evil-twin", "wpa-crack",
        "wpa2-attack", "wps-attack", "karma-attack", "rogue-ap",
        "bluetooth-attack", "ble-attack", "zigbee", "rfid",
        # Sniffing and interception
        "sniffing", "packet-capture", "traffic-analysis",
        "traffic-interception", "pcap-analysis",
        # Protocol-specific
        "vpn-attack", "ssl-stripping", "tls-attack",
        "coap", "mqtt", "modbus", "dnp3",
        "scada", "ics-security", "ot-security", "plc-attack",
    ],
    "malware": [
        # Malware development
        "malware", "malware-development", "malware-analysis",
        "malware-research", "malware-sample",
        # Reverse engineering
        "reverse-engineering", "reversing", "binary-analysis",
        "static-analysis", "dynamic-analysis",
        "disassembler", "decompiler", "ida", "ghidra", "binary-ninja",
        "x64dbg", "windbg", "gdb",
        # Malware techniques
        "unpacking", "deobfuscation", "config-extraction",
        "string-decryption", "api-hashing", "import-hashing",
        # Malware types
        "ransomware", "worm", "trojan", "spyware", "infostealer",
        "botnet", "bot-framework", "banking-trojan",
        "miner", "cryptominer", "adware",
        # Analysis tools
        "yara", "yara-rules", "sigma-rules", "snort-rules",
        "sandbox-analysis", "automated-analysis",
    ],
    "web_attack": [
        # Web application attacks
        "web-security", "web-exploitation", "web-pentest",
        "owasp", "owasp-top-10", "web-vulnerability",
        # Tools
        "burp-suite", "burp-extension", "zap-extension",
        "sqlmap", "nuclei-template",
        # API attacks
        "api-security", "api-pentest", "api-fuzzing",
        "graphql-attack", "rest-api-attack", "grpc-attack",
        # Fuzzing
        "fuzzing", "fuzzer", "web-fuzzer", "protocol-fuzzer",
        "mutation-fuzzing", "grammar-fuzzing", "afl", "libfuzzer",
        # Specific web attacks
        "cors-misconfiguration", "csp-bypass", "cache-poisoning",
        "request-smuggling", "http-desync", "websocket-attack",
        "race-condition", "time-of-check", "business-logic",
        # WAF bypass
        "waf-bypass", "firewall-bypass", "ips-evasion",
        "filter-bypass", "input-validation-bypass",
        # CMS exploits
        "cms-exploit", "wordpress-exploit", "joomla-exploit",
        "drupal-exploit", "magento-exploit",
    ],
    "mobile_iot": [
        # Mobile security
        "mobile-security", "mobile-pentest",
        "android-security", "android-exploit", "android-malware",
        "ios-security", "ios-exploit", "ios-jailbreak",
        "apk-analysis", "ipa-analysis", "smali",
        "frida", "objection", "drozer",
        # IoT and hardware
        "iot-security", "iot-pentest", "iot-exploit",
        "firmware-analysis", "firmware-extraction", "firmware-emulation",
        "embedded-security", "hardware-hacking",
        "jtag", "uart", "spi", "i2c", "can-bus",
        "zigbee-attack", "zwave-attack", "lora-attack",
    ],
}
```

#### Language Distribution

Offensive security code has a different language profile than general software:

| Language | Weight | Rationale |
|---|---|---|
| **Python** | 30% | Most red team tooling (Impacket, Empire agents, custom scripts) |
| **C/C++** | 25% | Exploits, shellcode, BOFs, native implants |
| **C#** | 15% | .NET offensive tools (GhostPack, SharpCollection) |
| **PowerShell** | 10% | Post-exploitation, AD enumeration, AMSI bypass |
| **Go** | 10% | Modern C2 agents (Sliver), cross-platform tooling |
| **Rust** | 5% | Emerging implant/loader development |
| **Shell/Bash** | 5% | Automation scripts, one-liners, enumeration |

**Note:** PowerShell (`.ps1`) and Shell (`.sh`) are not in the base `FileFilter.EXTENSIONS`. The specialized scraper must add these.

### New: Configuration Profile

#### File: `config/offsec_config.yaml`

```yaml
# Offensive Security Specialized Data Collection
data_collection:
  github_token: ${GITHUB_TOKEN}
  repos_dir: ./data/offsec_repositories
  captures_dir: ./data/offsec_captures
  catalog_db: ./data/offsec_catalog.db
  min_quality_score: 20  # Lower threshold — many security repos are small/niche

  # Seed organizations and users
  seed_orgs:
    - cmndcntrlcyber
    - BC-SECURITY
    - BishopFox
    - projectdiscovery
    - GhostPack
    - S3cur3Th1sSh1t
    - fortra

  seed_users:
    - gentilkiwi
    - swisskyrepo

  # Topic-based discovery — comprehensive offensive security taxonomy
  topics:
    # Exploitation & vulnerability research
    - exploit
    - exploit-development
    - vulnerability
    - vulnerability-research
    - zero-day
    - proof-of-concept
    - buffer-overflow
    - rce
    - deserialization
    - sql-injection
    - xss
    - ssrf
    - command-injection
    - kernel-exploit
    - browser-exploit
    # C2 frameworks
    - c2
    - c2-framework
    - command-and-control
    - rat
    - remote-access-trojan
    - beacon
    - implant
    - cobalt-strike
    - metasploit
    - sliver
    - havoc
    - mythic
    - empire
    - brute-ratel
    # Reconnaissance
    - recon
    - reconnaissance
    - enumeration
    - osint
    - subdomain-enumeration
    - port-scanner
    - asset-discovery
    - attack-surface
    - content-discovery
    # Initial access
    - phishing
    - spearphishing
    - social-engineering
    - payload-delivery
    - html-smuggling
    - macro
    # Post-exploitation
    - post-exploitation
    - lateral-movement
    - pivoting
    - persistence
    - backdoor
    - rootkit
    - webshell
    - privilege-escalation
    - privesc
    - process-injection
    - dll-injection
    - token-manipulation
    # Active Directory
    - active-directory
    - kerberoasting
    - bloodhound
    - ntlm-relay
    - dcsync
    - golden-ticket
    - pass-the-hash
    - adcs
    - certificate-abuse
    - azure-ad
    # Evasion
    - evasion
    - defense-evasion
    - av-bypass
    - edr-bypass
    - amsi-bypass
    - etw-bypass
    - unhooking
    - obfuscation
    - sandbox-evasion
    - living-off-the-land
    - lolbas
    - fileless
    - opsec
    - syscall
    # Credentials
    - credential-dumping
    - password-cracking
    - password-spraying
    - mimikatz
    - hashcat
    - brute-force
    - phishing-kit
    - evilginx
    - mfa-bypass
    # Payloads & shellcode
    - payload
    - shellcode
    - shellcode-loader
    - beacon-object-file
    - bof
    - reflective-dll
    - dll-sideloading
    - donut
    - dropper
    - stager
    # Cloud & container
    - cloud-security
    - cloud-pentest
    - aws-security
    - azure-security
    - gcp-security
    - kubernetes-security 
    - container-escape
    - iam-escalation
    # Network attacks
    - network-attack
    - man-in-the-middle
    - arp-spoofing
    - responder
    - wifi-attack
    - scada
    - ics-security
    # Web application
    - web-security
    - web-exploitation
    - owasp
    - burp-extension
    - api-security
    - fuzzing
    - waf-bypass
    # Malware
    - malware
    - malware-development
    - malware-analysis
    - reverse-engineering
    - ransomware
    - botnet
    - yara
    # Mobile & IoT
    - mobile-security
    - android-security
    - ios-security
    - frida
    - iot-security
    - firmware-analysis
    - hardware-hacking
    # Red team operations
    - red-team
    - redteam
    - red-teaming
    - adversary-simulation
    - purple-team
    - penetration-testing
    - pentest
    - pentesting
    - ethical-hacking
    - bug-bounty
    - offensive-security
    - offensive-tool
    - mitre-attack
    - threat-emulation
    # Crypto attacks
    - cryptanalysis
    - padding-oracle
    - side-channel
    # Windows internals
    - windows-internals
    - pe-format
    - com-hijacking
    - byovd
    # Linux techniques
    - linux-privesc
    - container-breakout
    - ebpf
    - lkm-rootkit

  # Stars-based search queries — cross-language coverage
  search_queries:
    # Python offensive tools
    - "topic:exploit language:python stars:>5"
    - "topic:red-team language:python stars:>5"
    - "topic:pentest language:python stars:>5"
    - "topic:osint language:python stars:>5"
    - "topic:malware-analysis language:python stars:>5"
    - "topic:c2 language:python stars:>5"
    - "topic:recon language:python stars:>5"
    - "topic:phishing language:python stars:>5"
    - "mimikatz OR impacket OR empire OR bloodhound language:python stars:>10"
    - "nuclei OR subfinder OR httpx language:go stars:>10"
    # Go offensive tools
    - "topic:c2 language:go stars:>10"
    - "topic:pentest language:go stars:>5"
    - "topic:red-team language:go stars:>5"
    - "sliver OR chisel OR ligolo language:go stars:>10"
    # C/C++ exploits and shellcode
    - "topic:exploit language:c stars:>5"
    - "topic:shellcode language:c stars:>3"
    - "topic:kernel-exploit language:c stars:>3"
    - "topic:buffer-overflow language:c stars:>3"
    - "shellcode OR payload OR bof language:c stars:>5"
    - "exploit OR poc OR cve language:cpp stars:>5"
    # C# offensive tools
    - "topic:pentest language:csharp stars:>5"
    - "topic:red-team language:csharp stars:>5"
    - "topic:evasion language:csharp stars:>3"
    - "evasion OR bypass OR amsi language:csharp stars:>5"
    - "sharp OR rubeus OR seatbelt language:csharp stars:>5"
    # PowerShell offensive
    - "topic:red-team language:powershell stars:>3"
    - "topic:post-exploitation language:powershell stars:>3"
    - "topic:active-directory language:powershell stars:>3"
    - "bypass OR amsi OR privesc language:powershell stars:>3"
    # Rust offensive
    - "topic:red-team language:rust stars:>3"
    - "topic:malware language:rust stars:>3"
    - "implant OR loader OR evasion language:rust stars:>3"
    # Nim offensive (emerging)
    - "topic:red-team language:nim stars:>2"
    - "topic:evasion language:nim stars:>2"
    # Cross-language technique searches
    - "topic:privilege-escalation stars:>10"
    - "topic:lateral-movement stars:>5"
    - "topic:persistence stars:>5"
    - "topic:credential-dumping stars:>5"
    - "topic:process-injection stars:>5"
    - "topic:dll-injection stars:>5"
    - "topic:token-manipulation stars:>3"
    - "topic:kerberoasting stars:>5"
    - "topic:ntlm-relay stars:>3"
    - "topic:adcs stars:>3"
    - "topic:cloud-pentest stars:>5"
    - "topic:container-escape stars:>3"
    - "topic:firmware-analysis stars:>5"
    - "topic:waf-bypass stars:>5"
    - "topic:api-security stars:>5"

  languages:
    - Python
    - C
    - C++
    - C#
    - Go
    - Ruby
    - Rust
    - PowerShell
    - Shell

  # Extended file extensions for security-relevant languages
  extra_extensions:
    PowerShell: [".ps1", ".psm1", ".psd1"]
    Shell: [".sh", ".bash", ".zsh"]
    Nim: [".nim"]

  capture_workers: 8
  viewport_width: 2560
  viewport_height: 1440
  font_size: 14
  theme: "Default Dark+"
```

### Modified: File Filter Extension

The base `FileFilter` needs extension for PowerShell and Shell scripts:

```python
# Additional extensions for offsec profile
OFFSEC_EXTENSIONS = {
    **FileFilter.EXTENSIONS,
    "PowerShell": [".ps1", ".psm1", ".psd1"],
    "Shell": [".sh", ".bash", ".zsh"],
    "Nim": [".nim"],
    "C": [".c"],  # Separate from C++ for exploit code
}
```

Also relax the line count constraints — many exploit POCs and payload scripts are shorter than the base 20-line minimum:

| Parameter | Base Value | OffSec Value | Rationale |
|---|---|---|---|
| `MIN_LINES` | 20 | 10 | Short exploit POCs, shellcode loaders |
| `MAX_LINES` | 500 | 1000 | Large framework modules (Metasploit) |
| `MIN_FILE_SIZE` | 200 B | 100 B | Compact shellcode generators |
| `MAX_FILE_SIZE` | 50 KB | 100 KB | Comprehensive tool files |

### New: Orchestrator Script

#### File: `phase1_data_collection/scripts/run_offsec_collection.py`

Mirrors `run_collection.py` but loads `offsec_config.yaml` and uses `OffSecScraper` instead of `GitHubScraper`. Same CLI interface:

```bash
python -m phase1_data_collection.scripts.run_offsec_collection \
    --config config/offsec_config.yaml \
    --skip-capture  # First run: just catalog and clone
```

## Dataset Metadata Enrichment

Each captured sample in the offensive security dataset should include additional metadata beyond the base schema:

```json
{
    "file_path": "...",
    "file_hash": "...",
    "language": "python",
    "line_count": 150,
    "offsec_domain": "post_exploit",
    "mitre_tactics": ["TA0006", "TA0008"],
    "keywords_matched": ["lateral-movement", "pass-the-hash", "ntlm"],
    "repo_org": "cmndcntrlcyber",
    "repo_stars": 45,
    "has_cve_reference": false
}
```

This metadata enables filtered fine-tuning — e.g., training only on `c2` domain code, or weighting `exploit` samples higher during loss computation.

## SQLite Schema Extension

Add an `offsec_metadata` table linked to the base `captures` table:

```sql
CREATE TABLE IF NOT EXISTS offsec_metadata (
    id INTEGER PRIMARY KEY,
    capture_id INTEGER REFERENCES captures(id),
    domain TEXT NOT NULL,           -- exploit, c2, recon, post_exploit, evasion, credential, payload, active_directory, cloud, network, malware, web_attack, mobile_iot
    mitre_tactics TEXT,             -- Comma-separated TIDs
    keywords_matched TEXT,          -- JSON array of matched keywords
    has_cve_reference BOOLEAN DEFAULT FALSE,
    cve_ids TEXT,                   -- Comma-separated CVE IDs found in code/comments
    UNIQUE(capture_id)
);

CREATE INDEX IF NOT EXISTS idx_offsec_domain ON offsec_metadata(domain);
```

## Files to Create

| File | Purpose |
|---|---|
| `config/offsec_config.yaml` | Offensive security collection profile |
| `phase1_data_collection/scrapers/offsec_scraper.py` | Security-focused repo discovery and scoring |
| `phase1_data_collection/scrapers/offsec_keywords.py` | Keyword dictionaries and MITRE mapping |
| `phase1_data_collection/scripts/run_offsec_collection.py` | Orchestrator for offsec pipeline |

## Files to Modify

| File | Change |
|---|---|
| `phase1_data_collection/scrapers/file_filter.py` | Add `OffSecFileFilter` subclass with relaxed constraints and extra extensions |
| `phase1_data_collection/scrapers/sqlite_catalog.py` | Add `offsec_metadata` table and `add_offsec_metadata()` method |

## Estimated Yield

| Source | Est. Repos | Est. Files | Est. Samples |
|---|---|---|---|
| cmndcntrlcyber (seed) | ~25 | ~500 | ~500 |
| Curated orgs/users | ~200 | ~15,000 | ~15,000 |
| Topic-based discovery | ~500 | ~25,000 | ~25,000 |
| Stars-based search | ~300 | ~10,000 | ~10,000 |
| **Total** | **~1,025** | **~50,500** | **~50,500** |

## Success Criteria

| Metric | Target |
|---|---|
| Total offensive security samples | >=40,000 |
| Domain coverage | All 13 domains represented |
| Language coverage | Python, C/C++, C#, Go, PowerShell minimum |
| cmndcntrlcyber repos cataloged | 100% of public repos |
| Samples with MITRE tactic tags | >=30% |

## Usage Context

This specialized dataset is designed for **authorized security research and defensive security training**. The fine-tuned model will be used for:
- Red team automation within authorized penetration testing engagements
- Defensive purple team exercises to understand attacker tooling
- Security code review and vulnerability pattern recognition
- CTF competition assistance

The dataset and resulting model are not intended for unauthorized access or malicious use.
