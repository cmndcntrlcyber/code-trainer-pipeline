"""
phase5_triage_deployment/scripts/eval_triage.py

Evaluate triage model routing accuracy and latency.

Runs against a live triage endpoint (RKLLM on OPi 5 Plus or any
OpenAI-compatible server). Tests whether the model correctly routes
requests to nexus-orchestrator, nexus-worker, or nexus-explore.

Usage:
    python -m src.phase5_triage_deployment.scripts.eval_triage \
        --endpoint http://nexus-opi5p:8088 \
        --output-dir data/eval_triage

    # Via LiteLLM:
    python -m src.phase5_triage_deployment.scripts.eval_triage \
        --endpoint http://nexus-control:4000 \
        --model nexus-triage
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ROUTING_SCENARIOS = [
    # → nexus-orchestrator (complex, multi-step)
    {"prompt": "Perform a full penetration test of the 10.10.10.0/24 network.", "expected_route": "nexus-orchestrator", "category": "recon"},
    {"prompt": "Audit the web application for OWASP Top 10 and generate a report.", "expected_route": "nexus-orchestrator", "category": "exploit"},
    {"prompt": "Enumerate all subdomains, scan each for open ports, and identify entry points.", "expected_route": "nexus-orchestrator", "category": "recon"},
    {"prompt": "Investigate the suspicious process on port 4444 and document findings.", "expected_route": "nexus-orchestrator", "category": "exploit"},
    {"prompt": "Create a comprehensive security assessment plan for the target infrastructure.", "expected_route": "nexus-orchestrator", "category": "report"},
    {"prompt": "Coordinate the team to perform red team operations across multiple targets.", "expected_route": "nexus-orchestrator", "category": "admin"},
    {"prompt": "Plan and execute a phased engagement: recon, scanning, exploitation, reporting.", "expected_route": "nexus-orchestrator", "category": "recon"},

    # → nexus-worker (single-tool, execution)
    {"prompt": "Run nmap -sS 10.10.10.5.", "expected_route": "nexus-worker", "category": "recon"},
    {"prompt": "Execute gobuster against https://target.htb.", "expected_route": "nexus-worker", "category": "exploit"},
    {"prompt": "Read /etc/crontab and list suspicious entries.", "expected_route": "nexus-worker", "category": "exploit"},
    {"prompt": "Run linpeas.sh and return the output.", "expected_route": "nexus-worker", "category": "exploit"},
    {"prompt": "Edit config.yaml to change the port from 8080 to 9090.", "expected_route": "nexus-worker", "category": "admin"},
    {"prompt": "Create a Python reverse shell script.", "expected_route": "nexus-worker", "category": "exploit"},
    {"prompt": "Run nuclei with the cves/ template directory against 10.10.10.5.", "expected_route": "nexus-worker", "category": "exploit"},
    {"prompt": "Parse this JSON output and extract the IP addresses.", "expected_route": "nexus-worker", "category": "admin"},

    # → nexus-explore (search, read, classify)
    {"prompt": "Search the codebase for hardcoded API keys.", "expected_route": "nexus-explore", "category": "code-review"},
    {"prompt": "Find all YAML config files in the project.", "expected_route": "nexus-explore", "category": "code-review"},
    {"prompt": "Read the README.md and summarize what the project does.", "expected_route": "nexus-explore", "category": "code-review"},
    {"prompt": "List all files in /var/log.", "expected_route": "nexus-explore", "category": "recon"},
    {"prompt": "Fetch the robots.txt from the target website.", "expected_route": "nexus-explore", "category": "recon"},
    {"prompt": "Check if 10.10.10.5 is within our engagement scope.", "expected_route": "nexus-explore", "category": "admin"},
    {"prompt": "Search for TODO comments across the codebase.", "expected_route": "nexus-explore", "category": "code-review"},

    # Ambiguous (should default to orchestrator)
    {"prompt": "Help me with this target.", "expected_route": "nexus-orchestrator", "category": "unknown"},
    {"prompt": "What should we do next?", "expected_route": "nexus-orchestrator", "category": "unknown"},
    {"prompt": "I'm stuck.", "expected_route": "nexus-orchestrator", "category": "unknown"},
]


def call_triage(endpoint: str, prompt: str, model: str, api_key: str, timeout: float) -> tuple[str, float]:
    """Call the triage endpoint and return (response_text, latency_ms)."""
    import requests

    url = f"{endpoint.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
        "temperature": 0.0,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    start = time.monotonic()
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    latency_ms = (time.monotonic() - start) * 1000

    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"].strip()
    return content, latency_ms


def parse_triage_response(response: str) -> dict:
    """Extract route and category from triage model response."""
    lower = response.lower().strip()

    # Try JSON parse first
    try:
        parsed = json.loads(response)
        return {
            "route": parsed.get("route", ""),
            "category": parsed.get("category", ""),
            "confidence": parsed.get("confidence", 0.0),
        }
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: check for route keywords in plain text
    route = "nexus-orchestrator"
    if "worker" in lower:
        route = "nexus-worker"
    elif "explore" in lower:
        route = "nexus-explore"
    elif "orchestrator" in lower:
        route = "nexus-orchestrator"

    return {"route": route, "category": "", "confidence": 0.0}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate triage model routing accuracy and latency"
    )
    parser.add_argument("--endpoint", default="http://nexus-opi5p:8088",
                        help="Triage server endpoint")
    parser.add_argument("--model", default="nexus-triage",
                        help="Model name for the API call")
    parser.add_argument("--api-key", default="sk-nexus-local")
    parser.add_argument("--output-dir", default="data/eval_triage")
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="Request timeout in seconds")
    parser.add_argument("--latency-target-ms", type=float, default=200.0,
                        help="P95 latency target in milliseconds")
    parser.add_argument("--accuracy-target", type=float, default=0.85,
                        help="Routing accuracy target")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("TRIAGE EVAL — Routing Accuracy + Latency")
    logger.info("  endpoint:       %s", args.endpoint)
    logger.info("  model:          %s", args.model)
    logger.info("  scenarios:      %d", len(ROUTING_SCENARIOS))
    logger.info("  accuracy target: %.0f%%", args.accuracy_target * 100)
    logger.info("  latency target:  %.0fms (P95)", args.latency_target_ms)
    logger.info("=" * 60)

    results = []
    latencies = []

    for i, scenario in enumerate(ROUTING_SCENARIOS):
        logger.info("  [%d/%d] %s", i + 1, len(ROUTING_SCENARIOS),
                     scenario["prompt"][:60])
        try:
            response, latency_ms = call_triage(
                args.endpoint, scenario["prompt"], args.model,
                args.api_key, args.timeout,
            )
            parsed = parse_triage_response(response)
            correct = parsed["route"] == scenario["expected_route"]
            latencies.append(latency_ms)

            results.append({
                "prompt": scenario["prompt"],
                "expected_route": scenario["expected_route"],
                "actual_route": parsed["route"],
                "correct": correct,
                "latency_ms": round(latency_ms, 1),
                "raw_response": response[:200],
                "category": scenario["category"],
            })
            logger.info("    %s → %s (%.0fms) %s",
                         parsed["route"], scenario["expected_route"],
                         latency_ms, "OK" if correct else "WRONG")

        except Exception as e:
            results.append({
                "prompt": scenario["prompt"],
                "expected_route": scenario["expected_route"],
                "actual_route": None,
                "correct": False,
                "error": str(e),
                "category": scenario["category"],
            })
            logger.warning("    ERROR: %s", e)

    correct_count = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = correct_count / total if total > 0 else 0.0

    latency_stats = {}
    if latencies:
        latencies_sorted = sorted(latencies)
        p50_idx = len(latencies_sorted) // 2
        p95_idx = int(len(latencies_sorted) * 0.95)
        latency_stats = {
            "mean_ms": round(sum(latencies) / len(latencies), 1),
            "p50_ms": round(latencies_sorted[p50_idx], 1),
            "p95_ms": round(latencies_sorted[min(p95_idx, len(latencies_sorted) - 1)], 1),
            "min_ms": round(min(latencies), 1),
            "max_ms": round(max(latencies), 1),
        }

    p95_latency = latency_stats.get("p95_ms", float("inf"))

    payload = {
        "eval_type": "triage-routing",
        "endpoint": args.endpoint,
        "model": args.model,
        "scenarios": total,
        "correct": correct_count,
        "accuracy": accuracy,
        "accuracy_target": args.accuracy_target,
        "meets_accuracy_target": accuracy >= args.accuracy_target,
        "latency": latency_stats,
        "latency_target_ms": args.latency_target_ms,
        "meets_latency_target": p95_latency <= args.latency_target_ms,
        "meets_all_targets": (
            accuracy >= args.accuracy_target
            and p95_latency <= args.latency_target_ms
        ),
        "per_route_accuracy": {},
        "results": results,
    }

    for route in ("nexus-orchestrator", "nexus-worker", "nexus-explore"):
        route_results = [r for r in results if r["expected_route"] == route]
        if route_results:
            route_correct = sum(1 for r in route_results if r["correct"])
            payload["per_route_accuracy"][route] = {
                "correct": route_correct,
                "total": len(route_results),
                "accuracy": round(route_correct / len(route_results), 3),
            }

    logger.info("=" * 60)
    logger.info("Routing accuracy: %d/%d (%.0f%%) — %s",
                correct_count, total, accuracy * 100,
                "PASS" if payload["meets_accuracy_target"] else "FAIL")
    if latency_stats:
        logger.info("Latency P95: %.0fms — %s",
                     p95_latency,
                     "PASS" if payload["meets_latency_target"] else "FAIL")
    for route, stats in payload["per_route_accuracy"].items():
        logger.info("  %s: %d/%d (%.0f%%)",
                     route, stats["correct"], stats["total"],
                     stats["accuracy"] * 100)
    logger.info("=" * 60)

    out_path = output_dir / "triage_eval_results.json"
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
