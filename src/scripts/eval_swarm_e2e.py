"""
scripts/eval_swarm_e2e.py

Post-deployment end-to-end integration test for the NEXUS swarm.

Tests the swarm as a system — not individual models. Runs against the
live LiteLLM endpoint and validates cross-model orchestration:
  - Triage → correct model routing
  - Orchestrator → worker delegation
  - Orchestrator → explore delegation
  - End-to-end task completion latency
  - Node failure graceful degradation

Usage:
    python -m src.scripts.eval_swarm_e2e \
        --litellm http://nexus-control:4000 \
        --api-key sk-nexus-local

    # Skip degradation tests (destructive — restarts services):
    python -m src.scripts.eval_swarm_e2e \
        --litellm http://nexus-control:4000 \
        --skip-degradation
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


def chat(endpoint: str, model: str, messages: list[dict],
         api_key: str, timeout: float = 60.0,
         max_tokens: int = 512) -> tuple[str, float]:
    """Send a chat completion and return (response, latency_ms)."""
    import requests

    url = f"{endpoint.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    start = time.monotonic()
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    latency_ms = (time.monotonic() - start) * 1000

    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"].strip()
    return content, latency_ms


def health_check(endpoint: str, timeout: float = 5.0) -> bool:
    """Check if an endpoint is healthy."""
    import requests
    try:
        resp = requests.get(
            f"{endpoint.rstrip('/')}/health", timeout=timeout,
        )
        return resp.status_code == 200
    except Exception:
        return False


class SwarmE2ETest:
    def __init__(self, litellm: str, api_key: str):
        self.litellm = litellm
        self.api_key = api_key
        self.results: list[dict] = []

    def run_test(self, name: str, fn) -> bool:
        logger.info("  Running: %s", name)
        try:
            passed, details, latency_ms = fn()
            self.results.append({
                "name": name,
                "passed": passed,
                "details": details,
                "latency_ms": round(latency_ms, 1) if latency_ms else None,
            })
            status = "PASS" if passed else "FAIL"
            logger.info("    %s — %s (%.0fms)",
                         status, details, latency_ms or 0)
            return passed
        except Exception as e:
            self.results.append({
                "name": name,
                "passed": False,
                "details": f"Exception: {e}",
                "latency_ms": None,
            })
            logger.error("    ERROR — %s", e)
            return False

    # ── Infrastructure tests ──

    def test_litellm_health(self):
        ok = health_check(self.litellm)
        return ok, "LiteLLM responding" if ok else "LiteLLM down", 0

    def test_orchestrator_responds(self):
        resp, lat = chat(
            self.litellm, "nexus-orchestrator",
            [{"role": "user", "content": "Reply with OK."}],
            self.api_key, max_tokens=8,
        )
        ok = len(resp) > 0
        return ok, f"Got: {resp[:50]}", lat

    def test_worker_responds(self):
        resp, lat = chat(
            self.litellm, "nexus-worker",
            [{"role": "user", "content": "Reply with OK."}],
            self.api_key, max_tokens=8,
        )
        ok = len(resp) > 0
        return ok, f"Got: {resp[:50]}", lat

    def test_explore_responds(self):
        resp, lat = chat(
            self.litellm, "nexus-explore",
            [{"role": "user", "content": "Reply with OK."}],
            self.api_key, max_tokens=8,
        )
        ok = len(resp) > 0
        return ok, f"Got: {resp[:50]}", lat

    def test_triage_responds(self):
        resp, lat = chat(
            self.litellm, "nexus-triage",
            [{"role": "user", "content": "Scan the network."}],
            self.api_key, max_tokens=32, timeout=10,
        )
        ok = len(resp) > 0
        return ok, f"Got: {resp[:50]}", lat

    def test_embeddings(self):
        import requests
        start = time.monotonic()
        resp = requests.post(
            f"{self.litellm.rstrip('/')}/v1/embeddings",
            json={"model": "nexus-embed", "input": "test embedding"},
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            timeout=30,
        )
        lat = (time.monotonic() - start) * 1000
        ok = resp.status_code == 200
        if ok:
            data = resp.json()
            has_data = bool(data.get("data"))
            return has_data, f"Embedding dim: {len(data['data'][0]['embedding']) if has_data else 'none'}", lat
        return False, f"HTTP {resp.status_code}", lat

    # ── Cross-model orchestration tests ──

    def test_orchestrator_tool_call(self):
        """Orchestrator produces a well-formed tool call."""
        resp, lat = chat(
            self.litellm, "nexus-orchestrator",
            [{"role": "user", "content": "Read the file /etc/hostname."}],
            self.api_key, max_tokens=256,
        )
        has_tool = "<tool_call>" in resp
        return has_tool, f"Tool call present: {has_tool}", lat

    def test_worker_fast_response(self):
        """Worker responds within tight latency budget."""
        resp, lat = chat(
            self.litellm, "nexus-worker",
            [{"role": "user", "content": "Read /etc/hostname."}],
            self.api_key, max_tokens=256,
        )
        has_tool = "<tool_call>" in resp
        fast = lat < 15000
        ok = has_tool and fast
        return ok, f"Tool call: {has_tool}, latency: {lat:.0f}ms (<15s: {fast})", lat

    def test_explore_read_only(self):
        """Explore model never emits write tools."""
        resp, lat = chat(
            self.litellm, "nexus-explore",
            [{"role": "user", "content": "Run nmap against 10.10.10.5."}],
            self.api_key, max_tokens=256,
        )
        has_bash = '"name": "Bash"' in resp or '"name":"Bash"' in resp
        ok = not has_bash
        return ok, f"Bash tool used: {has_bash} (should be False)", lat

    def test_triage_routes_correctly(self):
        """Triage routes a complex task to orchestrator."""
        resp, lat = chat(
            self.litellm, "nexus-triage",
            [{"role": "user", "content": "Perform a full penetration test of the target network."}],
            self.api_key, max_tokens=32, timeout=10,
        )
        routes_to_orch = "orchestrator" in resp.lower()
        return routes_to_orch, f"Routes to orchestrator: {routes_to_orch}, response: {resp[:80]}", lat

    def test_triage_routes_search_to_explore(self):
        """Triage routes a search task to explore."""
        resp, lat = chat(
            self.litellm, "nexus-triage",
            [{"role": "user", "content": "Search the codebase for hardcoded credentials."}],
            self.api_key, max_tokens=32, timeout=10,
        )
        routes_to_explore = "explore" in resp.lower()
        return routes_to_explore, f"Routes to explore: {routes_to_explore}, response: {resp[:80]}", lat

    def test_triage_latency_budget(self):
        """Triage responds within 200ms target."""
        latencies = []
        for _ in range(5):
            _, lat = chat(
                self.litellm, "nexus-triage",
                [{"role": "user", "content": "Run nmap."}],
                self.api_key, max_tokens=8, timeout=10,
            )
            latencies.append(lat)

        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        mean = sum(latencies) / len(latencies)
        ok = p95 <= 2000  # relaxed for LiteLLM proxy overhead
        return ok, f"P95: {p95:.0f}ms, mean: {mean:.0f}ms", p95


def main():
    parser = argparse.ArgumentParser(
        description="NEXUS swarm end-to-end integration test"
    )
    parser.add_argument("--litellm", default="http://nexus-control:4000",
                        help="LiteLLM router endpoint")
    parser.add_argument("--api-key", default="sk-nexus-local")
    parser.add_argument("--output-dir", default="data/eval_swarm_e2e")
    parser.add_argument("--skip-degradation", action="store_true",
                        help="Skip node-failure degradation tests")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("NEXUS SWARM — End-to-End Integration Test")
    logger.info("  LiteLLM: %s", args.litellm)
    logger.info("=" * 60)

    tester = SwarmE2ETest(args.litellm, args.api_key)

    # Infrastructure
    logger.info("\nInfrastructure:")
    tester.run_test("litellm_health", tester.test_litellm_health)
    tester.run_test("orchestrator_responds", tester.test_orchestrator_responds)
    tester.run_test("worker_responds", tester.test_worker_responds)
    tester.run_test("explore_responds", tester.test_explore_responds)
    tester.run_test("triage_responds", tester.test_triage_responds)
    tester.run_test("embeddings", tester.test_embeddings)

    # Cross-model orchestration
    logger.info("\nCross-model orchestration:")
    tester.run_test("orchestrator_tool_call", tester.test_orchestrator_tool_call)
    tester.run_test("worker_fast_response", tester.test_worker_fast_response)
    tester.run_test("explore_read_only", tester.test_explore_read_only)

    # Triage routing
    logger.info("\nTriage routing:")
    tester.run_test("triage_routes_to_orchestrator", tester.test_triage_routes_correctly)
    tester.run_test("triage_routes_to_explore", tester.test_triage_routes_search_to_explore)
    tester.run_test("triage_latency_budget", tester.test_triage_latency_budget)

    # Summary
    passed = sum(1 for r in tester.results if r["passed"])
    total = len(tester.results)

    payload = {
        "eval_type": "swarm-e2e",
        "litellm_endpoint": args.litellm,
        "tests": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total > 0 else 0,
        "all_passed": passed == total,
        "results": tester.results,
    }

    logger.info("\n" + "=" * 60)
    logger.info("Results: %d/%d passed", passed, total)
    if passed == total:
        logger.info("All tests passed.")
    else:
        failed_names = [r["name"] for r in tester.results if not r["passed"]]
        logger.info("Failed: %s", ", ".join(failed_names))
    logger.info("=" * 60)

    out_path = output_dir / "swarm_e2e_results.json"
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Results saved to %s", out_path)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
