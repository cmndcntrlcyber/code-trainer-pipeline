"""
phase5_gemma_deployment/scripts/benchmark.py

Benchmark inference speed (tokens/sec) for the deployed Gemma GGUF model.

Usage:
    python -m src.phase5_gemma_deployment.scripts.benchmark \
        --gguf models/gguf-gemma/model_q4_k_m.gguf \
        --llama-cpp /path/to/llama.cpp \
        --port 8080
"""
import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.phase5_gemma_deployment.inference.llama_cpp_server import LlamaCppServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BENCHMARK_PROMPTS = [
    "Write a Python function that computes Fibonacci numbers recursively.",
    "Implement a binary search tree in TypeScript with insert and search methods.",
    "Write a Go HTTP handler that returns JSON.",
    "Create a Rust struct for a TCP connection pool with methods.",
]


def main():
    parser = argparse.ArgumentParser(description="Benchmark Gemma GGUF inference speed")
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--llama-cpp", required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--n-runs", type=int, default=10)
    args = parser.parse_args()

    server = LlamaCppServer(
        gguf_path=args.gguf,
        llama_cpp_dir=args.llama_cpp,
        port=args.port,
        ctx_size=args.ctx_size,
    )

    with server:
        import requests
        tokens_per_sec = []

        for i in range(args.n_runs):
            prompt = BENCHMARK_PROMPTS[i % len(BENCHMARK_PROMPTS)]
            url = f"http://localhost:{args.port}/v1/chat/completions"

            start = time.perf_counter()
            resp = requests.post(url, json={
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens,
                "stream": False,
            }, timeout=120)
            elapsed = time.perf_counter() - start

            if resp.ok:
                data = resp.json()
                n_tokens = data.get("usage", {}).get("completion_tokens", args.max_tokens)
                tps = n_tokens / elapsed
                tokens_per_sec.append(tps)
                logger.info(f"Run {i+1}: {tps:.1f} t/s ({n_tokens} tokens in {elapsed:.2f}s)")

        if tokens_per_sec:
            print("\n" + "=" * 50)
            print("BENCHMARK RESULTS (Gemma-4-12B)")
            print("=" * 50)
            print(f"Runs:      {len(tokens_per_sec)}")
            print(f"Mean:      {statistics.mean(tokens_per_sec):.1f} t/s")
            print(f"Median:    {statistics.median(tokens_per_sec):.1f} t/s")
            print(f"Std dev:   {statistics.stdev(tokens_per_sec):.1f} t/s")
            print(f"Min/Max:   {min(tokens_per_sec):.1f} / {max(tokens_per_sec):.1f} t/s")
            print(f"Target:    >50 t/s")
            print("=" * 50)


if __name__ == "__main__":
    main()
