"""
phase5_deployment/inference/llama_cpp_server.py

Wrapper to launch and interact with the llama.cpp HTTP server.

Usage:
    server = LlamaCppServer(
        gguf_path="models/model_q4_k_m.gguf",
        llama_cpp_dir="/path/to/llama.cpp",
        port=8080,
    )
    server.start()
    response = server.generate("What code is shown in this screenshot?")
    server.stop()
"""
import logging
import subprocess
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


class LlamaCppServer:
    """Manages a llama.cpp server process for GGUF inference."""

    def __init__(
        self,
        gguf_path: str | Path,
        llama_cpp_dir: str | Path,
        port: int = 8080,
        n_gpu_layers: int = -1,     # -1 = all layers on GPU
        ctx_size: int = 8192,
        n_parallel: int = 1,
    ):
        self.gguf_path = Path(gguf_path)
        self.server_bin = Path(llama_cpp_dir) / "llama-server"
        self.port = port
        self.n_gpu_layers = n_gpu_layers
        self.ctx_size = ctx_size
        self.n_parallel = n_parallel
        self._process: subprocess.Popen | None = None

        if not self.server_bin.exists():
            raise FileNotFoundError(f"llama-server not found: {self.server_bin}")

    def start(self, wait_seconds: int = 15):
        """Start the llama.cpp server process."""
        cmd = [
            str(self.server_bin),
            "--model", str(self.gguf_path),
            "--port", str(self.port),
            "--n-gpu-layers", str(self.n_gpu_layers),
            "--ctx-size", str(self.ctx_size),
            "--parallel", str(self.n_parallel),
            "--chat-template", "chatml",
        ]
        logger.info(f"Starting llama.cpp server on port {self.port}...")
        self._process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for server to be ready
        base_url = f"http://localhost:{self.port}"
        for _ in range(wait_seconds):
            try:
                r = requests.get(f"{base_url}/health", timeout=1)
                if r.ok:
                    logger.info(f"Server ready: {base_url}")
                    return
            except requests.ConnectionError:
                pass
            time.sleep(1)
        raise RuntimeError(f"Server failed to start within {wait_seconds}s")

    def stop(self):
        """Stop the server process."""
        if self._process:
            self._process.terminate()
            self._process.wait(timeout=10)
            self._process = None
            logger.info("Server stopped")

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.1) -> str:
        """Generate completion via the server's chat endpoint."""
        url = f"http://localhost:{self.port}/v1/chat/completions"
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a code transcription assistant. When given a screenshot "
                        "of code in a VS Code editor, you reproduce the exact source code shown."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
