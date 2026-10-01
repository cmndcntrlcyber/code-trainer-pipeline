# NEXUS DISTRIBUTED SWARM — Scaffold v3.a (without P40)

> **Purpose:** Five-node swarm using only hardware currently in hand.
> No P40, no GEEKOM AX8 Max, no Razer Core X V2.
> Three NVIDIA GPUs, two RK3588 NPUs.
> The control plane colocates with a 3070 worker on daedelu5.

---

## 0 — Fleet Inventory

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      NEXUS DISTRIBUTED SWARM v3.a                          │
│                         (P40 excluded)                                     │
│                                                                            │
│  ┌─ ACEMAGIC K1 ───────────────────┐   ┌─ LAPTOP-3060 ───────────────────┐ │
│  │  GPU: RTX 5060 Ti (16 GB)       │   │  GPU: RTX 3060 Laptop (6 GB)    │ │
│  │  Host: Ryzen 7 7730U · 32 GB    │   │  Role: EDGE EXPLORE             │ │
│  │  Role: ORCHESTRATOR             │   │  Driver: 580.173.02             │ │
│  │  Driver: 580.173.02 / CUDA 13   │   │                                 │ │
│  │                                 │   │  Model:                         │ │
│  │  Model:                         │   │  • Gemma-4-E4B Q8 (classify)    │ │
│  │  • Qwen2.5-Coder-14B Q4_K_M    │   │  Port: :8086                    │ │
│  │    ORCHESTRATOR     :8080       │   │                                 │ │
│  └──────────────┬──────────────────┘   └──────────┬────────────────────── │ │
│                 │                                 │                        │
│  ┌─ OPi 5 Plus (32GB) ────────────┐   ┌─ OPi 5B (16GB) ─────────────────┐ │
│  │  SoC: RK3588 · NPU 6 TOPS      │   │  SoC: RK3588S · NPU 6 TOPS      │ │
│  │  Role: TRIAGE GATEWAY           │   │  Role: EDGE EMBEDDINGS           │ │
│  │  Dual 2.5GbE, M.2 NVMe         │   │  WiFi 6, BT 5.0                 │ │
│  │                                 │   │  128GB eMMC                      │ │
│  │  Models (RKLLM, NPU):          │   │                                  │ │
│  │  • Qwen3.5-0.8B w8a8           │   │  Models (llama.cpp ARM):         │ │
│  │    (request classification)     │   │  • nomic-embed Q8                │ │
│  │  Services:                      │   │  Port: :8087                     │ │
│  │  • Triage API       :8088       │   │                                  │ │
│  │  • Prometheus        :9090      │   │                                  │ │
│  │  • Grafana           :3000      │   │                                  │ │
│  └──────────────┬──────────────────┘   └──────────┬───────────────────────┘ │
│                 │                                 │                         │
│             ┌───▼─────────────────────────────────▼────┐                    │
│             │   daedelu5 (nexus-control)                │                    │
│             │   GPU: RTX 3070 Laptop (8 GB)            │                    │
│             │   Role: CONTROL PLANE + WORKER           │                    │
│             │                                          │                    │
│             │   Services:                              │                    │
│             │   • rust-nexus infra (A2A + ferry)       │                    │
│             │   • LiteLLM router          :4000        │                    │
│             │   • nexus-harness agent loop              │                    │
│             │                                          │                    │
│             │   Model:                                 │                    │
│             │   • Qwen3-8B Q4_K_M                      │                    │
│             │     WORKER              :8084             │                    │
│             └──────────────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key design: K1 as dedicated orchestrator

The ACEMAGIC K1 hosts the RTX 5060 Ti (16 GB), giving the orchestrator its own
dedicated GPU. The Qwen2.5-Coder-14B at Q4_K_M (~8.5 GB) leaves ~6 GB for
KV cache — enough for `--ctx-size 12288` with q8_0 quantization.

Advantages of the dedicated K1 orchestrator:
- Dense architecture means every parameter fires every token (no MoE routing noise at low quant)
- The Qwen line is more reliable for tool-calling in llama.cpp (no Gemma 4 format issues)
- Your RTPI pipeline already produces a Qwen V9 fine-tune with the Nexus persona
- The control plane (daedelu5) keeps its 3070 free for worker duty

The control plane on daedelu5 colocates with a Qwen3-8B worker on the 3070.
This is safe because the worker uses ~5.5 GB of the 3070's 8 GB, and the
control plane services (rust-nexus, LiteLLM, nexus-harness) are CPU-bound.

### Compute budget

| Node | Accelerator | Memory | Weights | KV headroom | Power |
|------|-------------|--------|---------|-------------|-------|
| K1 (nexus-k1) | RTX 5060 Ti | 16 GB VRAM | ~8.5 GB (Qwen2.5-14B Q4_K_M) | ~6 GB | ~180W |
| daedelu5 (nexus-control) | RTX 3070 Laptop | 8 GB VRAM | ~5.5 GB (Qwen3-8B Q4_K_M) | ~1.8 GB | ~80W |
| Laptop-3060 | RTX 3060 Laptop | 6 GB VRAM | ~3 GB (E4B Q8) | ~1.5 GB | ~80W |
| OPi 5 Plus | RK3588 NPU (RKLLM) | 32 GB sys | ~1.5 GB (Qwen3.5-0.8B w8a8) | ~30 GB | ~15W |
| OPi 5B | RK3588S CPU (ARM) | 16 GB sys | ~0.5 GB (embed) | ~15 GB | ~12W |

### Layout

```
K1          (16 GB):  Qwen2.5-14B Q4_K_M  ORCHESTRATOR     :8080
daedelu5    (8 GB):   Qwen3-8B Q4_K_M     WORKER           :8084
                      + CONTROL PLANE      :4000 :9100 :50051 :50052
3060        (6 GB):   Gemma-4-E4B Q8       EXPLORE/EDGE     :8086
OPi 5 Plus:           Qwen3.5-0.8B NPU    TRIAGE           :8088
OPi 5B:               nomic-embed ARM      EMBEDDINGS       :8087
```

### Hostnames

| Machine | Hostname | LAN IP | Purpose |
|---------|----------|--------|---------|
| daedelu5 | `nexus-control` | 192.168.8.248 | Control plane + worker |
| ACEMAGIC K1 | `nexus-k1` | 192.168.8.238 | Orchestrator (5060 Ti) |
| 3060 laptop | `nexus-3060` | 192.168.8.161 | Edge explore |
| OPi 5 Plus | `nexus-opi5p` | 192.168.8.224 | Triage + monitoring |
| OPi 5B | `nexus-opi5b` | 192.168.8.221 | Embeddings |

---

## 0.5 — Deployment Automation (SSH CA + Role-Based Deploy)

Two separate certificate authorities serve different planes:

- **SSH CA** — management plane: nexus-control authenticates to remote nodes for staging
- **rust-nexus mTLS CA** — data plane: inter-node service communication at runtime (§1)

### 0.5.1 SSH Certificate Authority

Generate the SSH CA on nexus-control (daedelu5). This CA signs both host
certificates (so the control plane trusts remote nodes without TOFU) and
user certificates (so remote nodes trust SSH from the control plane without
password auth or scattered `authorized_keys`).

```bash
# On nexus-control (daedelu5):

# Generate CA key pair
ssh-keygen -t ed25519 -f ~/.ssh/nexus-ca -C "nexus-swarm-ca" -N ""

# Sign the deploy user's public key
ssh-keygen -s ~/.ssh/nexus-ca \
  -I "deploy@nexus-control" \
  -n deploy \
  -V +52w \
  ~/.ssh/id_ed25519.pub
# Produces ~/.ssh/id_ed25519-cert.pub (valid 1 year)
```

### 0.5.2 Bootstrap Remote Nodes

For each remote node, sign its host key and configure it to trust the user CA.
This initial bootstrap requires one-time password/key auth per node.

```bash
#!/usr/bin/env bash
# File: ~/nexus-swarm/bootstrap-node.sh (run from nexus-control)
# Usage: ./bootstrap-node.sh <hostname> <ip>

set -euo pipefail
NODE="$1"
IP="$2"
CA_PUB="$HOME/.ssh/nexus-ca.pub"

echo "=== Bootstrapping $NODE ($IP) ==="

# Fetch the node's host public key
scp "${NODE}:/etc/ssh/ssh_host_ed25519_key.pub" "/tmp/${NODE}-host.pub"

# Sign the host key with the CA
ssh-keygen -s ~/.ssh/nexus-ca \
  -I "host-${NODE}" \
  -h \
  -n "${NODE},${IP}" \
  -V +52w \
  "/tmp/${NODE}-host.pub"

# Push the signed host cert and CA public key back
scp "/tmp/${NODE}-host-cert.pub" "${NODE}:/etc/ssh/ssh_host_ed25519_key-cert.pub"
scp "$CA_PUB" "${NODE}:/etc/ssh/nexus-ca.pub"

# Configure sshd to present the host cert and trust the user CA
ssh "$NODE" bash -s <<'REMOTE'
cat > /etc/ssh/sshd_config.d/nexus.conf <<'SSHD'
HostCertificate /etc/ssh/ssh_host_ed25519_key-cert.pub
TrustedUserCAKeys /etc/ssh/nexus-ca.pub
AuthorizedPrincipalsFile none
SSHD
systemctl reload sshd
REMOTE

echo "  $NODE bootstrapped."
```

```bash
# Bootstrap all remote nodes:
./bootstrap-node.sh nexus-k1 192.168.8.238
./bootstrap-node.sh nexus-3060 192.168.8.161
./bootstrap-node.sh nexus-opi5p 192.168.8.224
./bootstrap-node.sh nexus-opi5b 192.168.8.221

# Configure nexus-control to trust all CA-signed hosts:
echo "@cert-authority * $(cat ~/.ssh/nexus-ca.pub)" >> ~/.ssh/known_hosts

# Verify (no password, no TOFU):
for node in nexus-k1 nexus-3060 nexus-opi5p nexus-opi5b; do
  echo -n "  $node: "
  ssh -o BatchMode=yes "$node" hostname && echo "OK" || echo "FAIL"
done
```

### 0.5.3 Role Manifest

```bash
# File: ~/nexus-swarm/roles.conf (on nexus-control)

# Format: HOSTNAME ROLE CUDA_ARCH LLAMA_PORT
# Lines starting with # are comments. 'local' means no SSH needed.

nexus-control  control+worker  sm_86   8084
nexus-k1       orchestrator    sm_120  8080
nexus-3060     explore         sm_86   8086
nexus-opi5p    triage          npu     8088
nexus-opi5b    embed           arm     8087
```

### 0.5.4 Deployment Script

```bash
#!/usr/bin/env bash
# File: ~/nexus-swarm/deploy-swarm.sh (on nexus-control)
# Usage: ./deploy-swarm.sh [node...]  (no args = all nodes)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROLES_FILE="$SCRIPT_DIR/roles.conf"
HOSTS_FILE="$SCRIPT_DIR/hosts.conf"

# Parse role manifest
declare -A NODE_ROLE NODE_ARCH NODE_PORT
while read -r host role arch port; do
  [[ "$host" =~ ^#.*$ || -z "$host" ]] && continue
  NODE_ROLE["$host"]="$role"
  NODE_ARCH["$host"]="$arch"
  NODE_PORT["$host"]="$port"
done < "$ROLES_FILE"

TARGETS=("${@:-${!NODE_ROLE[@]}}")

# --- Common tasks (all remote nodes) ---
deploy_common() {
  local node="$1"
  [[ "${NODE_ROLE[$node]}" == control+worker ]] && return  # local node, skip SSH

  echo "=== [$node] Common setup ==="
  scp "$HOSTS_FILE" "${node}:/tmp/nexus-hosts"
  ssh "$node" bash -s <<'REMOTE'
    sudo hostnamectl set-hostname "$(hostname -s)"
    sudo cp /tmp/nexus-hosts /etc/hosts
REMOTE

  # Deploy rust-nexus agent binary + mTLS certs
  scp ~/rust-nexus/target/release/nexus-agent "${node}:~/rust-nexus/target/release/"
  scp ~/.nexus/certs/${node}.pem "${node}:~/.nexus/certs/"
  scp ~/.nexus/certs/${node}-key.pem "${node}:~/.nexus/certs/"
  scp ~/.nexus/certs/ca.pem "${node}:~/.nexus/certs/"

  # Deploy nexus-agent launch script + systemd unit
  scp "$SCRIPT_DIR/start-nexus-agent.sh" "${node}:~/nexus-swarm/"
  scp "$SCRIPT_DIR/systemd/nexus-agent.service" "${node}:/etc/systemd/system/"
}

# --- Role-specific tasks ---
deploy_role() {
  local node="$1"
  local role="${NODE_ROLE[$node]}"
  local arch="${NODE_ARCH[$node]}"
  local port="${NODE_PORT[$node]}"

  echo "=== [$node] Role: $role ==="

  case "$role" in
    orchestrator|worker|worker+explore)
      # Build llama.cpp with correct CUDA arch (if not already built)
      ssh "$node" bash -s "$arch" <<'REMOTE'
        ARCH="$1"
        cd ~/llama.cpp
        cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${ARCH#sm_}" \
              -DCMAKE_BUILD_TYPE=Release
        cmake --build build --config Release -j$(nproc)
REMOTE
      # Deploy model + launch script
      scp "$SCRIPT_DIR/start-${role}.sh" "${node}:~/nexus-swarm/"
      ;;

    explore)
      ssh "$node" bash -s "$arch" <<'REMOTE'
        ARCH="$1"
        cd ~/llama.cpp
        cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${ARCH#sm_}" \
              -DCMAKE_BUILD_TYPE=Release
        cmake --build build --config Release -j$(nproc)
REMOTE
      scp "$SCRIPT_DIR/start-explore.sh" "${node}:~/nexus-swarm/"
      ;;

    triage)
      scp "$SCRIPT_DIR/start-triage.sh" "${node}:~/nexus-swarm/"
      ;;

    embed)
      scp "$SCRIPT_DIR/start-embed.sh" "${node}:~/nexus-swarm/"
      ;;

    control+worker)
      echo "  Local node — run setup-control.sh manually."
      ;;
  esac
}

# --- Execute ---
for node in "${TARGETS[@]}"; do
  deploy_common "$node"
  deploy_role "$node"
done

echo "=== Deploy complete ==="
```

---

## 1 — Network Mesh (rust-nexus on LAN)

All 5 nodes on the same subnet. rust-nexus provides mTLS, A2A agent
registration, and ferry-routed tool execution.

### 1.1 Hostnames

```bash
# File: ~/nexus-swarm/hosts.conf (distributed by deploy-swarm.sh)
# Also applied locally on nexus-control:

cat << 'EOF' | sudo tee /etc/hosts
127.0.0.1     localhost
192.168.8.248  nexus-control
192.168.8.238 nexus-k1
192.168.8.161  nexus-3060
192.168.8.224  nexus-opi5p
192.168.8.221  nexus-opi5b
EOF
```

### 1.2 Build and deploy rust-nexus

```bash
# On nexus-control (daedelu5):
git clone https://github.com/cmndcntrlcyber/rust-nexus.git ~/rust-nexus
cd ~/rust-nexus
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
cargo build --workspace --release

# Generate mTLS certs for all 5 nodes:
./scripts/generate-certs.sh \
  --nodes nexus-control,nexus-k1,nexus-3060,nexus-opi5p,nexus-opi5b

# Distribute certs via SSH CA (no password prompts):
for node in nexus-k1 nexus-3060 nexus-opi5p nexus-opi5b; do
  ./scripts/transfer-prep.sh --ip $(getent hosts $node | awk '{print $1}') --user $(whoami)
done
```

### 1.3 Launch nexus-infra on nexus-control

```bash
#!/usr/bin/env bash
# File: ~/nexus-swarm/start-nexus-infra.sh (on nexus-control / daedelu5)

exec ~/rust-nexus/target/release/nexus-infra \
  --a2a-bind 0.0.0.0:50052 \
  --legacy-bind 0.0.0.0:50051 \
  --ferry-bind 0.0.0.0:9100 \
  --cert ~/.nexus/certs/nexus-control.pem \
  --key ~/.nexus/certs/nexus-control-key.pem \
  --ca ~/.nexus/certs/ca.pem \
  --enable-mdns \
  --audit-log /var/log/nexus-audit.jsonl \
  2>&1 | tee /tmp/nexus-infra.log
```

### 1.4 Launch nexus-agent on each compute node

```bash
#!/usr/bin/env bash
# File: ~/nexus-swarm/start-nexus-agent.sh (on each node)

NODE_ID="${NEXUS_NODE_ID:-nexus-control}"
INFRA_ADDR="nexus-control:50052"
LLAMA_PORT="${NEXUS_LLAMA_PORT:-8084}"

exec ~/rust-nexus/target/release/nexus-agent \
  --node-id "$NODE_ID" \
  --infra "$INFRA_ADDR" \
  --cert ~/.nexus/certs/${NODE_ID}.pem \
  --key ~/.nexus/certs/${NODE_ID}-key.pem \
  --ca ~/.nexus/certs/ca.pem \
  --advertise-port "$LLAMA_PORT" \
  --enable-mdns \
  --mode agent \
  2>&1 | tee /tmp/nexus-agent.log
```

| Node | NEXUS_NODE_ID | NEXUS_LLAMA_PORT | Capabilities |
|------|---------------|------------------|--------------|
| daedelu5 | `nexus-control` | `8084` | worker, control |
| K1 | `nexus-k1` | `8080` | orchestrator |
| 3060 laptop | `nexus-3060` | `8086` | explore |
| OPi 5 Plus | `nexus-opi5p` | `8088` | triage |
| OPi 5B | `nexus-opi5b` | `8087` | embed |

### 1.5 Verify

```bash
# From nexus-control (daedelu5):
grpcurl -cacert ~/.nexus/certs/ca.pem \
  -cert ~/.nexus/certs/nexus-control.pem \
  -key ~/.nexus/certs/nexus-control-key.pem \
  nexus-control:50052 nexus.a2a.RegistryLister/ListAgents
# Expected: 5 registered agents

curl -s http://nexus-control:9100/health
# Expected: {"status":"ok","agents":5}

for ep in nexus-k1:8080 nexus-control:8084 \
          nexus-3060:8086 nexus-opi5p:8088 nexus-opi5b:8087; do
  echo -n "  $ep: "
  curl -sf "http://$ep/health" > /dev/null 2>&1 && echo "UP" || echo "DOWN"
done
```

---

## 2 — Inference Servers

### 2.1 Build llama.cpp per architecture

```bash
# K1 (5060 Ti, sm_120):
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="120" -DCMAKE_BUILD_TYPE=Release

# nexus-control / daedelu5 (3070, sm_86) and LAPTOP-3060 (3060, sm_86):
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86" -DCMAKE_BUILD_TYPE=Release
```

### 2.2 K1 — Dedicated Orchestrator (:8080)

The 5060 Ti on K1 runs ONLY the orchestrator. Full 16 GB for one model =
maximum KV headroom for deep agent sessions.

```bash
#!/usr/bin/env bash
# File: ~/nexus-swarm/start-orchestrator.sh (on K1 / nexus-k1)

LLAMA=~/llama.cpp/build/bin/llama-server

# Use the RTPI Qwen V9 fine-tune if available, otherwise base Qwen2.5-14B
MODEL=~/models/Qwen2.5-Coder-14B-Q4_K_M.gguf

exec $LLAMA \
  --model "$MODEL" \
  --host 0.0.0.0 \
  --port 8080 \
  --n-gpu-layers 99 \
  --split-mode none \
  --ctx-size 12288 \
  --n-predict 4096 \
  --batch-size 1024 \
  --ubatch-size 512 \
  --flash-attn \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --slots \
  --slot-save-path /tmp/llama-cache-orch \
  --cont-batching \
  --metrics \
  2>&1 | tee /tmp/orchestrator.log
```

Notes:
- `--ctx-size 12288`: 16 GB VRAM - ~8.5 GB model = ~6 GB for KV, enough for
  12K context with q8_0 KV quantization.
- `--flash-attn`: sm_120 (Blackwell) supports flash-attn natively.
- No parallel workers on this GPU — the orchestrator gets all the VRAM.
- `--slot-save-path`: Prefix caching keeps the Nexus system prompt cached
  across turns.

### 2.3 nexus-control (daedelu5) — Worker (:8084)

```bash
#!/usr/bin/env bash
# File: ~/nexus-swarm/start-worker.sh (on nexus-control / daedelu5)

LLAMA=~/llama.cpp/build/bin/llama-server

exec $LLAMA \
  --model ~/models/Qwen3-8B-Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8084 \
  --n-gpu-layers 99 \
  --split-mode none \
  --ctx-size 4096 \
  --n-predict 2048 \
  --batch-size 512 \
  --ubatch-size 256 \
  --flash-attn \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --parallel 2 \
  --cont-batching \
  --metrics \
  2>&1 | tee /tmp/llama-worker.log
```

### 2.4 LAPTOP-3060 — Edge Explore (:8086)

```bash
#!/usr/bin/env bash
# File: ~/nexus-swarm/start-explore.sh (on laptop-3060)

LLAMA=~/llama.cpp/build/bin/llama-server

exec $LLAMA \
  --model ~/models/gemma-4-E4B-it-Q8_0.gguf \
  --host 0.0.0.0 \
  --port 8086 \
  --n-gpu-layers 99 \
  --ctx-size 4096 \
  --n-predict 1024 \
  --batch-size 512 \
  --flash-attn \
  --parallel 2 \
  --cont-batching \
  --metrics \
  2>&1 | tee /tmp/llama-explore.log
```

### 2.5 OPi 5 Plus — Triage Gateway (:8088)

The OPi 5 Plus runs RKLLM on its RK3588 NPU (6 TOPS) for ultra-low-latency
request classification. It also hosts Prometheus and Grafana for fleet monitoring.

**RKLLM runtime install:**

```bash
# On OPi 5 Plus (nexus-opi5p):

# Install RKLLM runtime (Rockchip NPU inference)
sudo apt update && sudo apt install -y python3-pip librknnrt-dev
pip3 install rkllm-toolkit

# Verify NPU is accessible
cat /sys/class/misc/npu/device/power/runtime_status
# Expected: active
```

**Model conversion to RKLLM format:**

```bash
# On a machine with enough RAM (nexus-control or K1):
python3 -c "
from rkllm.api import RKLLM
llm = RKLLM()
llm.load_huggingface('Qwen/Qwen3.5-0.8B', dtype='w8a8')
llm.build(
    do_quantization=True,
    optimization_level=1,
    quantized_dtype='w8a8',
    target_platform='rk3588'
)
llm.export_rkllm('./qwen3.5-0.8b-w8a8.rkllm')
"

# Transfer to OPi 5 Plus:
scp ./qwen3.5-0.8b-w8a8.rkllm nexus-opi5p:~/models/
```

**Triage HTTP shim + launch script:**

```bash
#!/usr/bin/env bash
# File: ~/nexus-swarm/start-triage.sh (on OPi 5 Plus / nexus-opi5p)

# RKLLM doesn't expose an OpenAI-compatible API natively.
# Use the rkllm-server HTTP shim (wraps the C API in a Flask endpoint).

exec python3 ~/nexus-swarm/rkllm-server.py \
  --model ~/models/qwen3.5-0.8b-w8a8.rkllm \
  --host 0.0.0.0 \
  --port 8088 \
  --max-tokens 32 \
  --system-prompt "You are a request classifier. Given a user request, respond with exactly one word: orchestrator, worker, explore, or embed. orchestrator = planning, reasoning, multi-step tasks. worker = code generation, tool calls, summarization. explore = search, read-only recon, file lookup. embed = embedding generation, similarity search." \
  2>&1 | tee /tmp/triage.log
```

```python
#!/usr/bin/env python3
# File: ~/nexus-swarm/rkllm-server.py (on OPi 5 Plus)
# Minimal OpenAI-compatible HTTP shim for RKLLM inference.

import argparse, json, time
from flask import Flask, request, jsonify
from rkllm.api import RKLLM

app = Flask(__name__)
llm = None
system_prompt = ""

@app.route("/v1/chat/completions", methods=["POST"])
def chat():
    data = request.json
    messages = data.get("messages", [])
    max_tokens = data.get("max_tokens", 32)

    prompt_parts = [system_prompt]
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prompt_parts.append(f"{role}: {content}")
    prompt_parts.append("assistant:")
    prompt = "\n".join(prompt_parts)

    result = llm.run(prompt, max_new_tokens=max_tokens)

    return jsonify({
        "id": f"triage-{int(time.time())}",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result.strip()},
            "finish_reason": "stop"
        }]
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "qwen3.5-0.8b-w8a8", "backend": "rkllm"})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--system-prompt", default="")
    args = parser.parse_args()

    system_prompt = args.system_prompt
    llm = RKLLM()
    llm.load_rkllm(args.model)

    app.run(host=args.host, port=args.port)
```

**Prometheus + Grafana install:**

```bash
# On OPi 5 Plus (nexus-opi5p):

# Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.53.0/prometheus-2.53.0.linux-arm64.tar.gz
tar xzf prometheus-*.tar.gz
sudo mv prometheus-*/prometheus /usr/local/bin/
sudo mv prometheus-*/promtool /usr/local/bin/
sudo mkdir -p /etc/prometheus /var/lib/prometheus

# Node exporter (for system metrics on all nodes)
wget https://github.com/prometheus/node_exporter/releases/download/v1.8.1/node_exporter-1.8.1.linux-arm64.tar.gz
tar xzf node_exporter-*.tar.gz
sudo mv node_exporter-*/node_exporter /usr/local/bin/

# Grafana
sudo apt install -y apt-transport-https software-properties-common
wget -q -O /usr/share/keyrings/grafana.key https://apt.grafana.com/gpg.key
echo "deb [signed-by=/usr/share/keyrings/grafana.key] https://apt.grafana.com stable main" \
  | sudo tee /etc/apt/sources.list.d/grafana.list
sudo apt update && sudo apt install -y grafana
```

### 2.6 OPi 5B — Edge Embeddings (:8087)

The OPi 5B runs llama.cpp compiled for ARM CPU (no CUDA, no NPU). The
nomic-embed model at Q8_0 is ~0.5 GB — well within the 16 GB system RAM.

**Build llama.cpp for ARM:**

```bash
# On OPi 5B (nexus-opi5b):
git clone https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
cd ~/llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

**Download embedding model:**

```bash
huggingface-cli download nomic-ai/nomic-embed-text-v1.5-GGUF \
  nomic-embed-text-v1.5.Q8_0.gguf --local-dir ~/models
```

**Launch script:**

```bash
#!/usr/bin/env bash
# File: ~/nexus-swarm/start-embed.sh (on OPi 5B / nexus-opi5b)

LLAMA=~/llama.cpp/build/bin/llama-server

exec $LLAMA \
  --model ~/models/nomic-embed-text-v1.5.Q8_0.gguf \
  --host 0.0.0.0 \
  --port 8087 \
  --embedding \
  --ctx-size 8192 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --threads $(nproc) \
  --parallel 4 \
  --cont-batching \
  --metrics \
  2>&1 | tee /tmp/llama-embed.log
```

Notes:
- `--embedding`: enables the `/v1/embeddings` endpoint (OpenAI-compatible).
- `--threads $(nproc)`: uses all available ARM cores (RK3588S has 8 cores).
- `--parallel 4`: handle multiple embedding requests concurrently.
- No `--n-gpu-layers`: CPU-only inference on the RK3588S.

---

## 3 — LiteLLM Routing (on nexus-control)

```yaml
# File: ~/nexus-swarm/litellm-config.yaml (on nexus-control / daedelu5)

model_list:
  # ═══ ORCHESTRATOR (5060 Ti on K1 — single instance) ════════
  - model_name: "nexus-orchestrator"
    litellm_params:
      model: "openai/nexus-orchestrator"
      api_base: "http://nexus-k1:8080/v1"
      api_key: "sk-no-key"
      max_tokens: 4096
      timeout: 300
    model_info:
      description: "Qwen2.5-Coder-14B Nexus persona on 5060 Ti — reasoner, planner"

  # ═══ WORKER (3070 on nexus-control — single worker) ════════
  - model_name: "nexus-worker"
    litellm_params:
      model: "openai/nexus-worker-control"
      api_base: "http://nexus-control:8084/v1"
      api_key: "sk-no-key"
      max_tokens: 2048
      timeout: 120
    model_info:
      description: "Qwen3-8B Q4_K_M on 3070 — tool-call, summarize, parse"

  # ═══ EXPLORE (3060 — read-only recon) ══════════════════════
  - model_name: "nexus-explore"
    litellm_params:
      model: "openai/nexus-explore-3060"
      api_base: "http://nexus-3060:8086/v1"
      api_key: "sk-no-key"
      max_tokens: 1024
      timeout: 60
    model_info:
      description: "Gemma-4-E4B on 3060 — edge explore agent"

  # ═══ EMBEDDINGS (OPi 5B) ═══════════════════════════════════
  - model_name: "nexus-embed"
    litellm_params:
      model: "openai/nexus-embed"
      api_base: "http://nexus-opi5b:8087/v1"
      api_key: "sk-no-key"
    model_info:
      description: "nomic-embed on Orange Pi 5B ARM"

  # ═══ TRIAGE (OPi 5 Plus NPU) ═══════════════════════════════
  - model_name: "nexus-triage"
    litellm_params:
      model: "openai/nexus-triage"
      api_base: "http://nexus-opi5p:8088/v1"
      api_key: "sk-no-key"
      max_tokens: 32
      timeout: 10
    model_info:
      description: "Qwen3.5-0.8B on RK3588 NPU — request classifier"

  # ═══ CLAUDE CODE ALIASES ═══════════════════════════════════
  - model_name: "sonnet"
    litellm_params:
      model: "openai/nexus-orchestrator"
      api_base: "http://nexus-k1:8080/v1"
      api_key: "sk-no-key"

  - model_name: "haiku"
    litellm_params:
      model: "openai/nexus-worker-control"
      api_base: "http://nexus-control:8084/v1"
      api_key: "sk-no-key"

router_settings:
  routing_strategy: "simple-shuffle"
  num_retries: 2
  timeout: 300
  retry_after: 5
  allowed_fails: 2
  cooldown_time: 60

general_settings:
  master_key: "sk-nexus-local"
  drop_params: true
```

### Key difference from v3

No `least-busy` load balancing — each model name maps to exactly one backend.
The 3070 on nexus-control is the only worker. When the P40 arrives (upgrade to
v3.b), the orchestrator moves to the P40, the K1's 5060 Ti joins the worker
pool, and `least-busy` re-enables across two worker instances.

---

## 4 — Agent Loop + Model Router

The nexus-harness agent loop runs on nexus-control and routes inference
through LiteLLM to the distributed GPU/NPU fleet.

### 4.1 Model routing in hardware swarm mode

The model router (`crates/nexus-agent/src/router.rs`) classifies each
request by `TaskType` and `Complexity`, resolves to a `Tier`, then maps
the tier to a LiteLLM model name:

```
TaskType (Orchestration/Coding/Analysis/Verification/Creative/Offensive)
    + Complexity (Simple/Moderate/Complex)
    → Tier (Reasoning or Execution)
    → LiteLLM model name
```

| Tier | LiteLLM model | Physical endpoint (v3.a) |
|------|---------------|--------------------------|
| Reasoning | `nexus-orchestrator` | K1 5060 Ti :8080 (Qwen2.5-14B) |
| Execution | `nexus-worker` | nexus-control 3070 :8084 (Qwen3-8B) |

When `hardware_swarm.enabled = true` in `config.toml`, set the Ollama
base URL to point at LiteLLM so all provider calls route through the
proxy transparently:

```toml
# .nexus/config.toml on nexus-control:

[ollama]
base_url = "http://nexus-control:4000"

[ollama.endpoints]
reasoning_base_url = "http://nexus-control:4000"
execution_base_url = "http://nexus-control:4000"
embedding_base_url = "http://nexus-control:4000"

[models]
reasoning = "nexus-orchestrator"
execution = "nexus-worker"
embedding = "nexus-embed"
```

LiteLLM resolves each model name to the correct physical endpoint per
the litellm-config.yaml (§3).

### 4.2 Agent loop flow

The core loop (`run_loop_inner`, lib.rs:1447) runs on nexus-control:

```
user input
  │
  ▼
intent analysis (classify domain/scope/complexity)
  │
  ▼
build system prompt (identity + env + tools + skills + memory)
  │
  ▼
call LiteLLM → nexus-orchestrator (K1 5060 Ti)
  │
  ▼
response = text + tool_calls?
  │
  ├── no tool calls → completion check → return final text
  │
  └── tool calls → for each call:
        permission gate (mode + allow/deny rules)
          ├─ Deny → tool-error result
          └─ Allow → PreToolUse hook
                       ├─ hook blocks → tool-error feedback
                       └─ hook passes → tool.run()
                                          → PostToolUse hook
                                          → append ToolResult
      → loop back to "call LiteLLM"
```

### 4.3 Tool dispatch

Tools execute locally on nexus-control (where the agent loop runs).
There is no ferry-first routing — all built-in tools (`Read`, `Write`,
`Edit`, `Bash`, `Glob`, `Grep`, etc.) run in the nexus-control filesystem.

For cross-node work, the agent uses **delegation** — the `TaskTool`
spawns a subagent that routes its own inference through LiteLLM to a
worker GPU. The subagent's tool calls still execute on nexus-control.

The rust-nexus **ferry** (`nexus-control:9100`) is available for
explicit cross-node task submission when a specific node's capability
is required (e.g., running a tool on a target-adjacent node).

### 4.4 Subagent spawning

Subagents are spawned via `AgentInner::run_subagent()` (lib.rs:1998).
They inherit the provider (LiteLLM) and route with `RouteHint::Coding`
(Execution tier → `nexus-worker`) unless overridden by the subagent
profile's `model:` frontmatter.

```
Orchestrator (K1)  ──delegates──►  Worker subagent (nexus-control 3070)
                                     │
                                     ├── tool calls execute on nexus-control
                                     └── results bubble back to orchestrator
```

### 4.5 Hooks

Hooks fire at 6 lifecycle points:

| Event | Blocking | Purpose in swarm |
|-------|----------|-----------------|
| `SessionStart` | No | Log swarm session start to audit |
| `PreToolUse` | Yes | Scope-check blocks out-of-scope targets |
| `PostToolUse` | No | Audit-log every tool invocation |
| `SessionSuspend` | No | Persist session state |
| `SessionResume` | No | Restore context |
| `Stop` | No | Cleanup |

Hook commands receive environment variables: `NEXUS_EVENT`,
`NEXUS_TOOL_NAME`, `NEXUS_SESSION_ID`, `NEXUS_ENGAGEMENT`,
`NEXUS_RESULTS_DIR`, `NEXUS_CONTEXT`.

---

## 5 — System Prompt

The system prompt is assembled by `build_system_prompt_full()`
(`crates/nexus-agent/src/prompt.rs`) in this order:

### 5.1 Prompt structure

```
┌─────────────────────────────────────────────────────┐
│ 1. Identity block (domain-conditional)              │
│    "You are Nexus, a [domain] agent..."             │
│                                                     │
│ 2. Environment                                      │
│    Working directory, OS, hostname                  │
│                                                     │
│ 3. Tools listing                                    │
│    - Read: Read a file from disk                    │
│    - Write: Create or overwrite a file              │
│    - Bash: Execute a shell command                  │
│    ...one line per registered tool                  │
│                                                     │
│ 4. Skills index (progressive disclosure)            │
│    - recon/nmap-scan: Network port scanning          │
│    - web/nuclei: Template-based vuln scanner         │
│    ...name + description only, body loads on invoke  │
│                                                     │
│ 5. Subagents index                                  │
│    - analyst: Code analysis and review agent         │
│    ...one line per configured agent profile          │
│                                                     │
│ 6. Project memory (NEXUS.md content)                │
│                                                     │
│ 7. Known pitfalls (systemic mistake analysis)       │
│    Anti-patterns detected in prior sessions          │
│                                                     │
│ 8. User profile (persistent characterization)       │
│                                                     │
│ 9. Delegation addendum (swarm-specific)             │
│                                                     │
│ 10. Terminal instruction                            │
│     "When the task is complete, reply with a final  │
│      message and do not request any more tool calls."│
└─────────────────────────────────────────────────────┘
```

### 5.2 Identity blocks

Five domain variants selected by intent analysis:

| Domain | Identity |
|--------|----------|
| Offensive | "You are Nexus, an offensive security agent running inside a hardened Kali Linux environment..." |
| Analysis | "You are Nexus, a code analysis agent..." |
| DevOps | "You are Nexus, a DevOps automation agent..." |
| Creative | "You are Nexus, a design and architecture agent..." |
| Coding/General | "You are Nexus, a local-first coding agent modeled on Claude Code's architecture..." |

### 5.3 Delegation addendum (v3.a topology)

Appended to the system prompt when hardware swarm is active:

```
Your inference runs on an RTX 5060 Ti (16 GB) on the K1 node.
You have one parallel worker (3070 on the control node) and one
explore agent (3060). Delegation is still free but sequential —
fan-out is limited to one worker call at a time unless the explore
agent handles the second.

1. Large tool output (>4000 chars) should be summarized by the worker
   before processing. Use the Task tool to delegate summarization.
2. Read-only search and file discovery can be delegated to the explore
   agent (Gemma-4-E4B on 3060, fast but small context).
3. Never process raw scan output (nmap, nuclei, etc.) yourself. Delegate
   parsing to the worker to keep your context window clean.
4. Embeddings are served by the OPi 5B — use them for semantic search
   over memories and prior session transcripts.
```

---

## 6 — Pipeline Branches (Training)

Fine-tuning pipelines for the worker and explore models. Training runs
asynchronously and does not block swarm operation.

### 6.1 Orchestrator — Qwen V9 Nexus persona

The RTPI pipeline already produces a Qwen2.5-Coder-14B fine-tune (V9)
with the Nexus persona. This is the preferred orchestrator model on K1.

Training data sources:
- Curated system prompt + response pairs from offensive engagements
- Tool-call sequences validated by the anti-pattern guard
- Plan→execute→verify transcripts from successful sessions

### 6.2 Worker — Qwen3-8B tool-call branch

Fine-tune Qwen3-8B for reliable structured output:

```bash
# Training data extraction (on nexus-control):
# Pull successful tool-call sequences from the pgvector intelligence store
psql "$NEXUS_INTEL_DATABASE_URL" -c "
  SELECT input_messages, tool_calls, tool_results
  FROM session_turns
  WHERE tool_call_success = true
    AND anti_pattern_violations = 0
  ORDER BY created_at DESC
  LIMIT 10000
" --csv > ~/training/worker-tool-calls.csv

# Convert to chat-ml format for fine-tuning
python3 ~/training/prepare-worker-dataset.py \
  --input ~/training/worker-tool-calls.csv \
  --output ~/training/worker-sft.jsonl
```

Target capabilities:
- Tool-call JSON generation (reliable structured output)
- Output summarization (compress large tool results)
- Code generation and editing

### 6.3 Explore — Gemma-4-E4B classification branch

Fine-tune Gemma-4-E4B for fast classification and search:

```bash
# Training data: file classification, search relevance scoring
psql "$NEXUS_INTEL_DATABASE_URL" -c "
  SELECT query, results, relevance_score
  FROM search_history
  WHERE relevance_score > 0.7
  ORDER BY created_at DESC
  LIMIT 5000
" --csv > ~/training/explore-search.csv
```

Target capabilities:
- Request classification (route to correct tier)
- Code search relevance scoring
- File/directory exploration summarization

---

## 7 — Systemd Units

One service unit per process, per node. All units follow the security
hardening patterns from rust-nexus deployment examples.

### 7.1 nexus-control (daedelu5) — 5 services

**nexus-infra.service** (rust-nexus infrastructure):

```ini
# File: /etc/systemd/system/nexus-infra.service (on nexus-control)
[Unit]
Description=Nexus Infrastructure (A2A + Ferry)
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
User=cmndcntrl
WorkingDirectory=/home/cmndcntrl
ExecStart=/home/cmndcntrl/nexus-swarm/start-nexus-infra.sh
Restart=on-failure
RestartSec=5s
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=/var/log /tmp
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

**nexus-agent.service** (rust-nexus agent on nexus-control):

```ini
# File: /etc/systemd/system/nexus-agent.service (on nexus-control)
[Unit]
Description=Nexus Agent (nexus-control)
After=nexus-infra.service
Requires=nexus-infra.service

[Service]
Type=exec
User=cmndcntrl
Environment=NEXUS_NODE_ID=nexus-control
Environment=NEXUS_LLAMA_PORT=8084
ExecStart=/home/cmndcntrl/nexus-swarm/start-nexus-agent.sh
Restart=on-failure
RestartSec=5s
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=/tmp
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

**nexus-worker.service** (llama-server worker on 3070):

```ini
# File: /etc/systemd/system/nexus-worker.service (on nexus-control)
[Unit]
Description=Nexus Worker (Qwen3-8B on 3070)
After=network-online.target

[Service]
Type=exec
User=cmndcntrl
ExecStart=/home/cmndcntrl/nexus-swarm/start-worker.sh
Restart=on-failure
RestartSec=10s
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=/home/cmndcntrl/models /tmp
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

**litellm.service** (LiteLLM router):

```ini
# File: /etc/systemd/system/litellm.service (on nexus-control)
[Unit]
Description=LiteLLM Router
After=network-online.target

[Service]
Type=exec
User=cmndcntrl
ExecStart=/usr/local/bin/litellm \
  --config /home/cmndcntrl/nexus-swarm/litellm-config.yaml \
  --port 4000
Restart=on-failure
RestartSec=5s
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=/tmp
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

**nexus-harness.service** (the agent loop itself):

```ini
# File: /etc/systemd/system/nexus-harness.service (on nexus-control)
[Unit]
Description=Nexus Harness Agent Loop
After=litellm.service nexus-infra.service nexus-worker.service
Wants=litellm.service nexus-infra.service nexus-worker.service

[Service]
Type=exec
User=cmndcntrl
WorkingDirectory=/home/cmndcntrl/code/nexus-harness
ExecStart=/home/cmndcntrl/code/nexus-harness/target/release/nexus serve
Restart=on-failure
RestartSec=5s
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=/home/cmndcntrl/code /tmp /var/log
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

### 7.2 nexus-k1 (ACEMAGIC K1) — 2 services

```ini
# File: /etc/systemd/system/nexus-agent.service (on nexus-k1)
[Unit]
Description=Nexus Agent (nexus-k1)
After=network-online.target

[Service]
Type=exec
User=cmndcntrl
Environment=NEXUS_NODE_ID=nexus-k1
Environment=NEXUS_LLAMA_PORT=8080
ExecStart=/home/cmndcntrl/nexus-swarm/start-nexus-agent.sh
Restart=on-failure
RestartSec=5s
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=/tmp

[Install]
WantedBy=multi-user.target
```

```ini
# File: /etc/systemd/system/nexus-orchestrator.service (on nexus-k1)
[Unit]
Description=Nexus Orchestrator (Qwen2.5-14B on 5060 Ti)
After=network-online.target

[Service]
Type=exec
User=cmndcntrl
ExecStart=/home/cmndcntrl/nexus-swarm/start-orchestrator.sh
Restart=on-failure
RestartSec=10s
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=/home/cmndcntrl/models /tmp
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

### 7.3 nexus-3060 — 2 services

Same pattern as K1 with `NEXUS_NODE_ID=nexus-3060`, `NEXUS_LLAMA_PORT=8086`,
and `ExecStart` pointing to `start-explore.sh`.

### 7.4 nexus-opi5p (OPi 5 Plus) — 4 services

`nexus-agent.service`, `nexus-triage.service` (runs `start-triage.sh`),
`prometheus.service`, `grafana-server.service` (installed via apt).

### 7.5 nexus-opi5b (OPi 5B) — 2 services

`nexus-agent.service` and `nexus-embed.service` (runs `start-embed.sh`).

### 7.6 Enable and start all services

```bash
# On each node (via deploy-swarm.sh or manually):
sudo systemctl daemon-reload

# nexus-control:
sudo systemctl enable --now nexus-infra nexus-agent nexus-worker litellm nexus-harness

# nexus-k1:
sudo systemctl enable --now nexus-agent nexus-orchestrator

# nexus-3060:
sudo systemctl enable --now nexus-agent nexus-explore

# nexus-opi5p:
sudo systemctl enable --now nexus-agent nexus-triage prometheus grafana-server

# nexus-opi5b:
sudo systemctl enable --now nexus-agent nexus-embed
```

---

## 8 — Health Monitor

### 8.1 Prometheus scrape config

```yaml
# File: /etc/prometheus/prometheus.yml (on nexus-opi5p)

global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  # rust-nexus infrastructure metrics
  - job_name: "nexus-infra"
    static_configs:
      - targets: ["nexus-control:9100"]
    metrics_path: /metrics

  # llama-server GPU inference endpoints
  - job_name: "llama-servers"
    static_configs:
      - targets:
          - "nexus-k1:8080"        # orchestrator
          - "nexus-control:8084"   # worker
          - "nexus-3060:8086"      # explore
          - "nexus-opi5b:8087"     # embed
    metrics_path: /metrics

  # RKLLM triage (custom /health endpoint, no /metrics)
  - job_name: "nexus-triage"
    static_configs:
      - targets: ["nexus-opi5p:8088"]
    metrics_path: /health
    scrape_interval: 30s

  # Node exporter (system metrics per host)
  - job_name: "node-exporter"
    static_configs:
      - targets:
          - "nexus-control:9101"
          - "nexus-k1:9101"
          - "nexus-3060:9101"
          - "nexus-opi5p:9101"
          - "nexus-opi5b:9101"

  # LiteLLM router metrics
  - job_name: "litellm"
    static_configs:
      - targets: ["nexus-control:4000"]
    metrics_path: /metrics
```

### 8.2 Grafana dashboards

Import or provision these dashboards on Grafana at `nexus-opi5p:3000`:

**Fleet Overview:**
- Connected agents gauge (`nexus_a2a_connected_agents`)
- Per-node CPU/RAM from node-exporter
- Inference latency histograms from llama-server `/metrics`
- Ferry task throughput (`nexus_ferry_tasks_total`)

**Model Health:**
- Tokens/sec per endpoint (llama-server `prompt_tokens_processed`, `tokens_predicted`)
- Queue depth per slot
- KV cache utilization
- Error rates and timeouts

**Agent Activity (requires pgstore):**
- Task completion rate
- Tool-call success vs error ratio
- Anti-pattern violation frequency
- Session duration distribution

### 8.3 Health check timer

```bash
#!/usr/bin/env bash
# File: ~/nexus-swarm/health-check.sh (on nexus-control)
# Run via systemd timer every 60 seconds.

set -euo pipefail
LOG="/var/log/nexus-health.log"
ENDPOINTS=(
  "nexus-k1:8080"
  "nexus-control:8084"
  "nexus-3060:8086"
  "nexus-opi5p:8088"
  "nexus-opi5b:8087"
  "nexus-control:9100"
  "nexus-control:4000"
)

echo "$(date -Iseconds) health-check start" >> "$LOG"
FAIL=0
for ep in "${ENDPOINTS[@]}"; do
  if ! curl -sf --max-time 5 "http://$ep/health" > /dev/null 2>&1; then
    echo "  FAIL: $ep" >> "$LOG"
    FAIL=$((FAIL + 1))
  fi
done

if [ "$FAIL" -gt 0 ]; then
  echo "  $FAIL endpoint(s) down" >> "$LOG"
else
  echo "  all endpoints UP" >> "$LOG"
fi
```

```ini
# File: /etc/systemd/system/nexus-health.timer (on nexus-control)
[Unit]
Description=Nexus Swarm Health Check Timer

[Timer]
OnBootSec=60s
OnUnitActiveSec=60s
AccuracySec=5s

[Install]
WantedBy=timers.target
```

```ini
# File: /etc/systemd/system/nexus-health.service (on nexus-control)
[Unit]
Description=Nexus Swarm Health Check

[Service]
Type=oneshot
User=cmndcntrl
ExecStart=/home/cmndcntrl/nexus-swarm/health-check.sh
```

---

## 9 — End-to-End Swarm Test

A validation script that exercises every tier in the swarm.

```bash
#!/usr/bin/env bash
# File: ~/nexus-swarm/test-swarm-e2e.sh (run from nexus-control)

set -euo pipefail
LITELLM="http://nexus-control:4000"
PASS=0
FAIL=0

check() {
  local name="$1" cmd="$2"
  if eval "$cmd" > /dev/null 2>&1; then
    echo "  ✓ $name"
    PASS=$((PASS + 1))
  else
    echo "  ✗ $name"
    FAIL=$((FAIL + 1))
  fi
}

echo "=== Nexus Swarm v3.a End-to-End Test ==="

# --- Infrastructure ---
echo ""
echo "Infrastructure:"
check "nexus-infra health" \
  "curl -sf http://nexus-control:9100/health"
check "LiteLLM health" \
  "curl -sf http://nexus-control:4000/health"
check "gRPC registry" \
  "grpcurl -plaintext nexus-control:50052 grpc.health.v1.Health/Check"

# --- Inference endpoints ---
echo ""
echo "Inference endpoints:"
check "Orchestrator (K1:8080)" \
  "curl -sf http://nexus-k1:8080/health"
check "Worker (control:8084)" \
  "curl -sf http://nexus-control:8084/health"
check "Explore (3060:8086)" \
  "curl -sf http://nexus-3060:8086/health"
check "Triage (opi5p:8088)" \
  "curl -sf http://nexus-opi5p:8088/health"
check "Embed (opi5b:8087)" \
  "curl -sf http://nexus-opi5b:8087/health"

# --- Model inference ---
echo ""
echo "Model inference via LiteLLM:"
check "Orchestrator completion" \
  "curl -sf $LITELLM/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -H 'Authorization: Bearer sk-nexus-local' \
    -d '{\"model\":\"nexus-orchestrator\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply OK\"}],\"max_tokens\":4}'"

check "Worker completion" \
  "curl -sf $LITELLM/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -H 'Authorization: Bearer sk-nexus-local' \
    -d '{\"model\":\"nexus-worker\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply OK\"}],\"max_tokens\":4}'"

check "Explore completion" \
  "curl -sf $LITELLM/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -H 'Authorization: Bearer sk-nexus-local' \
    -d '{\"model\":\"nexus-explore\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply OK\"}],\"max_tokens\":4}'"

check "Triage classification" \
  "curl -sf $LITELLM/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -H 'Authorization: Bearer sk-nexus-local' \
    -d '{\"model\":\"nexus-triage\",\"messages\":[{\"role\":\"user\",\"content\":\"Scan the target network\"}],\"max_tokens\":4}'"

# --- Embeddings ---
echo ""
echo "Embeddings:"
check "Embedding generation" \
  "curl -sf http://nexus-opi5b:8087/v1/embeddings \
    -H 'Content-Type: application/json' \
    -d '{\"model\":\"nomic-embed\",\"input\":\"test embedding\"}'"

# --- Monitoring ---
echo ""
echo "Monitoring:"
check "Prometheus (opi5p:9090)" \
  "curl -sf http://nexus-opi5p:9090/-/ready"
check "Grafana (opi5p:3000)" \
  "curl -sf http://nexus-opi5p:3000/api/health"

# --- Nexus harness ---
echo ""
echo "Nexus harness:"
check "nexus-cli one-shot" \
  "cd /home/cmndcntrl/code/nexus-harness && \
   timeout 60 cargo run -p nexus-cli --release -- run 'Reply with OK' 2>/dev/null | grep -q 'OK'"

# --- Summary ---
echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ] && echo "All tests passed." || exit 1
```

### 9.1 Run the test

```bash
chmod +x ~/nexus-swarm/test-swarm-e2e.sh
~/nexus-swarm/test-swarm-e2e.sh
```

### 9.2 Expected output

```
=== Nexus Swarm v3.a End-to-End Test ===

Infrastructure:
  ✓ nexus-infra health
  ✓ LiteLLM health
  ✓ gRPC registry

Inference endpoints:
  ✓ Orchestrator (K1:8080)
  ✓ Worker (control:8084)
  ✓ Explore (3060:8086)
  ✓ Triage (opi5p:8088)
  ✓ Embed (opi5b:8087)

Model inference via LiteLLM:
  ✓ Orchestrator completion
  ✓ Worker completion
  ✓ Explore completion
  ✓ Triage classification

Embeddings:
  ✓ Embedding generation

Monitoring:
  ✓ Prometheus (opi5p:9090)
  ✓ Grafana (opi5p:3000)

Nexus harness:
  ✓ nexus-cli one-shot

=== Results: 16 passed, 0 failed ===
All tests passed.
```

### 9.3 File manifest

| Node | Files |
|------|-------|
| nexus-control | `start-nexus-infra.sh`, `start-nexus-agent.sh`, `start-worker.sh`, `litellm-config.yaml`, `deploy-swarm.sh`, `bootstrap-node.sh`, `roles.conf`, `hosts.conf`, `health-check.sh`, `test-swarm-e2e.sh` |
| nexus-k1 | `start-nexus-agent.sh`, `start-orchestrator.sh` |
| nexus-3060 | `start-nexus-agent.sh`, `start-explore.sh` |
| nexus-opi5p | `start-nexus-agent.sh`, `start-triage.sh`, `rkllm-server.py`, `prometheus.yml` |
| nexus-opi5b | `start-nexus-agent.sh`, `start-embed.sh` |

---

## Upgrade Path: v3.a → v3.b

When the GEEKOM AX8 Max + P40 arrive:

1. Bootstrap the AX8 Max via the SSH CA (`bootstrap-node.sh nexus-p40 <ip>`)
2. Add `nexus-p40  orchestrator  sm_61  8080` to `roles.conf`
3. Run `deploy-swarm.sh nexus-p40` to stage llama.cpp (sm_61) and the P40 orchestrator
4. Deploy Gemma 26B-A4B Q4_K_M to the P40 as the new orchestrator (:8080)
5. Reconfigure the K1: stop the 14B orchestrator, start Qwen3-8B Q6_K
   as a worker (:8081) + Gemma-4-E4B explore (:8082)
6. Update `roles.conf`: K1 changes from `orchestrator` to `worker+explore`
7. Update LiteLLM: point `nexus-orchestrator` to `nexus-p40:8080`,
   add `nexus-k1:8081` as a second `nexus-worker` entry,
   switch `routing_strategy` to `least-busy`
8. The fleet is now v3.b — two workers load-balanced, MoE orchestrator
   on 24 GB, and the K1's full 16 GB available for worker+explore

---

## Execution Order

**Phase A0 — SSH CA bootstrap (~10 min):**
1. §0.5.1 — Generate SSH CA on nexus-control
2. §0.5.2 — Bootstrap all 4 remote nodes (sign host keys, push certs)
3. §0.5.2 — Verify passwordless SSH to all nodes

**Phase A1 — LAN + rust-nexus mesh (~20 min):**
4. §1.1 — Hostnames + /etc/hosts (via deploy-swarm.sh or manual)
5. §1.2 — Build rust-nexus on nexus-control, generate mTLS certs, distribute
6. §1.3 — Launch nexus-infra on nexus-control
7. §1.4 — Launch nexus-agent on all 5 nodes
8. §1.5 — Verify mesh (5 agents, all endpoints UP)

**Phase B — GPU inference servers (~30 min per node):**
9. §2.1 — Build llama.cpp per architecture (sm_120 on K1, sm_86 on nexus-control + 3060)
10. §2.2 — Download models per node
11. §2.3–2.4 — Launch servers, verify /health

**Phase B2 — NPU inference servers (~20 min per board):**
12. §2.5 — OPi 5 Plus triage + monitoring, OPi 5B embeddings

**Phase C — Control plane services (nexus-control, ~30 min):**
13. §3 — LiteLLM config + launch + route verification
14. §4 — Agent loop, tools (ferry-first), system prompt

**Phase D — Production hardening (~20 min):**
15. §7 — Systemd units per node
16. §8 — End-to-end swarm test

**Phase E — Training (async):**
17. §5 — Pipeline branches for worker + explore models
