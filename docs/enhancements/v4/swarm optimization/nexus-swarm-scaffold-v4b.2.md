# NEXUS DISTRIBUTED SWARM — Scaffold v3.b (with P40)

> **Purpose:** Six-node swarm with dedicated P40 orchestrator.
> Four NVIDIA GPUs, two RK3588 NPUs.
> The control plane colocates with a 3070 worker on daedelu5.
> This is the target-state architecture. See v3.a for the
> interim config while the GEEKOM AX8 Max + P40 are pending.

---

## 0 — Fleet Inventory

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      NEXUS DISTRIBUTED SWARM v3.b                          │
│                         (full fleet with P40)                              │
│                                                                            │
│  ┌─ GEEKOM AX8 Max ────────────────┐   ┌─ ACEMAGIC K1 ───────────────────┐ │
│  │  GPU: Tesla P40 (24 GB)          │   │  GPU: RTX 5060 Ti (16 GB)       │ │
│  │  via Razer Core X V2 + USB4      │   │  Host: Ryzen 7 7730U · 32 GB    │ │
│  │  Host: Ryzen 7 8745HS            │   │  Role: PRIMARY WORKER           │ │
│  │  32GB DDR5, dual 2.5GbE          │   │  Driver: 580.173.02 / CUDA 13   │ │
│  │  Role: ORCHESTRATOR              │   │                                 │ │
│  │                                  │   │  Models:                        │ │
│  │  Model:                          │   │  • Qwen3-8B Q6_K (tool-call)    │ │
│  │  • Gemma 26B-A4B Q4_K_M          │   │    WORKER           :8081       │ │
│  │    (~14.5 GB + ~8 GB KV)         │   │  • Gemma-4-E4B Q8 (explore)     │ │
│  │  Port: :8080                     │   │    EXPLORE          :8082       │ │
│  └──────────────┬───────────────────┘   └──────────┬──────────────────────┘ │
│                 │                                  │                        │
│  ┌─ LAPTOP-3060 ───────────────────┐                                       │
│  │  GPU: RTX 3060 Laptop (6 GB)    │                                       │
│  │  Role: EDGE EXPLORE             │                                       │
│  │  Driver: 580.173.02             │                                       │
│  │                                 │                                       │
│  │  Model:                         │                                       │
│  │  • Gemma-4-E4B Q8 (classify)    │                                       │
│  │  Port: :8086                    │                                       │
│  └──────────────┬──────────────────┘                                       │
│                 │                                                           │
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
│             │   Role: CONTROL PLANE + PARALLEL WORKER  │                    │
│             │                                          │                    │
│             │   Services:                              │                    │
│             │   • rust-nexus infra (A2A + ferry)       │                    │
│             │   • LiteLLM router          :4000        │                    │
│             │   • nexus-harness agent loop              │                    │
│             │                                          │                    │
│             │   Model:                                 │                    │
│             │   • Qwen3-8B Q4_K_M                      │                    │
│             │     PARALLEL WORKER     :8084             │                    │
│             └──────────────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What the P40 unlocks vs v3.a

| Dimension | v3.a (without P40) | v3.b (with P40) |
|-----------|-------------------|-----------------|
| Orchestrator model | Qwen2.5-14B Q4_K_M (dense, 14B) | Gemma 26B-A4B Q4_K_M (MoE, 26B/4B-active) |
| Orchestrator VRAM | 16 GB (K1's 5060 Ti) | 24 GB (dedicated P40) |
| Orchestrator context | 12K (tight KV budget) | 16K+ (8 GB KV headroom) |
| Worker pool | 1 GPU (3070 only) | 2 GPUs (K1's 5060 Ti + 3070, load-balanced) |
| Worker throughput | Sequential subtasks | Parallel fan-out (2 workers) |
| K1 5060 Ti role | Orchestrator (locked) | Worker + explore (unlocked 16 GB) |
| Tool-call reliability | Strong (Qwen native) | Requires Gemma fixer (§6) |
| Total VRAM | 30 GB across 3 GPUs | 54 GB across 4 GPUs |

### Compute budget

| Node | Accelerator | Memory | Weights | KV headroom | Power |
|------|-------------|--------|---------|-------------|-------|
| AX8 Max (nexus-p40) | Tesla P40 (CUDA) | 24 GB VRAM | ~14.5 GB (Gemma 26B Q4_K_M) | ~8 GB | ~280W |
| K1 (nexus-k1) | RTX 5060 Ti (CUDA) | 16 GB VRAM | ~6.6 GB (Qwen3-8B Q6_K) + ~1.8 GB (E4B Q8) | ~7 GB | ~180W |
| daedelu5 (nexus-control) | RTX 3070 Laptop (CUDA) | 8 GB VRAM | ~5.5 GB (Qwen3-8B Q4_K_M) | ~1.8 GB | ~80W |
| Laptop-3060 | RTX 3060 (CUDA) | 6 GB VRAM | ~3 GB (E4B Q8) | ~1.5 GB | ~80W |
| OPi 5 Plus | RK3588 NPU (RKLLM) | 32 GB sys | ~1.5 GB (Qwen3.5-0.8B w8a8) | ~30 GB | ~15W |
| OPi 5B | RK3588S CPU (ARM) | 16 GB sys | ~0.5 GB (embed) | ~15 GB | ~12W |

### P40 host: GEEKOM AX8 Max

https://www.amazon.com/GEEKOM-AX8-Max-Computers-Business/dp/B0DSPB26NK

- Ryzen 7 8745HS (8C/16T, boost 4.9 GHz, Zen 4)
- **Two USB4 ports (40 Gbps)** — connects Razer Core X V2 for P40 eGPU
- Dual 2.5GbE LAN — plugs into the rust-nexus mesh
- DDR5-5600, expandable to 128 GB via dual SO-DIMM
- 1TB PCIe Gen4 NVMe
- ~$639 (32GB/1TB config)

### Hostnames

| Machine | Hostname | LAN IP | Purpose |
|---------|----------|--------|---------|
| daedelu5 | `nexus-control` | 192.168.8.248 | Control plane + parallel worker |
| GEEKOM AX8 Max | `nexus-p40` | TBD | Orchestrator (P40 eGPU) |
| ACEMAGIC K1 | `nexus-k1` | 192.168.8.238 | Primary worker + explore (5060 Ti) |
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
./bootstrap-node.sh nexus-p40 <TBD>
./bootstrap-node.sh nexus-k1 192.168.8.238
./bootstrap-node.sh nexus-3060 192.168.8.161
./bootstrap-node.sh nexus-opi5p 192.168.8.224
./bootstrap-node.sh nexus-opi5b 192.168.8.221

# Configure nexus-control to trust all CA-signed hosts:
echo "@cert-authority * $(cat ~/.ssh/nexus-ca.pub)" >> ~/.ssh/known_hosts

# Verify (no password, no TOFU):
for node in nexus-p40 nexus-k1 nexus-3060 nexus-opi5p nexus-opi5b; do
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
nexus-p40      orchestrator    sm_61   8080
nexus-k1       worker+explore  sm_120  8081,8082
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
    orchestrator|worker+explore)
      # Build llama.cpp with correct CUDA arch (if not already built)
      ssh "$node" bash -s "$arch" <<'REMOTE'
        ARCH="$1"
        cd ~/llama.cpp
        cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${ARCH#sm_}" \
              -DCMAKE_BUILD_TYPE=Release
        cmake --build build --config Release -j$(nproc)
REMOTE
      # Deploy model + launch scripts
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

All 6 nodes on the same subnet. rust-nexus provides mTLS, A2A agent
registration, and ferry-routed tool execution.

### 1.1 Hostnames

```bash
# File: ~/nexus-swarm/hosts.conf (distributed by deploy-swarm.sh)
# Also applied locally on nexus-control:

cat << 'EOF' | sudo tee /etc/hosts
127.0.0.1     localhost
192.168.8.248  nexus-control
<TBD>          nexus-p40
192.168.8.238  nexus-k1
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

# Generate mTLS certs for all 6 nodes:
./scripts/generate-certs.sh \
  --nodes nexus-control,nexus-p40,nexus-k1,nexus-3060,nexus-opi5p,nexus-opi5b

# Distribute certs via SSH CA (no password prompts):
for node in nexus-p40 nexus-k1 nexus-3060 nexus-opi5p nexus-opi5b; do
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
| AX8 Max | `nexus-p40` | `8080` | orchestrator |
| K1 | `nexus-k1` | `8081` | worker, explore |
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
# Expected: 6 registered agents

curl -s http://nexus-control:9100/health
# Expected: {"status":"ok","agents":6}

for ep in nexus-p40:8080 nexus-k1:8081 nexus-k1:8082 \
          nexus-control:8084 nexus-3060:8086 nexus-opi5p:8088 nexus-opi5b:8087; do
  echo -n "  $ep: "
  curl -sf "http://$ep/health" > /dev/null 2>&1 && echo "UP" || echo "DOWN"
done
```

---

## 2 — GPU Inference Servers

### 2.1 Build llama.cpp per architecture

```bash
# K1 (5060 Ti, sm_120):
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="120" -DCMAKE_BUILD_TYPE=Release

# AX8 MAX / P40 (sm_61):
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="61" -DCMAKE_BUILD_TYPE=Release

# nexus-control / daedelu5 (3070, sm_86) and LAPTOP-3060 (3060, sm_86):
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86" -DCMAKE_BUILD_TYPE=Release
```

### 2.2 Model downloads

```bash
# AX8 MAX: Orchestrator
huggingface-cli download cmndcntrlcyber/gemma26b-offsec-coder-gguf \
  gemma-4-26B-A4B-it-Q4_K_M.gguf --local-dir ~/models

# K1: Worker + Explore
huggingface-cli download bartowski/Qwen3-8B-GGUF Qwen3-8B-Q6_K.gguf --local-dir ~/models
# + Gemma-4-E4B Q8 (per HF availability)

# nexus-control (daedelu5): Parallel Worker
huggingface-cli download bartowski/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf --local-dir ~/models

# LAPTOP-3060: Explore
# Gemma-4-E4B Q8 (per HF availability)
```

### 2.3 Server launch scripts

**AX8 MAX — Orchestrator (:8080)**

```bash
#!/usr/bin/env bash
# File: ~/nexus-swarm/start-orchestrator.sh (on AX8 Max / nexus-p40)

LLAMA=~/llama.cpp/build/bin/llama-server
MODEL=~/models/gemma-4-26B-A4B-it-Q4_K_M.gguf

exec $LLAMA \
  --model "$MODEL" \
  --host 0.0.0.0 \
  --port 8080 \
  --n-gpu-layers 99 \
  --split-mode none \
  --ctx-size 16384 \
  --n-predict 4096 \
  --batch-size 512 \
  --ubatch-size 256 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --slots \
  --slot-save-path /tmp/llama-cache-orch \
  --cont-batching \
  --metrics \
  2>&1 | tee /tmp/orchestrator.log
```

Notes: `--flash-attn` omitted for Pascal (sm_61) — test and add if stable.
`--ctx-size 16384` — the P40's 24 GB with ~14.5 GB model leaves ~8 GB for
KV cache, enough for 16K context with q8_0 quantization.

**K1 — Worker (:8081) + Explore (:8082)**

```bash
#!/usr/bin/env bash
# File: ~/nexus-swarm/start-worker-main.sh (on K1 / nexus-k1)

LLAMA=~/llama.cpp/build/bin/llama-server

exec $LLAMA \
  --model ~/models/Qwen3-8B-Q6_K.gguf \
  --host 0.0.0.0 \
  --port 8081 \
  --n-gpu-layers 99 \
  --split-mode none \
  --ctx-size 8192 \
  --n-predict 2048 \
  --batch-size 1024 \
  --ubatch-size 512 \
  --flash-attn \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --slots \
  --parallel 3 \
  --cont-batching \
  --metrics \
  2>&1 | tee /tmp/llama-worker-main.log
```

```bash
#!/usr/bin/env bash
# File: ~/nexus-swarm/start-worker-explore.sh (on K1 / nexus-k1)

LLAMA=~/llama.cpp/build/bin/llama-server

exec $LLAMA \
  --model ~/models/gemma-4-E4B-it-Q8_0.gguf \
  --host 0.0.0.0 \
  --port 8082 \
  --n-gpu-layers 99 \
  --ctx-size 4096 \
  --n-predict 1024 \
  --batch-size 512 \
  --flash-attn \
  --parallel 2 \
  --cont-batching \
  --metrics \
  2>&1 | tee /tmp/llama-worker-explore.log
```

**nexus-control (daedelu5) — Parallel Worker (:8084)**

```bash
#!/usr/bin/env bash
# File: ~/nexus-swarm/start-worker-parallel.sh (on nexus-control / daedelu5)

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
  2>&1 | tee /tmp/llama-worker-parallel.log
```

**LAPTOP-3060 — Edge Explore (:8086)**

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

---

## 2b — NPU Inference Servers (Orange Pi boards)

### 2b.1 OPi 5 Plus — Triage Gateway (:8088)

The OPi 5 Plus runs RKLLM on its RK3588 NPU (6 TOPS) for ultra-low-latency
request classification. It also hosts Prometheus and Grafana for fleet monitoring.

**RKLLM runtime install:**

```bash
# On OPi 5 Plus (nexus-opi5p):
sudo apt update && sudo apt install -y python3-pip librknnrt-dev
pip3 install rkllm-toolkit

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
llm.build(do_quantization=True, optimization_level=1,
          quantized_dtype='w8a8', target_platform='rk3588')
llm.export_rkllm('./qwen3.5-0.8b-w8a8.rkllm')
"
scp ./qwen3.5-0.8b-w8a8.rkllm nexus-opi5p:~/models/
```

**Launch script + HTTP shim:**

```bash
#!/usr/bin/env bash
# File: ~/nexus-swarm/start-triage.sh (on OPi 5 Plus / nexus-opi5p)

exec python3 ~/nexus-swarm/rkllm-server.py \
  --model ~/models/qwen3.5-0.8b-w8a8.rkllm \
  --host 0.0.0.0 \
  --port 8088 \
  --max-tokens 32 \
  --system-prompt "You are a request classifier. Given a user request, respond with exactly one word: orchestrator, worker, explore, or embed." \
  2>&1 | tee /tmp/triage.log
```

The `rkllm-server.py` OpenAI-compatible HTTP shim is identical to v3.a §2.5.

**Prometheus + Grafana:** Install per v3.a §2.5 instructions.

### 2b.2 OPi 5B — Edge Embeddings (:8087)

```bash
# Build llama.cpp for ARM (on OPi 5B / nexus-opi5b):
git clone https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
cd ~/llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

# Download model:
huggingface-cli download nomic-ai/nomic-embed-text-v1.5-GGUF \
  nomic-embed-text-v1.5.Q8_0.gguf --local-dir ~/models
```

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

---

## 3 — LiteLLM Routing (on nexus-control)

```yaml
# File: ~/nexus-swarm/litellm-config.yaml (on nexus-control / daedelu5)

model_list:
  # ═══ ORCHESTRATOR (P40 on AX8 Max) ═════════════════════════
  - model_name: "nexus-orchestrator"
    litellm_params:
      model: "openai/nexus-orchestrator"
      api_base: "http://nexus-p40:8080/v1"
      api_key: "sk-no-key"
      max_tokens: 4096
      timeout: 300
    model_info:
      description: "Gemma 26B-A4B Nexus persona on P40 — reasoner, planner"

  # ═══ WORKER POOL (load-balanced: K1's 5060 Ti + nexus-control's 3070) ═══
  - model_name: "nexus-worker"
    litellm_params:
      model: "openai/nexus-worker-k1"
      api_base: "http://nexus-k1:8081/v1"
      api_key: "sk-no-key"
      max_tokens: 2048
      timeout: 120
    model_info:
      description: "Qwen3-8B Q6_K on K1's 5060 Ti — primary worker (fastest)"

  - model_name: "nexus-worker"
    litellm_params:
      model: "openai/nexus-worker-control"
      api_base: "http://nexus-control:8084/v1"
      api_key: "sk-no-key"
      max_tokens: 2048
      timeout: 120
    model_info:
      description: "Qwen3-8B Q4_K_M on nexus-control's 3070 — parallel worker"

  # ═══ EXPLORE (E4B on K1's 5060 Ti + 3060, load-balanced) ═══
  - model_name: "nexus-explore"
    litellm_params:
      model: "openai/nexus-explore-k1"
      api_base: "http://nexus-k1:8082/v1"
      api_key: "sk-no-key"
      max_tokens: 1024
      timeout: 60
    model_info:
      description: "Gemma-4-E4B on K1's 5060 Ti — fast explore"

  - model_name: "nexus-explore"
    litellm_params:
      model: "openai/nexus-explore-3060"
      api_base: "http://nexus-3060:8086/v1"
      api_key: "sk-no-key"
      max_tokens: 1024
      timeout: 60
    model_info:
      description: "Gemma-4-E4B on 3060 — edge explore"

  # ═══ EMBEDDINGS (OPi 5B) ═══════════════════════════════════
  - model_name: "nexus-embed"
    litellm_params:
      model: "openai/nexus-embed"
      api_base: "http://nexus-opi5b:8087/v1"
      api_key: "sk-no-key"

  # ═══ TRIAGE (OPi 5 Plus NPU) ═══════════════════════════════
  - model_name: "nexus-triage"
    litellm_params:
      model: "openai/nexus-triage"
      api_base: "http://nexus-opi5p:8088/v1"
      api_key: "sk-no-key"
      max_tokens: 32
      timeout: 10

  # ═══ CLAUDE CODE ALIASES ═══════════════════════════════════
  - model_name: "sonnet"
    litellm_params:
      model: "openai/nexus-orchestrator"
      api_base: "http://nexus-p40:8080/v1"
      api_key: "sk-no-key"

  - model_name: "haiku"
    litellm_params:
      model: "openai/nexus-worker-k1"
      api_base: "http://nexus-k1:8081/v1"
      api_key: "sk-no-key"

router_settings:
  routing_strategy: "least-busy"
  num_retries: 2
  timeout: 300
  retry_after: 5
  allowed_fails: 2
  cooldown_time: 60

general_settings:
  master_key: "sk-nexus-local"
  drop_params: true
```

### Key difference from v3.a

`least-busy` routing across two `nexus-worker` and two `nexus-explore`
instances. When the orchestrator fans out two parallel subtasks, each
lands on a different GPU — one on K1's 5060 Ti, one on nexus-control's 3070.

---

## 4 — Agent Loop + Model Router

The nexus-harness agent loop runs on nexus-control and routes inference
through LiteLLM to the distributed GPU/NPU fleet.

### 4.1 Model routing in hardware swarm mode

The model router (`crates/nexus-agent/src/router.rs`) classifies each
request by `TaskType` and `Complexity`, resolves to a `Tier`, then maps
the tier to a LiteLLM model name:

| Tier | LiteLLM model | Physical endpoint (v3.b) |
|------|---------------|--------------------------|
| Reasoning | `nexus-orchestrator` | AX8 Max P40 :8080 (Gemma 26B-A4B) |
| Execution | `nexus-worker` | K1 5060 Ti :8081 OR nexus-control 3070 :8084 (load-balanced) |

When `hardware_swarm.enabled = true`:

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

### 4.2 Agent loop flow

Same as v3.a §4.2 with one key difference: when the orchestrator (Gemma 26B
on P40) returns tool calls, they pass through the **Gemma tool-call fixer**
(§6) before dispatch. The fixer repairs any malformed JSON before the
permission gate evaluates the call.

```
call LiteLLM → nexus-orchestrator (AX8 Max P40)
  │
  ▼
Gemma tool-call fixer (§6) — repair malformed JSON
  │
  ▼
for each tool_call: permission gate → hooks → tool.run()
```

### 4.3 Tool dispatch

Tools execute locally on nexus-control. For delegation, the `TaskTool`
spawns subagents that route through LiteLLM to workers. With v3.b's
load-balanced worker pool, LiteLLM's `least-busy` strategy distributes
subagent inference across K1's 5060 Ti and nexus-control's 3070.

```
Orchestrator (P40)  ──delegates──►  Worker subagent A (K1 5060 Ti)
                    ──delegates──►  Worker subagent B (nexus-control 3070)
                                     │  (simultaneous — different GPUs)
                                     ├── tool calls execute on nexus-control
                                     └── results bubble back to orchestrator
```

### 4.4 Hooks

Same as v3.a §4.5.

---

## 5 — System Prompt

### 5.1 Prompt structure

Same as v3.a §5.1. The system prompt is assembled by
`build_system_prompt_full()` with identity, environment, tools, skills,
subagents, memory, pitfalls, user profile, and delegation addendum.

### 5.2 Delegation addendum (v3.b topology)

```
Your inference runs on a Tesla P40 (24 GB) on the AX8 Max node.
Workers run on separate GPUs (K1's 5060 Ti + nexus-control's 3070,
load-balanced via LiteLLM). You have two parallel workers and two
explore agents. Delegation is free — fan out independent subtasks
to keep your context window clean.

1. Large tool output (>4000 chars) is auto-summarized by a worker.
   Use the Task tool to delegate. The worker lands on whichever GPU
   is least busy (K1 or nexus-control).
2. Read-only search is delegated to explore agents (Gemma-4-E4B on
   K1 :8082 + 3060 :8086, load-balanced).
3. You can dispatch two subtasks simultaneously — they land on
   different GPUs. LiteLLM's least-busy routing ensures no collision.
4. Never process raw scan output yourself. Send to a worker first.
5. Embeddings are served by the OPi 5B — use them for semantic search
   over memories and prior session transcripts.
```

---

## 6 — Gemma Tool-Call Fixer

The Gemma 26B-A4B orchestrator may produce malformed tool-call JSON.
Unlike Qwen models (which have native tool-call support in llama.cpp),
Gemma's tool-call output can contain:

- Trailing text after the JSON object
- Missing closing braces or brackets
- Unquoted keys or single-quoted strings
- Escaped characters that break JSON parsing
- Tool call wrapped in markdown code fences

### 6.1 Fix strategy

A three-stage repair pipeline runs between `provider.chat()` and
tool-call extraction:

**Stage 1 — Regex cleanup (fast, handles ~80% of cases):**

```python
import re, json

def fix_gemma_tool_call(raw: str) -> str:
    # Strip markdown code fences
    raw = re.sub(r'```(?:json)?\s*', '', raw)
    raw = re.sub(r'```\s*$', '', raw)

    # Strip trailing text after the last closing brace/bracket
    match = re.search(r'(\{.*\}|\[.*\])', raw, re.DOTALL)
    if match:
        raw = match.group(1)

    # Fix single quotes → double quotes (outside of string values)
    raw = raw.replace("'", '"')

    # Try parsing
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        return None  # escalate to stage 2
```

**Stage 2 — Structural repair (handles ~15% of remaining):**

```python
def repair_json_structure(raw: str) -> str:
    # Balance braces
    open_braces = raw.count('{') - raw.count('}')
    open_brackets = raw.count('[') - raw.count(']')
    raw += '}' * max(0, open_braces)
    raw += ']' * max(0, open_brackets)

    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        return None  # escalate to stage 3
```

**Stage 3 — LLM re-prompt (fallback for ~5%):**

Route the malformed output to a Qwen3-8B worker (which handles
structured output reliably) with a repair prompt:

```
The following tool call JSON is malformed. Fix it and return ONLY
valid JSON, nothing else:

{malformed_output}
```

This adds one worker inference call (~200ms on the 5060 Ti) but only
triggers for the ~5% of cases that regex/structural repair can't fix.

### 6.2 Integration point

In the agent loop (`run_loop_inner`), after receiving the orchestrator's
response and before extracting tool calls:

```rust
// Pseudocode — actual implementation in nexus-agent/src/lib.rs
let response = provider.chat(&request).await?;

// Gemma fixer: only when orchestrator is Gemma
if config.hardware_swarm.enabled && is_gemma_orchestrator(&config) {
    for content in &mut response.content {
        if let Content::ToolUse(ref mut tc) = content {
            if let Err(e) = serde_json::from_value::<Value>(tc.input.clone()) {
                tc.input = fix_tool_call_json(&tc.input, &provider, &config).await?;
            }
        }
    }
}
```

### 6.3 When the fixer is NOT needed

- **v3.a**: Qwen2.5-14B orchestrator → native tool-call support, no fixer
- **v3.b with Qwen orchestrator**: if the RTPI fine-tune is used instead
  of Gemma, disable the fixer
- **Claude API mode**: Anthropic models handle tool calls natively

---

## 7 — Pipeline Branches (Training)

### 7.1 Orchestrator — Gemma 26B Nexus persona

Fine-tune the Gemma 26B-A4B with the Nexus persona and improved
tool-call formatting (reducing the fixer's stage 3 fallback rate).

Training data:
- Curated tool-call sequences from successful sessions
- Synthetic tool-call JSON examples with correct formatting
- Plan→execute→verify transcripts

### 7.2 Worker — Qwen3-8B tool-call branch

Same as v3.a §6.2 — fine-tune for reliable structured output,
summarization, and code generation.

### 7.3 Explore — Gemma-4-E4B classification branch

Same as v3.a §6.3 — fine-tune for fast classification and search.

---

## 8 — Systemd Units

### 8.1 nexus-control (daedelu5) — 5 services

Same as v3.a §7.1: `nexus-infra`, `nexus-agent`, `nexus-worker` (renamed
`nexus-worker-parallel` since it's a parallel worker alongside K1),
`litellm`, `nexus-harness`.

### 8.2 nexus-p40 (AX8 Max) — 2 services

```ini
# File: /etc/systemd/system/nexus-agent.service (on nexus-p40)
[Unit]
Description=Nexus Agent (nexus-p40)
After=network-online.target

[Service]
Type=exec
User=cmndcntrl
Environment=NEXUS_NODE_ID=nexus-p40
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
# File: /etc/systemd/system/nexus-orchestrator.service (on nexus-p40)
[Unit]
Description=Nexus Orchestrator (Gemma 26B-A4B on P40)
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

### 8.3 nexus-k1 (ACEMAGIC K1) — 3 services

`nexus-agent.service`, `nexus-worker-main.service` (Qwen3-8B Q6_K :8081),
`nexus-worker-explore.service` (Gemma-4-E4B Q8 :8082).

```ini
# File: /etc/systemd/system/nexus-worker-main.service (on nexus-k1)
[Unit]
Description=Nexus Worker Main (Qwen3-8B Q6_K on 5060 Ti)
After=network-online.target

[Service]
Type=exec
User=cmndcntrl
ExecStart=/home/cmndcntrl/nexus-swarm/start-worker-main.sh
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

```ini
# File: /etc/systemd/system/nexus-worker-explore.service (on nexus-k1)
[Unit]
Description=Nexus Explore (Gemma-4-E4B Q8 on 5060 Ti)
After=network-online.target

[Service]
Type=exec
User=cmndcntrl
ExecStart=/home/cmndcntrl/nexus-swarm/start-worker-explore.sh
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

### 8.4 Other nodes

- **nexus-3060**: `nexus-agent.service` + `nexus-explore.service` (same as v3.a)
- **nexus-opi5p**: `nexus-agent.service` + `nexus-triage.service` + `prometheus.service` + `grafana-server.service`
- **nexus-opi5b**: `nexus-agent.service` + `nexus-embed.service`

### 8.5 Enable and start all services

```bash
# nexus-control:
sudo systemctl enable --now nexus-infra nexus-agent nexus-worker-parallel litellm nexus-harness

# nexus-p40:
sudo systemctl enable --now nexus-agent nexus-orchestrator

# nexus-k1:
sudo systemctl enable --now nexus-agent nexus-worker-main nexus-worker-explore

# nexus-3060:
sudo systemctl enable --now nexus-agent nexus-explore

# nexus-opi5p:
sudo systemctl enable --now nexus-agent nexus-triage prometheus grafana-server

# nexus-opi5b:
sudo systemctl enable --now nexus-agent nexus-embed
```

---

## 9 — Health Monitor

### 9.1 Prometheus scrape config

```yaml
# File: /etc/prometheus/prometheus.yml (on nexus-opi5p)

global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: "nexus-infra"
    static_configs:
      - targets: ["nexus-control:9100"]

  - job_name: "llama-servers"
    static_configs:
      - targets:
          - "nexus-p40:8080"       # orchestrator (P40)
          - "nexus-k1:8081"        # worker (5060 Ti)
          - "nexus-k1:8082"        # explore (5060 Ti)
          - "nexus-control:8084"   # parallel worker (3070)
          - "nexus-3060:8086"      # edge explore
          - "nexus-opi5b:8087"     # embed
    metrics_path: /metrics

  - job_name: "nexus-triage"
    static_configs:
      - targets: ["nexus-opi5p:8088"]
    metrics_path: /health
    scrape_interval: 30s

  - job_name: "node-exporter"
    static_configs:
      - targets:
          - "nexus-control:9101"
          - "nexus-p40:9101"
          - "nexus-k1:9101"
          - "nexus-3060:9101"
          - "nexus-opi5p:9101"
          - "nexus-opi5b:9101"

  - job_name: "litellm"
    static_configs:
      - targets: ["nexus-control:4000"]
    metrics_path: /metrics
```

### 9.2 Grafana dashboards

Same dashboard structure as v3.a §8.2 with additional panels for:
- Worker pool load balance (K1 vs nexus-control request distribution)
- P40 orchestrator VRAM utilization and KV cache fill
- Gemma tool-call fixer stage distribution (stage 1/2/3 hit rates)

### 9.3 Health check timer

Same as v3.a §8.3 with the addition of `nexus-p40:8080`, `nexus-k1:8081`,
and `nexus-k1:8082` to the endpoint list.

---

## 10 — End-to-End Swarm Test

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

echo "=== Nexus Swarm v3.b End-to-End Test ==="

echo ""
echo "Infrastructure:"
check "nexus-infra health" \
  "curl -sf http://nexus-control:9100/health"
check "LiteLLM health" \
  "curl -sf http://nexus-control:4000/health"

echo ""
echo "Inference endpoints:"
check "Orchestrator (P40:8080)" \
  "curl -sf http://nexus-p40:8080/health"
check "Worker-main (K1:8081)" \
  "curl -sf http://nexus-k1:8081/health"
check "Worker-explore (K1:8082)" \
  "curl -sf http://nexus-k1:8082/health"
check "Worker-parallel (control:8084)" \
  "curl -sf http://nexus-control:8084/health"
check "Explore (3060:8086)" \
  "curl -sf http://nexus-3060:8086/health"
check "Triage (opi5p:8088)" \
  "curl -sf http://nexus-opi5p:8088/health"
check "Embed (opi5b:8087)" \
  "curl -sf http://nexus-opi5b:8087/health"

echo ""
echo "Model inference via LiteLLM:"
check "Orchestrator completion" \
  "curl -sf $LITELLM/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -H 'Authorization: Bearer sk-nexus-local' \
    -d '{\"model\":\"nexus-orchestrator\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply OK\"}],\"max_tokens\":4}'"

check "Worker completion (load-balanced)" \
  "curl -sf $LITELLM/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -H 'Authorization: Bearer sk-nexus-local' \
    -d '{\"model\":\"nexus-worker\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply OK\"}],\"max_tokens\":4}'"

check "Explore completion (load-balanced)" \
  "curl -sf $LITELLM/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -H 'Authorization: Bearer sk-nexus-local' \
    -d '{\"model\":\"nexus-explore\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply OK\"}],\"max_tokens\":4}'"

check "Triage classification" \
  "curl -sf $LITELLM/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -H 'Authorization: Bearer sk-nexus-local' \
    -d '{\"model\":\"nexus-triage\",\"messages\":[{\"role\":\"user\",\"content\":\"Scan the target network\"}],\"max_tokens\":4}'"

echo ""
echo "Load balance verification (2 consecutive worker calls):"
ROUTE1=$(curl -sf $LITELLM/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-nexus-local' \
  -d '{"model":"nexus-worker","messages":[{"role":"user","content":"Reply OK"}],"max_tokens":4}' \
  2>/dev/null | jq -r '.model // "unknown"')
ROUTE2=$(curl -sf $LITELLM/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-nexus-local' \
  -d '{"model":"nexus-worker","messages":[{"role":"user","content":"Reply OK"}],"max_tokens":4}' \
  2>/dev/null | jq -r '.model // "unknown"')
echo "  Worker call 1 routed to: $ROUTE1"
echo "  Worker call 2 routed to: $ROUTE2"

echo ""
echo "Embeddings:"
check "Embedding generation" \
  "curl -sf http://nexus-opi5b:8087/v1/embeddings \
    -H 'Content-Type: application/json' \
    -d '{\"model\":\"nomic-embed\",\"input\":\"test embedding\"}'"

echo ""
echo "Monitoring:"
check "Prometheus (opi5p:9090)" \
  "curl -sf http://nexus-opi5p:9090/-/ready"
check "Grafana (opi5p:3000)" \
  "curl -sf http://nexus-opi5p:3000/api/health"

echo ""
echo "Nexus harness:"
check "nexus-cli one-shot" \
  "cd /home/cmndcntrl/code/nexus-harness && \
   timeout 60 cargo run -p nexus-cli --release -- run 'Reply with OK' 2>/dev/null | grep -q 'OK'"

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ] && echo "All tests passed." || exit 1
```

### 10.1 File manifest

| Node | Files |
|------|-------|
| nexus-control | `start-nexus-infra.sh`, `start-nexus-agent.sh`, `start-worker-parallel.sh`, `litellm-config.yaml`, `deploy-swarm.sh`, `bootstrap-node.sh`, `roles.conf`, `hosts.conf`, `health-check.sh`, `test-swarm-e2e.sh` |
| nexus-p40 | `start-nexus-agent.sh`, `start-orchestrator.sh` |
| nexus-k1 | `start-nexus-agent.sh`, `start-worker-main.sh`, `start-worker-explore.sh` |
| nexus-3060 | `start-nexus-agent.sh`, `start-explore.sh` |
| nexus-opi5p | `start-nexus-agent.sh`, `start-triage.sh`, `rkllm-server.py`, `prometheus.yml` |
| nexus-opi5b | `start-nexus-agent.sh`, `start-embed.sh` |

---

## Execution Order

**Phase A0 — SSH CA bootstrap (~10 min):**
1. §0.5.1 — Generate SSH CA on nexus-control
2. §0.5.2 — Bootstrap all 5 remote nodes (sign host keys, push certs)
3. §0.5.2 — Verify passwordless SSH to all nodes

**Phase A1 — LAN + rust-nexus mesh (~20 min):**
4. §1.1 — Hostnames + /etc/hosts on all 6 nodes (via deploy-swarm.sh)
5. §1.2 — Build rust-nexus on nexus-control, generate mTLS certs, distribute via SSH CA
6. §1.3 — Launch nexus-infra on nexus-control
7. §1.4 — Launch nexus-agent on all 6 nodes
8. §1.5 — Verify mesh (6 agents, all endpoints UP)

**Phase B — GPU inference servers (~30 min per node):**
9. §2.1 — Build llama.cpp per architecture (sm_120 on K1, sm_61 on AX8 Max, sm_86 on nexus-control + 3060)
10. §2.2 — Download models per node
11. §2.3 — Launch servers, verify /health

**Phase B2 — NPU inference servers (~20 min per board):**
12. §2b — OPi 5 Plus triage + monitoring, OPi 5B embeddings

**Phase C — Control plane services (nexus-control, ~30 min):**
13. §3 — LiteLLM config + launch + route verification
14. §4 — Agent loop, tools (ferry-first), system prompt
15. §6 — Gemma tool-call fixer

**Phase D — Production hardening (~20 min):**
16. §7 — Systemd units per node
17. §8 — End-to-end swarm test

**Phase E — Training (async):**
18. §5 — Pipeline branches for worker + explore models
