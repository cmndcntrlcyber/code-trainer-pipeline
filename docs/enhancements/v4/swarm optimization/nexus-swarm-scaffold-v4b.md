# NEXUS DISTRIBUTED SWARM — Scaffold v3.b (with P40)

> **Purpose:** Seven-node swarm with dedicated P40 orchestrator.
> Four NVIDIA GPUs, two RK3588 NPUs, one control plane.
> This is the target-state architecture. See v3.a for the
> interim config while the GEEKOM AX8 Max + P40 are pending.

---

## 0 — Fleet Inventory

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      NEXUS DISTRIBUTED SWARM v3.b                          │
│                         (full fleet with P40)                              │
│                                                                            │
│  ┌─ GEEKOM AX8 Max ────────────┐   ┌─ DESKTOP ────────────────────────┐   │
│  │  GPU: Tesla P40 (24 GB)      │   │  GPU: RTX 5060 Ti (16 GB)       │   │
│  │  via Razer Core X V2 + USB4  │   │  Role: PRIMARY WORKER            │   │
│  │  Host: Ryzen 7 8745HS        │   │  Driver: 580.173.02 / CUDA 13   │   │
│  │  32GB DDR5, dual 2.5GbE      │   │                                  │   │
│  │  Role: ORCHESTRATOR          │   │  Models:                         │   │
│  │                              │   │  • Qwen3-8B Q6_K (tool-call)     │   │
│  │  Model:                      │   │    WORKER           :8081        │   │
│  │  • Gemma 26B-A4B Q4_K_M      │   │  • Gemma-4-E4B Q8 (explore)     │   │
│  │    (~14.5 GB + ~8 GB KV)     │   │    EXPLORE          :8082        │   │
│  │  Port: :8080                 │   │                                  │   │
│  └──────────────┬───────────────┘   └──────────┬───────────────────────┘   │
│                 │                               │                          │
│  ┌─ LAPTOP-3070 ────────────────┐   ┌─ LAPTOP-3060 ───────────────────┐   │
│  │  GPU: RTX 3070 Laptop (8 GB) │   │  GPU: RTX 3060 Laptop (6 GB)    │   │
│  │  Role: PARALLEL WORKER       │   │  Role: EDGE WORKER               │   │
│  │  Driver: 560.35.03           │   │  Driver: 580.173.02             │   │
│  │                              │   │                                  │   │
│  │  Model:                      │   │  Model:                          │   │
│  │  • Qwen3-8B Q4_K_M           │   │  • Gemma-4-E4B Q8 (classify)    │   │
│  │    (~5.5 GB + ~2 GB KV)      │   │  Port: :8086                     │   │
│  │  Port: :8084                 │   │                                  │   │
│  └──────────────┬───────────────┘   └──────────┬───────────────────────┘   │
│                 │                               │                          │
│  ┌─ OPi 5 Plus (32GB) ─────────┐   ┌─ OPi 5B (16GB) ─────────────────┐   │
│  │  SoC: RK3588 · NPU 6 TOPS   │   │  SoC: RK3588S · NPU 6 TOPS      │   │
│  │  Role: TRIAGE GATEWAY        │   │  Role: EDGE EMBEDDINGS           │   │
│  │  Dual 2.5GbE, M.2 NVMe      │   │  WiFi 6, BT 5.0                 │   │
│  │                              │   │  128GB eMMC                      │   │
│  │  Models (RKLLM, NPU):       │   │                                  │   │
│  │  • Qwen3.5-0.8B w8a8         │   │  Models (llama.cpp ARM):         │   │
│  │    (request classification)   │   │  • nomic-embed Q8                │   │
│  │  Services:                   │   │  Port: :8087                     │   │
│  │  • Triage API       :8088    │   │                                  │   │
│  │  • Prometheus        :9090    │   │                                  │   │
│  │  • Grafana           :3000    │   │                                  │   │
│  └──────────────┬───────────────┘   └──────────┬───────────────────────┘   │
│                 │                               │                          │
│             ┌───▼───────────────────────────────▼────┐                     │
│             │   ACEMAGIC K1 (no GPU, no NPU)         │                     │
│             │   Ryzen 7 7730U · 32 GB DDR4           │                     │
│             │   Role: CONTROL PLANE                  │                     │
│             │                                        │                     │
│             │   Services:                            │                     │
│             │   • rust-nexus infra (A2A + ferry)     │                     │
│             │   • LiteLLM router          :4000      │                     │
│             │   • nexus-harness agent loop            │                     │
│             └────────────────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What the P40 unlocks vs v3.a

| Dimension | v3.a (without P40) | v3.b (with P40) |
|-----------|-------------------|-----------------|
| Orchestrator model | Qwen2.5-14B Q4_K_M (dense, 14B) | Gemma 26B-A4B Q4_K_M (MoE, 26B/4B-active) |
| Orchestrator VRAM | 16 GB (shared with desktop) | 24 GB (dedicated P40) |
| Orchestrator context | 12K (tight KV budget) | 16K+ (8 GB KV headroom) |
| Worker pool | 1 GPU (3070 only) | 2 GPUs (5060 Ti + 3070, load-balanced) |
| Worker throughput | Sequential subtasks | Parallel fan-out (2 workers) |
| 5060 Ti role | Orchestrator (locked) | Worker + explore (unlocked 16 GB) |
| Tool-call reliability | Strong (Qwen native) | Requires Gemma fixer (§6) |
| Total VRAM | 30 GB across 3 GPUs | 54 GB across 4 GPUs |

### Compute budget

| Node | Accelerator | Memory | Weights | KV headroom | Power |
|------|-------------|--------|---------|-------------|-------|
| AX8 Max | Tesla P40 (CUDA) | 24 GB VRAM | ~14.5 GB (Gemma 26B Q4_K_M) | ~8 GB | ~280W |
| Desktop | RTX 5060 Ti (CUDA) | 16 GB VRAM | ~6.6 GB (Qwen3-8B Q6_K) + ~1.8 GB (E4B Q8) | ~7 GB | ~180W |
| Laptop-3070 | RTX 3070 (CUDA) | 8 GB VRAM | ~5.5 GB (Qwen3-8B Q4_K_M) | ~1.8 GB | ~80W |
| Laptop-3060 | RTX 3060 (CUDA) | 6 GB VRAM | ~3 GB (E4B Q8) | ~1.5 GB | ~80W |
| OPi 5 Plus | RK3588 NPU (RKLLM) | 32 GB sys | ~1.5 GB (Qwen3.5-0.8B w8a8) | ~30 GB | ~15W |
| OPi 5B | RK3588S CPU (ARM) | 16 GB sys | ~0.5 GB (embed) | ~15 GB | ~12W |
| K1 | None | 32 GB DDR4 | — | — | ~45W |

### P40 host: GEEKOM AX8 Max

https://www.amazon.com/GEEKOM-AX8-Max-Computers-Business/dp/B0DSPB26NK

- Ryzen 7 8745HS (8C/16T, boost 4.9 GHz, Zen 4)
- **Two USB4 ports (40 Gbps)** — connects Razer Core X V2 for P40 eGPU
- Dual 2.5GbE LAN — plugs into the rust-nexus mesh
- DDR5-5600, expandable to 128 GB via dual SO-DIMM
- 1TB PCIe Gen4 NVMe
- ~$639 (32GB/1TB config)

### Hostnames

| Machine | Hostname | LAN IP (example) | Purpose |
|---------|----------|-------------------|---------|
| Desktop | `nexus-desktop` | 192.168.1.10 | Primary worker + explore |
| GEEKOM AX8 Max | `nexus-p40` | 192.168.1.11 | Orchestrator (P40 eGPU) |
| 3070 laptop | `nexus-3070` | 192.168.1.12 | Parallel worker |
| 3060 laptop | `nexus-3060` | 192.168.1.13 | Edge explore |
| OPi 5 Plus | `nexus-opi5p` | 192.168.1.14 | Triage + monitoring |
| OPi 5B | `nexus-opi5b` | 192.168.1.15 | Embeddings |
| ACEMAGIC K1 | `nexus-control` | 192.168.1.20 | Control plane |

---

## 1 — Network Mesh (rust-nexus on LAN)

All 7 nodes on the same subnet. rust-nexus provides mTLS, A2A agent
registration, and ferry-routed tool execution.

### 1.1 Hostnames

```bash
# On every node:
sudo hostnamectl set-hostname <HOSTNAME>

cat << 'EOF' | sudo tee -a /etc/hosts
192.168.1.10  nexus-desktop
192.168.1.11  nexus-p40
192.168.1.12  nexus-3070
192.168.1.13  nexus-3060
192.168.1.14  nexus-opi5p
192.168.1.15  nexus-opi5b
192.168.1.20  nexus-control
EOF
```

### 1.2 Build and deploy rust-nexus

```bash
# On K1 (nexus-control):
git clone https://github.com/cmndcntrlcyber/rust-nexus.git ~/rust-nexus
cd ~/rust-nexus
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
cargo build --workspace --release

# Generate mTLS certs for all 7 nodes:
./scripts/generate-certs.sh \
  --nodes nexus-desktop,nexus-p40,nexus-3070,nexus-3060,nexus-opi5p,nexus-opi5b,nexus-control

# Distribute certs:
for node in nexus-desktop nexus-p40 nexus-3070 nexus-3060 nexus-opi5p nexus-opi5b; do
  ./scripts/transfer-prep.sh --ip $(getent hosts $node | awk '{print $1}') --user $(whoami)
done
```

### 1.3 Launch nexus-infra on K1

```bash
#!/usr/bin/env bash
# File: ~/nexus-swarm/start-nexus-infra.sh (on K1)

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

### 1.4 Launch nexus-agent on each GPU/NPU node

```bash
#!/usr/bin/env bash
# File: ~/nexus-swarm/start-nexus-agent.sh (on each node)

NODE_ID="${NEXUS_NODE_ID:-nexus-desktop}"
INFRA_ADDR="nexus-control:50052"
LLAMA_PORT="${NEXUS_LLAMA_PORT:-8081}"

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
| Desktop | `nexus-desktop` | `8081` | worker, explore |
| AX8 Max | `nexus-p40` | `8080` | orchestrator |
| 3070 laptop | `nexus-3070` | `8084` | worker |
| 3060 laptop | `nexus-3060` | `8086` | explore |
| OPi 5 Plus | `nexus-opi5p` | `8088` | triage |
| OPi 5B | `nexus-opi5b` | `8087` | embed |

### 1.5 Verify

```bash
# From K1:
grpcurl -cacert ~/.nexus/certs/ca.pem \
  -cert ~/.nexus/certs/nexus-control.pem \
  -key ~/.nexus/certs/nexus-control-key.pem \
  nexus-control:50052 nexus.a2a.RegistryLister/ListAgents
# Expected: 6 registered agents

curl -s http://nexus-control:9100/health
# Expected: {"status":"ok","agents":6}

for ep in nexus-p40:8080 nexus-desktop:8081 nexus-desktop:8082 \
          nexus-3070:8084 nexus-3060:8086 nexus-opi5p:8088 nexus-opi5b:8087; do
  echo -n "  $ep: "
  curl -sf "http://$ep/health" > /dev/null 2>&1 && echo "UP" || echo "DOWN"
done
```

---

## 2 — GPU Inference Servers

### 2.1 Build llama.cpp per architecture

```bash
# DESKTOP (5060 Ti, sm_120):
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="120" -DCMAKE_BUILD_TYPE=Release

# AX8 MAX / P40 (sm_61):
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="61" -DCMAKE_BUILD_TYPE=Release

# LAPTOPS (Ampere, sm_86):
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86" -DCMAKE_BUILD_TYPE=Release
```

### 2.2 Model downloads

```bash
# AX8 MAX: Orchestrator
huggingface-cli download cmndcntrlcyber/gemma26b-offsec-coder-gguf \
  gemma-4-26B-A4B-it-Q4_K_M.gguf --local-dir ~/models

# DESKTOP: Worker + Explore
huggingface-cli download bartowski/Qwen3-8B-GGUF Qwen3-8B-Q6_K.gguf --local-dir ~/models
# + Gemma-4-E4B Q8 (per HF availability)

# LAPTOP-3070: Worker
huggingface-cli download bartowski/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf --local-dir ~/models

# LAPTOP-3060: Explore
# Gemma-4-E4B Q8 (per HF availability)
```

### 2.3 Server launch scripts

**AX8 MAX — Orchestrator (:8080)**

```bash
#!/usr/bin/env bash
# File: ~/nexus-swarm/start-orchestrator.sh (on AX8 Max)

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

**DESKTOP — Worker (:8081) + Explore (:8082)**

```bash
#!/usr/bin/env bash
# File: ~/nexus-swarm/start-worker-main.sh (on desktop)

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
# File: ~/nexus-swarm/start-worker-explore.sh (on desktop)

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

**LAPTOP-3070 — Worker (:8084)** and **LAPTOP-3060 — Edge (:8086):**
Identical to v3.a §2.2 and §2.3.

---

## 2b — NPU Inference Servers (Orange Pi boards)

Identical to v3 §2b. OPi 5 Plus runs triage (:8088) + monitoring.
OPi 5B runs embeddings (:8087).

---

## 3 — LiteLLM Routing (on K1)

```yaml
# File: ~/nexus-swarm/litellm-config.yaml (on K1)

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

  # ═══ WORKER POOL (load-balanced: 5060 Ti + 3070) ═══════════
  - model_name: "nexus-worker"
    litellm_params:
      model: "openai/nexus-worker-desktop"
      api_base: "http://nexus-desktop:8081/v1"
      api_key: "sk-no-key"
      max_tokens: 2048
      timeout: 120
    model_info:
      description: "Qwen3-8B Q6_K on 5060 Ti — primary worker (fastest)"

  - model_name: "nexus-worker"
    litellm_params:
      model: "openai/nexus-worker-3070"
      api_base: "http://nexus-3070:8084/v1"
      api_key: "sk-no-key"
      max_tokens: 2048
      timeout: 120
    model_info:
      description: "Qwen3-8B Q4_K_M on 3070 — parallel worker"

  # ═══ EXPLORE (E4B on 5060 Ti + 3060, load-balanced) ════════
  - model_name: "nexus-explore"
    litellm_params:
      model: "openai/nexus-explore-desktop"
      api_base: "http://nexus-desktop:8082/v1"
      api_key: "sk-no-key"
      max_tokens: 1024
      timeout: 60
    model_info:
      description: "Gemma-4-E4B on 5060 Ti — fast explore"

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
      model: "openai/nexus-worker-desktop"
      api_base: "http://nexus-desktop:8081/v1"
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
lands on a different GPU.

---

## 4–10 — Remaining Sections

Agent loop, tools (ferry-first), system prompt, pipeline branches,
tool-call fixer, systemd units, health monitor, and end-to-end test
are identical to v3 (see the full v3 scaffold document).

The system prompt delegation addendum for v3.b:
```
Your inference runs on a Tesla P40 (24 GB). Workers run on separate
GPUs (5060 Ti + 3070, load-balanced). You have two parallel workers
and two explore agents. Delegation is free — fan out independent
subtasks to keep your context window clean.

1. Large tool output (>4000 chars) is auto-summarized by a worker.
2. Read-only search is delegated to explore agents (E4B on 5060 Ti + 3060).
3. You can dispatch two subtasks simultaneously — they land on different GPUs.
4. Never process raw scan output yourself. Send to a worker first.
```

---

## Execution Order

**Phase A — LAN + rust-nexus mesh (~30 min):**
1. §1.1 — Hostnames + /etc/hosts on all 7 nodes
2. §1.2 — Build rust-nexus, generate certs, distribute
3. §1.3 — Launch nexus-infra on K1
4. §1.4 — Launch nexus-agent on all 6 compute nodes
5. §1.5 — Verify mesh (6 agents, all endpoints UP)

**Phase B — GPU inference servers (~30 min per node):**
6. §2.1 — Build llama.cpp per architecture (sm_120, sm_61, sm_86)
7. §2.2 — Download models per node
8. §2.3 — Launch servers, verify /health

**Phase B2 — NPU inference servers (~20 min per board):**
9. §2b — OPi 5 Plus triage + monitoring, OPi 5B embeddings

**Phase C — Control plane (K1, ~30 min):**
10. §3 — LiteLLM config + launch + route verification
11. §4 — Agent loop, tools (ferry-first), system prompt
12. §6 — Gemma tool-call fixer

**Phase D — Production hardening (~20 min):**
13. §7 — Systemd units per node
14. §8 — End-to-end swarm test

**Phase E — Training (async):**
15. §5 — Pipeline branches for worker + explore models
