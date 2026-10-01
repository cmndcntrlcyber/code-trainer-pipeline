# NEXUS DISTRIBUTED SWARM — Scaffold v3.a (without P40)

> **Purpose:** Six-node swarm using only hardware currently in hand.
> No P40, no GEEKOM AX8 Max, no Razer Core X V2.
> Three NVIDIA GPUs, two RK3588 NPUs, one control plane.

---

## 0 — Fleet Inventory

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      NEXUS DISTRIBUTED SWARM v3.a                          │
│                         (P40 excluded)                                     │
│                                                                            │
│  ┌─ DESKTOP ──────────────────────┐   ┌─ LAPTOP-3070 ─────────────────┐   │
│  │  GPU: RTX 5060 Ti (16 GB)      │   │  GPU: RTX 3070 Laptop (8 GB)  │   │
│  │  Role: ORCHESTRATOR + WORKER   │   │  Role: PARALLEL WORKER         │   │
│  │  Driver: 580.173.02 / CUDA 13  │   │  Driver: 560.35.03            │   │
│  │                                │   │                                │   │
│  │  Models:                       │   │  Model:                        │   │
│  │  • Qwen2.5-Coder-14B Q4_K_M   │   │  • Qwen3-8B Q4_K_M            │   │
│  │    ORCHESTRATOR     :8080      │   │    (~5.5 GB + ~2 GB KV)       │   │
│  │  • Qwen3-8B Q6_K               │   │  Port: :8084                  │   │
│  │    WORKER           :8081      │   │                                │   │
│  │  • Gemma-4-E4B Q8              │   └──────────┬─────────────────────┘   │
│  │    EXPLORE          :8082      │              │                         │
│  └──────────────┬─────────────────┘              │                         │
│                 │                                │                         │
│  ┌─ LAPTOP-3060 ──────────────────┐              │                         │
│  │  GPU: RTX 3060 Laptop (6 GB)   │              │                         │
│  │  Role: EDGE WORKER              │              │                         │
│  │  Driver: 580.173.02            │              │                         │
│  │                                │              │                         │
│  │  Model:                        │              │                         │
│  │  • Gemma-4-E4B Q8 (classify)   │              │                         │
│  │  Port: :8086                   │              │                         │
│  └──────────────┬─────────────────┘              │                         │
│                 │                                │                         │
│  ┌─ OPi 5 Plus (32GB) ───────────┐   ┌─ OPi 5B (16GB) ───────────────┐   │
│  │  SoC: RK3588 · NPU 6 TOPS     │   │  SoC: RK3588S · NPU 6 TOPS    │   │
│  │  Role: TRIAGE GATEWAY          │   │  Role: EDGE EMBEDDINGS         │   │
│  │  Dual 2.5GbE, M.2 NVMe        │   │  WiFi 6, BT 5.0               │   │
│  │                                │   │  128GB eMMC                    │   │
│  │  Models (RKLLM, NPU):         │   │                                │   │
│  │  • Qwen3.5-0.8B w8a8           │   │  Models (llama.cpp ARM):       │   │
│  │    (request classification)     │   │  • nomic-embed Q8              │   │
│  │  Services:                     │   │  Port: :8087                   │   │
│  │  • Triage API       :8088      │   │                                │   │
│  │  • Prometheus        :9090      │   │                                │   │
│  │  • Grafana           :3000      │   │                                │   │
│  └──────────────┬─────────────────┘   └──────────┬─────────────────────┘   │
│                 │                                │                         │
│             ┌───▼────────────────────────────────▼────┐                    │
│             │   ACEMAGIC K1 (no GPU, no NPU)          │                    │
│             │   Ryzen 7 7730U · 32 GB DDR4            │                    │
│             │   Role: CONTROL PLANE                   │                    │
│             │                                         │                    │
│             │   Services:                             │                    │
│             │   • rust-nexus infra (A2A + ferry)      │                    │
│             │   • LiteLLM router          :4000       │                    │
│             │   • nexus-harness agent loop             │                    │
│             └─────────────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key design change: 5060 Ti pulls double duty

Without the P40's 24 GB, the orchestrator model must fit on the 5060 Ti (16 GB).
The Gemma 26B-A4B at IQ4_XS (~13 GB) technically fits but leaves only ~3 GB for
KV cache — too tight for agent loops that accumulate deep context.

**The Qwen2.5-Coder-14B (dense, 14B params) at Q4_K_M (~8.5 GB) is the right
orchestrator for this config.** Advantages over the squeezed Gemma MoE:
- 8.5 GB weights + 6 GB KV headroom at `--ctx-size 8192`
- Dense architecture means every parameter fires every token (no MoE routing noise at low quant)
- The Qwen line is more reliable for tool-calling in llama.cpp (no Gemma 4 format issues)
- Your RTPI pipeline already produces a Qwen V9 fine-tune with the Nexus persona

The 5060 Ti also co-hosts a Qwen3-8B worker (:8081) and Gemma-4-E4B explore (:8082),
using the remaining VRAM. Total VRAM budget:

| Instance | Weights | KV + overhead |
|----------|---------|---------------|
| Qwen2.5-14B Q4_K_M (orchestrator) | ~8.5 GB | ~4.5 GB |
| Qwen3-8B Q6_K (worker) | ~6.6 GB | — |
| Gemma-4-E4B Q8 (explore) | ~1.8 GB | — |
| **Total** | **~16.9 GB** | **OVERCOMMIT** |

**This does not fit simultaneously.** The 5060 Ti must time-share or serialize.

**Two viable layouts:**

**Layout A — Dedicated orchestrator (recommended):**
Run ONLY the Qwen2.5-14B orchestrator on the 5060 Ti. Move the worker to
the 3070 (already there) and the explore to the 3060. The orchestrator gets
the full 16 GB: ~8.5 GB weights + ~6 GB KV at ctx-size 12288.

```
5060 Ti (16 GB):  Qwen2.5-14B Q4_K_M  ORCHESTRATOR   :8080
3070    (8 GB):   Qwen3-8B Q4_K_M     WORKER          :8084
3060    (6 GB):   Gemma-4-E4B Q8      EXPLORE/EDGE    :8086
OPi 5 Plus:       Qwen3.5-0.8B NPU   TRIAGE          :8088
OPi 5B:           nomic-embed ARM     EMBEDDINGS      :8087
K1:               —                   CONTROL PLANE   :4000 :9100
```

**Layout B — Shared orchestrator + worker (advanced):**
Run the orchestrator and a SMALL worker on the 5060 Ti using llama-server's
multi-model support or two separate instances with careful VRAM pinning.
Only viable if you drop the orchestrator to IQ4_XS (~7 GB) and the worker
to Qwen3-8B IQ4_XS (~4.5 GB). Tight but possible.

**This scaffold uses Layout A.** Layout B is documented for future reference.

### Compute budget

| Node | Accelerator | Memory | Weights | KV headroom | Power |
|------|-------------|--------|---------|-------------|-------|
| Desktop | RTX 5060 Ti | 16 GB VRAM | ~8.5 GB (Qwen2.5-14B Q4_K_M) | ~6 GB | ~180W |
| Laptop-3070 | RTX 3070 | 8 GB VRAM | ~5.5 GB (Qwen3-8B Q4_K_M) | ~1.8 GB | ~80W |
| Laptop-3060 | RTX 3060 | 6 GB VRAM | ~3 GB (E4B Q8) | ~1.5 GB | ~80W |
| OPi 5 Plus | RK3588 NPU | 32 GB sys | ~1.5 GB (Qwen3.5-0.8B w8a8) | ~30 GB | ~15W |
| OPi 5B | RK3588S CPU | 16 GB sys | ~0.5 GB (embed) | ~15 GB | ~12W |
| K1 | None | 32 GB DDR4 | — | — | ~45W |

### Hostnames

| Machine | Hostname | LAN IP (example) | Purpose |
|---------|----------|-------------------|---------|
| Desktop | `nexus-desktop` | 192.168.1.10 | Orchestrator |
| 3070 laptop | `nexus-3070` | 192.168.1.12 | Worker |
| 3060 laptop | `nexus-3060` | 192.168.1.13 | Edge explore |
| OPi 5 Plus | `nexus-opi5p` | 192.168.1.14 | Triage + monitoring |
| OPi 5B | `nexus-opi5b` | 192.168.1.15 | Embeddings |
| ACEMAGIC K1 | `nexus-control` | 192.168.1.20 | Control plane |

---

## 1 — Network Mesh (rust-nexus on LAN)

Identical to v3 §1 except: 6 nodes (no nexus-p40), 6 sets of mTLS certs.

```bash
# /etc/hosts on every node:
cat << 'EOF' | sudo tee -a /etc/hosts
192.168.1.10  nexus-desktop
192.168.1.12  nexus-3070
192.168.1.13  nexus-3060
192.168.1.14  nexus-opi5p
192.168.1.15  nexus-opi5b
192.168.1.20  nexus-control
EOF

# mTLS cert generation (on K1):
./scripts/generate-certs.sh --nodes nexus-desktop,nexus-3070,nexus-3060,nexus-opi5p,nexus-opi5b,nexus-control
```

All other §1 steps (rust-nexus build, infra launch, agent deploy, mesh verify)
are identical to v3 — just remove the `nexus-p40` entries throughout.

---

## 2 — Inference Servers

### 2.1 DESKTOP — Dedicated Orchestrator (:8080)

The 5060 Ti runs ONLY the orchestrator. Full 16 GB for one model = maximum
KV headroom for deep agent sessions.

```bash
#!/usr/bin/env bash
# File: ~/nexus-swarm/start-orchestrator.sh (on desktop)

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

Key differences from v3:
- `--ctx-size 12288` (up from 8192): the 5060 Ti at 16 GB with ~8.5 GB model
  leaves ~6 GB for KV — enough for 12K context with q8_0 KV quantization.
- `--flash-attn`: sm_120 (Blackwell) supports flash-attn natively.
- No parallel workers on this GPU — the orchestrator gets all the VRAM.
- `--slot-save-path`: Prefix caching is critical here since the Nexus system
  prompt loads once and stays cached across turns.

### 2.2 LAPTOP-3070 — Worker (:8084)

```bash
#!/usr/bin/env bash
# File: ~/nexus-swarm/start-server.sh (on laptop-3070)

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
  2>&1 | tee /tmp/llama-server.log
```

### 2.3 LAPTOP-3060 — Edge Explore (:8086)

```bash
#!/usr/bin/env bash
# File: ~/nexus-swarm/start-server.sh (on laptop-3060)

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
  2>&1 | tee /tmp/llama-server.log
```

### 2.4 NPU nodes — identical to v3 §2b

OPi 5 Plus triage (:8088) and OPi 5B embeddings (:8087) are unchanged.
See v3 §2b.1 and §2b.2.

---

## 3 — LiteLLM Routing (on K1)

```yaml
# File: ~/nexus-swarm/litellm-config.yaml (on K1)

model_list:
  # ═══ ORCHESTRATOR (5060 Ti — single instance) ══════════════
  - model_name: "nexus-orchestrator"
    litellm_params:
      model: "openai/nexus-orchestrator"
      api_base: "http://nexus-desktop:8080/v1"
      api_key: "sk-no-key"
      max_tokens: 4096
      timeout: 300
    model_info:
      description: "Qwen2.5-Coder-14B Nexus persona on 5060 Ti — reasoner, planner"

  # ═══ WORKER (3070 only — single worker node without P40) ═══
  - model_name: "nexus-worker"
    litellm_params:
      model: "openai/nexus-worker-3070"
      api_base: "http://nexus-3070:8084/v1"
      api_key: "sk-no-key"
      max_tokens: 2048
      timeout: 120
    model_info:
      description: "Qwen3-8B Q4_K_M on 3070 — tool-call, summarize, parse"

  # ═══ EXPLORE (3060 — read-only recon) ═══════════════════════
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
      api_base: "http://nexus-desktop:8080/v1"
      api_key: "sk-no-key"

  - model_name: "haiku"
    litellm_params:
      model: "openai/nexus-worker-3070"
      api_base: "http://nexus-3070:8084/v1"
      api_key: "sk-no-key"

router_settings:
  routing_strategy: "simple-shuffle"  # Single instance per model, no balancing needed
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
The 3070 is the only worker. When the P40 arrives (upgrade to v3.b), the
orchestrator moves to the P40 and the 5060 Ti rejoins the worker pool,
re-enabling `least-busy` across two worker instances.

---

## 4–10 — Remaining Sections

All other sections (agent loop, tools, system prompt, pipeline branches,
tool-call fixer, systemd units, health monitor, end-to-end test) are
identical to v3 with these adjustments:

1. **Agent loop `_call_model`**: `AgentRole.ORCHESTRATOR` maps to
   `nexus-orchestrator` which now routes to the 5060 Ti, not the P40.

2. **System prompt delegation addendum**: Update to reflect one worker
   (3070) instead of a pool:
   ```
   Your inference runs on an RTX 5060 Ti (16 GB). You have one
   parallel worker (3070) and one explore agent (3060). Delegation
   is still free but sequential — fan-out is limited to one worker
   call at a time unless the explore agent handles the second.
   ```

3. **Health monitor**: Remove `nexus-p40:8080` from the endpoint list.
   The desktop's :8080 is the orchestrator endpoint.

4. **File manifest**: Desktop has `start-orchestrator.sh` (not three
   separate worker scripts). No `nexus-p40` node entry.

---

## Upgrade Path: v3.a → v3.b

When the GEEKOM AX8 Max + P40 arrive:

1. Set up the AX8 Max with the Razer Core X V2 + P40
2. Build llama.cpp on the AX8 Max (sm_61)
3. Deploy Gemma 26B-A4B Q4_K_M to the P40 as the new orchestrator (:8080)
4. Reconfigure the 5060 Ti: stop the 14B orchestrator, start Qwen3-8B Q6_K
   as a worker (:8081) + Gemma-4-E4B explore (:8082)
5. Update LiteLLM: point `nexus-orchestrator` to `nexus-p40:8080`,
   add `nexus-desktop:8081` as a second `nexus-worker` entry,
   switch `routing_strategy` to `least-busy`
6. The fleet is now v3.b — two workers load-balanced, MoE orchestrator
   on 24 GB, and the 5060 Ti's full 16 GB available for workers
