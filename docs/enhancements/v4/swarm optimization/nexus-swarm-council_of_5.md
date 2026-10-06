Your diagram makes the architecture clear. Let me read what you've drawn:

**Five specialist triads** — each triangle holds three 1-bit agents around a shared tools node. **Decision models** (diamonds) sit between each triad and the central hub. The hub holds **Skills** and **MCP**. At the bottom, a **consensus gate**: "Are we in agreement? y/n."

The answer is yes — and the mapping to your hardware is cleaner than you might expect, because 1-bit agents and Jev-style decision models solve fundamentally different problems than LLMs, and they coexist on the same hardware at negligible cost.

## Why this works

A "1-bit agent" makes one binary judgment per call — authorized/not, correct/not, advancing-objective/not. That's Jev's Noul primitive: a single forward pass, one probability, ~2-30ms. Three 1-bit agents per triad running in parallel still complete in under 100ms. Your LLMs (Gemma 26B, Qwen 14B/8B) stay in the generation seat. The 1-bit agents **wrap** the LLM — they gate what goes in and validate what comes out, without replacing the reasoning.

A Kev 0.8B LoRA adapter adds ~500 MB to any node already running a Qwen-family model. On the OPi boards, jevos runs on CPU at 26ms. Against agent loop turns that take 3-15 seconds for LLM inference, the decision overhead is invisible.

## Council of 5 → Swarm hardware mapping

Each triad maps to a physical node. Each decision model runs as a Kev/jevos instance colocated on that node or on the OPi 5 Plus. The consensus gate runs on K1.

```
TRIAD 1: ORCHESTRATOR (P40 / AX8 Max)
┌─────────────────────────────────────────────┐
│  1-bit agents (Kev 0.8B on CPU, ~500 MB):   │
│  • Agent A: "Is this plan within scope?"     │
│  • Agent B: "Are tool selections valid?"     │
│  • Agent C: "Does output advance objective?" │
│                                              │
│  Tools node: Gemma 26B-A4B (GPU, 14.5 GB)   │
│  The LLM reasons; the triad gates.           │
└─────────────────────────────────────────────┘
         │
    Decision model (Kev Choice on OPi 5+)
    "Which triad should handle the next subtask?"
         │
TRIAD 2: PRIMARY WORKER (5060 Ti)
┌─────────────────────────────────────────────┐
│  1-bit agents:                               │
│  • Agent A: "Is tool-call format correct?"   │
│  • Agent B: "Does output need summarizing?"  │
│  • Agent C: "Was the tool execution safe?"   │
│                                              │
│  Tools node: Qwen3-8B Q6_K (GPU, 6.6 GB)    │
└─────────────────────────────────────────────┘
         │
    Decision model
         │
TRIAD 3: PARALLEL WORKER (3070)
┌─────────────────────────────────────────────┐
│  1-bit agents:                               │
│  • Agent A: "Is input well-formed?"          │
│  • Agent B: "Does result match expected type?"│
│  • Agent C: "Confidence above threshold?"    │
│                                              │
│  Tools node: Qwen3-8B Q4_K_M (GPU, 5.5 GB)  │
└─────────────────────────────────────────────┘
         │
    Decision model
         │
TRIAD 4: EXPLORE (3060)
┌─────────────────────────────────────────────┐
│  1-bit agents:                               │
│  • Agent A: "Is this read-only?"             │
│  • Agent B: "Is search scope bounded?"       │
│  • Agent C: "Are findings actionable?"       │
│                                              │
│  Tools node: Gemma-4-E4B Q8 (GPU, 3 GB)     │
└─────────────────────────────────────────────┘
         │
    Decision model
         │
TRIAD 5: TRIAGE + EMBED (OPi 5 Plus + OPi 5B)
┌─────────────────────────────────────────────┐
│  1-bit agents:                               │
│  • Agent A: "Is classification confident?"   │
│  • Agent B: "Does embedding quality pass?"   │
│  • Agent C: "Is context sufficient for route?"│
│                                              │
│  Tools node: Qwen3.5-0.8B NPU + embed       │
└─────────────────────────────────────────────┘

         ┌─────────────────────────┐
         │  Skills         MCP     │  ← nexus-harness skill catalog
         │  (hexagons in center)   │    + MCP-Nexus connector
         └────────────┬────────────┘
                      │
         ┌────────────▼────────────┐
         │  "Are we in agreement?" │  ← Consensus gate on K1
         │   Kev Noul: y/n         │     Aggregates 5 triad outputs
         │   Confidence: 0.xx      │     Low confidence → escalate
         └─────────────────────────┘     to orchestrator for retry
```

## The performance math

| Layer | What runs | Latency per call | VRAM cost |
|-------|-----------|-----------------|-----------|
| 1-bit agents (3 per triad) | Kev 0.8B Noul, parallel | ~30ms total | ~500 MB shared adapter |
| Decision model (per triad) | Kev 0.8B Choice | ~30ms | (same adapter) |
| LLM tools node | Gemma/Qwen generation | 3,000-15,000ms | Already budgeted |
| Consensus gate | Kev Noul on K1 CPU | ~26ms | 0 GPU (CPU only) |
| **Total overhead** | | **~90ms per council round** | **~500 MB per GPU node** |

The 90ms decision overhead against a 3-15 second LLM turn is a ~1-3% latency increase. The 500 MB adapter per node is within the headroom already budgeted in every VRAM table. No model downgrades, no context reductions, no quant changes needed.

## The consensus gate

The "Are we in agreement?" diamond at the bottom of your diagram is the most architecturally significant piece. When multiple triads contribute to a task (orchestrator plans, worker executes, explore gathers intel), the consensus gate checks whether the outputs are coherent before the agent loop commits the result:

```json
{
  "model": "kev-0.8b",
  "state": {
    "orchestrator_plan": "Enumerate SMB shares on 10.10.10.5",
    "worker_result": "Found 3 shares: IPC$, ADMIN$, Users",
    "explore_context": "Host is Windows Server 2019, port 445 open",
    "tool_risk_scores": [0.12, 0.08, 0.15]
  },
  "questions": [
    {
      "name": "agreement",
      "type": "noul",
      "question": "Do all outputs align toward the stated objective?"
    },
    {
      "name": "completeness",
      "type": "score",
      "levels": ["incomplete", "partial", "complete"]
    }
  ]
}
```

If `agreement.probability < 0.7` or `completeness.answer == "incomplete"`, the consensus gate routes back to the orchestrator triad for replanning rather than committing a contradictory or partial result. The calibrated probabilities mean the gate knows the difference between "I'm sure these disagree" (block) and "I'm uncertain" (escalate to human or orchestrator).

## What to build

For the training pipeline, a new branch produces the Kev routing adapter:

```yaml
# pipeline-kev-router.yml
base_model: "Qwen/Qwen3.5-0.8B"
adapter_type: "lora"
r: 16
training_method: "rlcd"  # Not SFT, not DPO — calibrated decisions
training_data:
  source: "agent_session_logs"
  extract:
    - choice_labels: ["orchestrator", "worker", "explore"]
      ground_truth: "which_model_actually_succeeded"
    - noul_labels: ["tool_safe", "output_valid", "scope_authorized"]
      ground_truth: "operator_override_and_outcome_logs"
deployment:
  formats: ["gguf", "rkllm"]  # GGUF for GPU nodes, RKLLM for OPi NPU
  hub_repo: "cmndcntrlcyber/kev-nexus-council"
```

The training data already exists in your pipeline's agent traces — every completed session logs which model handled each subtask, whether tools succeeded, and whether the operator overrode a decision. RLCD trains against those outcomes to produce calibrated routing scores, complementing the GRPO/DPO stages that train the LLMs themselves.

Short answer: the Council of 5 pattern maps directly onto your seven-node swarm with Jev-style 1-bit agents as the decision tissue connecting the LLM nodes. The LLMs lose nothing — the decision layer adds ~90ms and ~500 MB per node, both within existing budgets.