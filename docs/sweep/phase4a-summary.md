# Phase 4A — Validation Sweep Summary

**Adapter base:** `cmndcntrlcyber/qwen14b-code-trainer-{conservative,standard,aggressive}`

## Ranking by eval_loss

| Rank | Config | LoRA r/α | LR | bs × accum | Stage | eval_loss | Adapter |
|---:|---|---|---|---|---|---|---|
| 1 | `aggressive` | 64/128 | 3e-04 | 4 × 4 | ERROR | 0.4724 | [link](https://hf.co/cmndcntrlcyber/qwen14b-code-trainer-aggressive) |
| 2 | `standard` | 32/64 | 2e-04 | 2 × 8 | ERROR | 0.4798 | [link](https://hf.co/cmndcntrlcyber/qwen14b-code-trainer-standard) |
| 3 | `conservative` | 16/32 | 1e-04 | 1 × 16 | ERROR | 0.4819 | [link](https://hf.co/cmndcntrlcyber/qwen14b-code-trainer-conservative) |

## Best config: `aggressive` (eval_loss = 0.4724)

Recommended next step — Phase 4B full training (3 epochs) on the best config:

```bash
python -m src.phase4_qwen_finetuning.scripts.launch_full_training \
  --config src/config/config.yaml \
  --best-config aggressive --wait
```

## Raw results (per config)

### `aggressive` — job `69f7659298a8d679adfb8b8e`

```json
{
  "config": {
    "name": "aggressive",
    "lora_r": 64,
    "lora_alpha": 128,
    "learning_rate": 0.0003,
    "batch_size": 4,
    "gradient_accumulation": 4,
    "effective_batch": 16
  },
  "model_id": "Qwen/Qwen2.5-Coder-14B-Instruct",
  "dataset": "cmndcntrlcyber/code-trainer-offsec-dataset@main",
  "num_epochs": 1,
  "eval_loss": 0.4724135994911194,
  "eval_runtime": 677.8803
}
```

### `standard` — job `69f746449d85bec4d76f140e`

```json
{
  "config": {
    "name": "standard",
    "lora_r": 32,
    "lora_alpha": 64,
    "learning_rate": 0.0002,
    "batch_size": 2,
    "gradient_accumulation": 8,
    "effective_batch": 16
  },
  "model_id": "Qwen/Qwen2.5-Coder-14B-Instruct",
  "dataset": "cmndcntrlcyber/code-trainer-offsec-dataset@main",
  "num_epochs": 1,
  "eval_loss": 0.4798138439655304,
  "eval_runtime": 648.4817
}
```

### `conservative` — job `69f7658e98a8d679adfb8b8c`

```json
{
  "config": {
    "name": "conservative",
    "lora_r": 16,
    "lora_alpha": 32,
    "learning_rate": 0.00015,
    "batch_size": 1,
    "gradient_accumulation": 16,
    "effective_batch": 16
  },
  "model_id": "Qwen/Qwen2.5-Coder-14B-Instruct",
  "dataset": "cmndcntrlcyber/code-trainer-offsec-dataset@main",
  "num_epochs": 1,
  "eval_loss": 0.48192790150642395,
  "eval_runtime": 653.418
}
```
