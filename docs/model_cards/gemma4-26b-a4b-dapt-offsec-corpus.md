---
license: apache-2.0
tags:
- dapt
- domain-adaptation
- offensive-security
- code-corpus
size_categories:
- 10K<n<100K
task_categories:
- text-generation
---

# gemma4-26b-a4b-dapt-offsec-corpus

Offensive security code corpus for **domain-adaptive continued pretraining
(DAPT)**. Plain text code documents extracted from offensive security GitHub
repositories, used to ground language models in security tooling patterns
before SFT.

Part of the Code-Trainer / RTPI pipeline
([GitHub](https://github.com/cmndcntrlcyber/code-trainer-pipeline)).

## Dataset summary

| Metric | Value |
|---|---|
| Total rows | 20,680 |
| Format | `{"text": "..."}` — plain text continuation |
| Domain | Offensive security (exploit frameworks, C2, recon, privesc, post-exploitation) |
| Shared with | Qwen DAPT pipeline (same corpus, different dataset name) |

## Format

Each row is a single code document:

```json
{
  "text": "#!/usr/bin/env python3\n# Nmap wrapper for automated service enumeration\nimport subprocess\n..."
}
```

No chat formatting is applied — this is raw text continuation data for
continued pretraining, not instruction tuning.

## Data sources

Offensive security GitHub repositories cloned and chunked by the Phase 1c
ingestion pipeline. Repositories are quality-filtered by the
`QualityScorer` (stars, activity, documentation, code quality, community
signals) and files are filtered to 10-1000 lines.

## Build pipeline

```bash
python -m src.phase3b_dapt.data.prepare_corpus \
    --config src/config/pipeline-gemma26b.yml
```

The corpus builder:
1. Scans `data/offensive-security/repositories/` for code files
2. Filters by line count (10-1000 lines) and file size
3. Chunks long files at natural boundaries
4. Outputs the dataset in HuggingFace `datasets` format

## Intended use

* **DAPT training:** used by
  [`gemma4-26b-a4b-dapt-offsec`](https://huggingface.co/cmndcntrlcyber/gemma4-26b-a4b-dapt-offsec)
  (Gemma pipeline) and the corresponding Qwen DAPT adapter for
  domain-adaptive continued pretraining.
* **Out of scope:** this dataset contains offensive security code from
  public repositories. It is intended for research and training in
  controlled environments, not for enabling unauthorized access.

## Limitations

* **Public code only.** All source material is from public GitHub
  repositories under permissive licenses.
* **No deduplication across repos.** Some code patterns (common exploit
  templates, standard C2 frameworks) may appear in multiple documents.
* **English-centric.** Comments and documentation strings are predominantly
  in English.

## Reproducibility

* **Code:** [github.com/cmndcntrlcyber/code-trainer-pipeline](https://github.com/cmndcntrlcyber/code-trainer-pipeline)
  (`src/phase3b_dapt/data/`)
* **Config:** `src/config/pipeline-gemma26b.yml` (`gemma_dapt` section)
