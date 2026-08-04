"""
Comparative evaluator for abliterated model variants.

Runs custom metrics (refusal rate, KL divergence, perplexity) and
lm-eval-harness benchmarks across all abliterated variants + control,
then produces a ranked comparison.
"""
import gc
import json
import logging
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .metrics import (
    load_harmful_prompts,
    load_harmless_prompts,
    measure_kl_divergence,
    measure_perplexity,
    measure_refusal_rate,
)

logger = logging.getLogger(__name__)


class AbliterationEvaluator:
    def __init__(self, config: dict, work_dir: Path):
        self.config = config
        self.work_dir = work_dir
        self.results_dir = work_dir / "evaluation_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        eval_cfg = config.get("evaluation", {})
        self.harmful_dataset = eval_cfg.get(
            "harmful_dataset", "Undi95/orthogonal-activation-steering-TOXIC"
        )
        self.harmless_dataset = eval_cfg.get(
            "harmless_dataset", "tatsu-lab/alpaca"
        )
        self.n_eval_samples = eval_cfg.get("n_eval_samples", 200)
        self.benchmark_tasks = eval_cfg.get(
            "benchmark_tasks", ["gsm8k", "mmlu", "truthfulqa_mc2"]
        )
        self.num_fewshot = eval_cfg.get("num_fewshot", 0)
        self.batch_size = eval_cfg.get("batch_size", 16)

    def evaluate_variant(
        self,
        variant_name: str,
        model_dir: Path,
        original_model_dir: Path | None = None,
    ) -> dict:
        """
        Run all evaluations on a single model variant.

        Args:
            variant_name:       e.g. "obliteratus", "control"
            model_dir:          Path to the model to evaluate.
            original_model_dir: Path to the un-abliterated model (for KL div).
                                None for the control variant.
        """
        logger.info("Evaluating variant: %s from %s", variant_name, model_dir)
        results = {"variant": variant_name, "model_dir": str(model_dir)}

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        harmful_prompts = load_harmful_prompts(
            self.harmful_dataset, self.n_eval_samples
        )
        harmless_prompts = load_harmless_prompts(
            self.harmless_dataset, self.n_eval_samples
        )

        logger.info("%s: measuring refusal rate...", variant_name)
        results["refusal"] = measure_refusal_rate(
            model, tokenizer, harmful_prompts
        )

        logger.info("%s: measuring perplexity...", variant_name)
        results["perplexity"] = measure_perplexity(
            model, tokenizer, harmless_prompts[:50]
        )

        if original_model_dir is not None:
            logger.info("%s: measuring KL divergence...", variant_name)
            original_model = AutoModelForCausalLM.from_pretrained(
                str(original_model_dir),
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            original_model.eval()
            results["kl_divergence"] = measure_kl_divergence(
                original_model, model, tokenizer, harmless_prompts[:100]
            )
            del original_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("%s: running lm-eval benchmarks...", variant_name)
        results["benchmarks"] = self._run_lm_eval(variant_name, model_dir)

        out_path = self.results_dir / f"{variant_name}_results.json"
        out_path.write_text(json.dumps(results, indent=2, default=str))
        logger.info("%s: results saved to %s", variant_name, out_path)

        return results

    def _run_lm_eval(self, variant_name: str, model_dir: Path) -> dict:
        """Run lm-evaluation-harness benchmarks."""
        all_results = {}
        for task in self.benchmark_tasks:
            out_dir = self.results_dir / f"lm_eval_{variant_name}_{task}"
            out_dir.mkdir(exist_ok=True)

            cmd = [
                "lm_eval",
                "--model", "hf",
                "--model_args",
                f"pretrained={model_dir},dtype=bfloat16,trust_remote_code=True",
                "--tasks", task,
                "--num_fewshot", str(self.num_fewshot),
                "--batch_size", str(self.batch_size),
                "--device", "cuda",
                "--output_path", str(out_dir),
            ]
            logger.info("%s/%s: %s", variant_name, task, " ".join(cmd))
            try:
                subprocess.run(cmd, check=True, timeout=3600)
                result_files = sorted(out_dir.rglob("results_*.json"))
                if result_files:
                    raw = json.loads(result_files[-1].read_text())
                    all_results[task] = raw.get("results", {}).get(task, {})
                else:
                    all_results[task] = {"error": "no results file found"}
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                logger.error("%s/%s failed: %s", variant_name, task, e)
                all_results[task] = {"error": str(e)}

        return all_results

    def evaluate_all(
        self,
        variants: dict[str, Path],
        original_model_dir: Path,
    ) -> dict:
        """
        Evaluate all variants and produce a comparative summary.

        Args:
            variants: {"obliteratus": Path(...), "nousresearch": Path(...), ...}
            original_model_dir: Path to un-abliterated merged model.
        """
        all_results = {}

        all_results["control"] = self.evaluate_variant(
            "control", original_model_dir
        )

        for name, model_dir in variants.items():
            all_results[name] = self.evaluate_variant(
                name, model_dir, original_model_dir
            )

        summary = self._build_summary(all_results)
        summary_path = self.results_dir / "comparative_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str))
        logger.info("Comparative summary saved to %s", summary_path)

        return summary

    def _build_summary(self, all_results: dict) -> dict:
        """Build a ranked comparison table from all variant results."""
        rows = []
        for name, result in all_results.items():
            row = {"variant": name}
            if "refusal" in result:
                row["refusal_rate"] = result["refusal"]["refusal_rate"]
            if "perplexity" in result:
                row["perplexity"] = result["perplexity"]["perplexity"]
            if "kl_divergence" in result:
                row["kl_divergence"] = result["kl_divergence"]["kl_divergence_mean"]
            if "benchmarks" in result:
                for task, task_result in result["benchmarks"].items():
                    if isinstance(task_result, dict) and "acc" in task_result:
                        row[f"{task}_acc"] = task_result["acc"]
                    elif isinstance(task_result, dict) and "acc,none" in task_result:
                        row[f"{task}_acc"] = task_result["acc,none"]
            rows.append(row)

        rows.sort(key=lambda r: (r.get("refusal_rate", 1.0), r.get("kl_divergence", 999)))

        return {
            "ranking": rows,
            "best_by_refusal": min(rows, key=lambda r: r.get("refusal_rate", 1.0))["variant"],
            "best_by_kl": min(
                [r for r in rows if "kl_divergence" in r],
                key=lambda r: r["kl_divergence"],
                default={"variant": "unknown"},
            )["variant"],
        }
