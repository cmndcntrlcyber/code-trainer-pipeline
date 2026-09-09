"""
phase4c_rl/scripts/precompute_farca_annotations.py

Offline pre-computation of FARCA claim annotations on ingested session data.
Runs the claim extraction and verification pipeline on all training examples
so that FARCA-GRPO training can load pre-computed claims instead of running
the extraction pipeline during training.

Usage:
    python -m src.phase4c_rl.scripts.precompute_farca_annotations \
        --input data/oco_converted/train.jsonl \
        --output data/farca_annotations/claims.jsonl

    # With specific config:
    python -m src.phase4c_rl.scripts.precompute_farca_annotations \
        --input data/oco_converted/train.jsonl \
        --output data/farca_annotations/claims.jsonl \
        --config src/config/pipeline-50.yml
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute FARCA claim annotations on training data"
    )
    parser.add_argument(
        "--input", required=True,
        help="Input JSONL file (e.g. data/oco_converted/train.jsonl)",
    )
    parser.add_argument(
        "--output", default="data/farca_annotations/claims.jsonl",
        help="Output JSONL with per-example claim annotations",
    )
    parser.add_argument(
        "--config", default=None,
        help="Pipeline config for FARCA hyperparams (reads rl_training.farca_grpo)",
    )
    parser.add_argument(
        "--max-examples", type=int, default=None,
        help="Process at most N examples (for testing)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from src.phase4_qwen_finetuning.hf_skills.nexus_tools import NEXUS_TOOLS_V10
    from src.phase4c_rl.farca.config import FARCAConfig
    from src.phase4c_rl.farca.claim_extractor import ClaimExtractor
    from src.phase4c_rl.farca.claim_verifier import ClaimVerifier
    from src.phase4c_rl.farca.counterfactual_attribution import CounterfactualAttributor
    from src.phase4c_rl.farca.pipeline import _schema_to_evidence_sentences

    # Load FARCA config from pipeline YAML if provided
    farca_params = {}
    if args.config:
        from src.config.settings import load_config
        config = load_config(args.config)
        farca_params = (
            config.get("rl_training", {}).get("farca_grpo", {})
        )

    farca_config = FARCAConfig.from_dict(farca_params)
    farca_config.tool_schema = NEXUS_TOOLS_V10

    tool_names = {
        e.get("function", {}).get("name", "")
        for e in NEXUS_TOOLS_V10
        if e.get("function", {}).get("name")
    }
    extractor = ClaimExtractor(tool_names=tool_names)
    verifier = ClaimVerifier(
        tool_schema=NEXUS_TOOLS_V10,
        nli_model_id=farca_config.nli_model_id,
        nli_weight=farca_config.nli_weight,
        rule_weight=farca_config.rule_weight,
        device=farca_config.device,
    )
    attributor = CounterfactualAttributor(
        verifier=verifier,
        sentence_encoder_id=farca_config.sentence_encoder_id,
        k_rel=farca_config.k_rel,
        mu=farca_config.mu,
        tau=farca_config.tau,
        device=farca_config.device,
    )
    evidence_sentences = _schema_to_evidence_sentences(NEXUS_TOOLS_V10)
    evidence_string = " ".join(evidence_sentences)

    # Process input
    examples = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    if args.max_examples:
        examples = examples[: args.max_examples]

    logger.info("Processing %d examples from %s", len(examples), input_path)

    stats = {
        "total_examples": len(examples),
        "examples_with_claims": 0,
        "total_claims": 0,
        "claims_by_type": {"tool_selection": 0, "argument_claim": 0, "reasoning_chain": 0},
        "mean_reliability": 0.0,
    }
    all_weights: list[float] = []

    with open(output_path, "w") as out:
        for idx, example in enumerate(examples):
            messages = example.get("messages", [])
            # Find assistant messages with tool calls (the completions)
            completions = []
            for msg in messages:
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if "<tool_call>" in content:
                        completions.append(content)

            example_claims = []
            for completion in completions:
                # Use a simple integer token-ID stand-in (no tokenizer needed for precompute)
                fake_ids = list(range(len(completion)))
                result = extractor.extract(completion, fake_ids, _FakeTokenizer())

                for claim in result.claims:
                    verification = verifier.verify(claim, evidence_string)
                    reliability = attributor.compute_reliability(
                        claim, verification, evidence_sentences
                    )

                    example_claims.append({
                        "text": claim.text,
                        "type": claim.claim_type,
                        "sentence": claim.source_sentence,
                        "factual_score": round(verification.factual_score, 4),
                        "reliability_weight": round(reliability.reliability_weight, 4),
                        "weighted_score": round(reliability.weighted_score, 4),
                        "method": verification.method,
                    })

                    stats["claims_by_type"][claim.claim_type] = (
                        stats["claims_by_type"].get(claim.claim_type, 0) + 1
                    )
                    all_weights.append(reliability.reliability_weight)

            if example_claims:
                stats["examples_with_claims"] += 1
            stats["total_claims"] += len(example_claims)

            record = {
                "source_file": example.get("source_file", f"example_{idx}"),
                "num_claims": len(example_claims),
                "claims": example_claims,
            }
            out.write(json.dumps(record) + "\n")

            if (idx + 1) % 50 == 0:
                logger.info(
                    "  processed %d/%d examples (%d claims so far)",
                    idx + 1, len(examples), stats["total_claims"],
                )

    if all_weights:
        stats["mean_reliability"] = round(sum(all_weights) / len(all_weights), 4)

    stats_path = output_path.parent / "annotation_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))

    logger.info("=" * 60)
    logger.info("FARCA Annotation Pre-computation Complete")
    logger.info("  Examples processed:    %d", stats["total_examples"])
    logger.info("  Examples with claims:  %d", stats["examples_with_claims"])
    logger.info("  Total claims:          %d", stats["total_claims"])
    logger.info("  By type:               %s", stats["claims_by_type"])
    logger.info("  Mean reliability:      %.4f", stats["mean_reliability"])
    logger.info("  Output:                %s", output_path)
    logger.info("  Stats:                 %s", stats_path)
    logger.info("=" * 60)


class _FakeTokenizer:
    """Minimal tokenizer stand-in for pre-computation (no model needed)."""
    pad_token_id = 0

    @staticmethod
    def decode(ids):
        return "".join(chr(32 + (i % 95)) for i in ids)


if __name__ == "__main__":
    main()
