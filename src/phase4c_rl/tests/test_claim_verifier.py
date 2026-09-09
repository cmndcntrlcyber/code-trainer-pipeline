"""Tests for FARCA Module 2: Claim Verifier."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.phase4c_rl.farca.claim_extractor import AtomicClaim
from src.phase4c_rl.farca.claim_verifier import ClaimVerifier

SAMPLE_SCHEMA = [
    {"type": "function", "function": {
        "name": "Read",
        "description": "Read the contents of a file at the given path.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path to read."}
        }, "required": ["path"]}
    }},
    {"type": "function", "function": {
        "name": "Bash",
        "description": "Run a shell command and return its output.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "Shell command."},
            "timeout_ms": {"type": "integer", "description": "Timeout."}
        }, "required": ["command"]}
    }},
]

EVIDENCE = "The Read tool reads file contents. The Bash tool runs shell commands."


def _claim(text: str, ctype: str = "tool_selection") -> AtomicClaim:
    return AtomicClaim(
        text=text, source_sentence=text, sentence_index=0,
        claim_index=0, token_span=(0, 10), claim_type=ctype,
    )


class TestToolSelectionVerification:
    def test_valid_tool_gets_high_score(self):
        verifier = ClaimVerifier(SAMPLE_SCHEMA, device="cpu")
        result = verifier.verify(_claim("I should use Read to check the file"), EVIDENCE)
        assert result.entailment_score == 1.0
        assert result.factual_score == 1.0
        assert result.method == "rule"

    def test_invalid_tool_gets_low_score(self):
        verifier = ClaimVerifier(SAMPLE_SCHEMA, device="cpu")
        result = verifier.verify(_claim("I should use FakeToolXYZ to scan"), EVIDENCE)
        assert result.entailment_score == 0.0
        assert result.factual_score == -1.0

    def test_multiple_valid_tools(self):
        verifier = ClaimVerifier(SAMPLE_SCHEMA, device="cpu")
        result = verifier.verify(
            _claim("I need Read and Bash to complete this task"), EVIDENCE
        )
        assert result.entailment_score == 1.0


class TestArgumentVerification:
    def test_valid_argument(self):
        verifier = ClaimVerifier(SAMPLE_SCHEMA, device="cpu")
        claim = _claim("The Read tool needs a path argument", "argument_claim")
        result = verifier.verify(claim, EVIDENCE)
        assert result.entailment_score == 1.0
        assert result.method == "rule"

    def test_argument_with_tool_reference(self):
        verifier = ClaimVerifier(SAMPLE_SCHEMA, device="cpu")
        claim = _claim("The Bash command parameter is required", "argument_claim")
        result = verifier.verify(claim, EVIDENCE)
        assert result.entailment_score == 1.0


class TestReasoningVerification:
    def test_reasoning_uses_nli(self):
        verifier = ClaimVerifier(SAMPLE_SCHEMA, device="cpu")
        claim = _claim(
            "First we need to examine the directory structure",
            "reasoning_chain",
        )
        result = verifier.verify(claim, EVIDENCE)
        assert result.method == "nli"
        assert 0.0 <= result.entailment_score <= 1.0


class TestBatchVerification:
    def test_batch_returns_correct_count(self):
        verifier = ClaimVerifier(SAMPLE_SCHEMA, device="cpu")
        claims = [
            _claim("Use Read to check the file"),
            _claim("Use Bash to run the command"),
            _claim("This is a reasoning step", "reasoning_chain"),
        ]
        results = verifier.verify_batch(claims, EVIDENCE)
        assert len(results) == 3
