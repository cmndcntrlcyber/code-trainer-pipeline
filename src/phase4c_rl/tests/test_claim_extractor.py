"""Tests for FARCA Module 1: Claim Extractor."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.phase4c_rl.farca.claim_extractor import ClaimExtractor, AtomicClaim


def _mock_tokenizer():
    tok = MagicMock()
    tok.decode = lambda ids: "".join(chr(65 + (i % 26)) for i in ids)
    tok.pad_token_id = 0
    return tok


class TestReasoningPrefixExtraction:
    def test_extracts_text_before_tool_call(self):
        ext = ClaimExtractor()
        completion = "I should use Read to check the file.\n<tool_call>{}</tool_call>"
        result = ext._extract_reasoning_prefix(completion)
        assert "<tool_call>" not in result
        assert "Read" in result

    def test_full_completion_when_no_tool_call(self):
        ext = ClaimExtractor()
        completion = "This is just reasoning without any tool call."
        result = ext._extract_reasoning_prefix(completion)
        assert result == completion


class TestSentenceSegmentation:
    def test_splits_on_periods(self):
        ext = ClaimExtractor()
        text = "I need to read the file. Then I will edit it."
        sentences = ext._segment_sentences(text)
        assert len(sentences) == 2

    def test_splits_on_newlines(self):
        ext = ClaimExtractor()
        text = "First line of reasoning\nSecond line of reasoning"
        sentences = ext._segment_sentences(text)
        assert len(sentences) == 2

    def test_filters_short_fragments(self):
        ext = ClaimExtractor()
        text = "OK.\nI need to use the Read tool to examine this configuration file."
        sentences = ext._segment_sentences(text)
        assert all(len(s) >= 8 for s in sentences)


class TestVerifiability:
    def test_filler_rejected(self):
        ext = ClaimExtractor()
        assert not ext._is_verifiable("Let me think about this.")
        assert not ext._is_verifiable("Sure, I can help.")
        assert not ext._is_verifiable("OK, let's proceed.")

    def test_substantive_accepted(self):
        ext = ClaimExtractor()
        assert ext._is_verifiable("I should use the Read tool to examine the file.")
        assert ext._is_verifiable("The path parameter should be src/main.py.")

    def test_long_filler_accepted(self):
        ext = ClaimExtractor()
        assert ext._is_verifiable(
            "Let me use the Read tool to examine the configuration file at src/config.yaml"
        )


class TestClaimClassification:
    def test_tool_selection(self):
        ext = ClaimExtractor()
        assert ext._classify("I should use Read to check the file") == "tool_selection"
        assert ext._classify("The Bash command will list files") == "tool_selection"

    def test_argument_claim(self):
        ext = ClaimExtractor()
        assert ext._classify("The path should be src/main.py") == "argument_claim"
        assert ext._classify("The command to run is ls -la") == "argument_claim"

    def test_reasoning_chain(self):
        ext = ClaimExtractor()
        assert ext._classify("First we need to examine the structure") == "reasoning_chain"


class TestDecomposition:
    def test_splits_on_and(self):
        ext = ClaimExtractor()
        result = ext._decompose("I will use Read and then Edit the file")
        assert len(result) == 2

    def test_preserves_single_claim(self):
        ext = ClaimExtractor()
        result = ext._decompose("I should use the Read tool")
        assert len(result) == 1


class TestFullExtraction:
    def test_extracts_claims_from_reasoning(self):
        ext = ClaimExtractor()
        completion = (
            "I need to use the Read tool to examine src/config.yaml. "
            "The file contains important settings.\n"
            '<tool_call>{"name":"Read","arguments":{"path":"src/config.yaml"}}</tool_call>'
        )
        token_ids = list(range(len(completion)))
        tok = _mock_tokenizer()

        result = ext.extract(completion, token_ids, tok)
        assert len(result.claims) >= 1
        assert any(c.claim_type == "tool_selection" for c in result.claims)
        assert result.reasoning_text == completion[:completion.index("<tool_call>")]

    def test_no_claims_when_no_reasoning(self):
        ext = ClaimExtractor()
        completion = '<tool_call>{"name":"Read","arguments":{"path":"x"}}</tool_call>'
        result = ext.extract(completion, list(range(50)), _mock_tokenizer())
        assert len(result.claims) == 0

    def test_token_span_within_bounds(self):
        ext = ClaimExtractor()
        completion = "I should use Read to check the config file.\n<tool_call>{}</tool_call>"
        token_ids = list(range(len(completion)))
        tok = _mock_tokenizer()

        result = ext.extract(completion, token_ids, tok)
        for claim in result.claims:
            assert claim.token_span[0] >= 0
            assert claim.token_span[1] <= len(token_ids)
            assert claim.token_span[0] < claim.token_span[1]
