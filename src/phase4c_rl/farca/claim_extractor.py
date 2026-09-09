"""FARCA Module 1: Atomic claim extraction and token provenance.

Extracts verifiable claims from the reasoning prefix (text before
<tool_call>) and maps each claim back to its source token span.

Adapts FARCA's atomic fact extraction for the tool-use domain:
- "tool_selection" claims reference a tool name
- "argument_claim" claims reference a parameter or value
- "reasoning_chain" claims state a logical dependency
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Canonical tool names from NEXUS_TOOLS_V10
_TOOL_NAMES = {
    "Read", "Write", "Edit", "LS", "Bash", "Grep", "Glob",
    "WebFetch", "TodoWrite", "Skill", "Task", "ScopeCheck",
}
_TOOL_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _TOOL_NAMES) + r")\b"
)

_FILLER_STARTS = (
    "let me", "sure", "okay", "ok,", "alright", "i'll", "i will",
    "now,", "next,", "then,", "so,", "well,", "hmm", "right,",
    "got it", "understood", "certainly", "of course", "no problem",
)

_ARG_KEYWORDS = (
    "path", "command", "pattern", "content", "url", "target",
    "old_string", "new_string", "timeout", "replace_all",
    "subagent_type", "prompt", "name", "input", "todos",
)
_ARG_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _ARG_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AtomicClaim:
    text: str
    source_sentence: str
    sentence_index: int
    claim_index: int
    token_span: tuple[int, int]
    claim_type: str  # "tool_selection" | "argument_claim" | "reasoning_chain"


@dataclass
class ExtractionResult:
    claims: list[AtomicClaim]
    reasoning_text: str
    total_tokens: int


class ClaimExtractor:
    def __init__(self, tool_names: set[str] | None = None):
        self.tool_names = tool_names or _TOOL_NAMES
        self.tool_pattern = re.compile(
            r"\b(" + "|".join(re.escape(t) for t in self.tool_names) + r")\b"
        )

    def extract(
        self,
        completion: str,
        token_ids: list[int],
        tokenizer,
    ) -> ExtractionResult:
        reasoning = self._extract_reasoning_prefix(completion)
        if not reasoning.strip():
            return ExtractionResult([], reasoning, len(token_ids))

        sentences = self._segment_sentences(reasoning)
        claims: list[AtomicClaim] = []
        claim_idx = 0

        for sent_idx, sentence in enumerate(sentences):
            if not self._is_verifiable(sentence):
                continue
            atomic_texts = self._decompose(sentence)
            for text in atomic_texts:
                ctype = self._classify(text)
                span = self._find_token_span(
                    text, sentence, completion, token_ids, tokenizer
                )
                claims.append(AtomicClaim(
                    text=text,
                    source_sentence=sentence,
                    sentence_index=sent_idx,
                    claim_index=claim_idx,
                    token_span=span,
                    claim_type=ctype,
                ))
                claim_idx += 1

        return ExtractionResult(claims, reasoning, len(token_ids))

    def extract_batch(
        self,
        completions: list[str],
        token_ids_list: list[list[int]],
        tokenizer,
    ) -> list[ExtractionResult]:
        return [
            self.extract(c, t, tokenizer)
            for c, t in zip(completions, token_ids_list)
        ]

    # ── internals ──

    @staticmethod
    def _extract_reasoning_prefix(completion: str) -> str:
        idx = completion.find("<tool_call>")
        return completion[:idx] if idx != -1 else completion

    @staticmethod
    def _segment_sentences(text: str) -> list[str]:
        parts: list[str] = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            for s in re.split(r"(?<=[.!?])\s+", line):
                s = s.strip()
                if len(s) >= 8:
                    parts.append(s)
        return parts

    @staticmethod
    def _is_verifiable(sentence: str) -> bool:
        lower = sentence.lower().strip()
        if any(lower.startswith(f) for f in _FILLER_STARTS):
            if len(lower) < 40:
                return False
        if lower in ("", ".", "...", "```"):
            return False
        return True

    def _decompose(self, sentence: str) -> list[str]:
        parts = re.split(r"\b(?:and|also|additionally|furthermore)\b", sentence)
        result: list[str] = []
        for p in parts:
            p = p.strip().rstrip(".,;:")
            if len(p) >= 8:
                result.append(p)
        return result if result else [sentence]

    def _classify(self, text: str) -> str:
        if self.tool_pattern.search(text):
            return "tool_selection"
        if _ARG_PATTERN.search(text):
            return "argument_claim"
        return "reasoning_chain"

    @staticmethod
    def _find_token_span(
        claim_text: str,
        source_sentence: str,
        completion: str,
        token_ids: list[int],
        tokenizer,
    ) -> tuple[int, int]:
        search = source_sentence
        char_start = completion.find(search)
        if char_start == -1:
            search = claim_text
            char_start = completion.find(search)
        if char_start == -1:
            return (0, len(token_ids))

        char_end = char_start + len(search)

        cum = 0
        tok_start = 0
        tok_end = len(token_ids)
        for i, tid in enumerate(token_ids):
            piece = tokenizer.decode([tid])
            piece_end = cum + len(piece)
            if cum <= char_start < piece_end and tok_start == 0:
                tok_start = i
            if piece_end >= char_end:
                tok_end = i + 1
                break
            cum = piece_end

        return (tok_start, min(tok_end, len(token_ids)))
