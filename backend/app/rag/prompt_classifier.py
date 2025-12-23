"""Rule-based prompt type classifier.

Detects coarse prompt types to drive RAG filtering and augmentation without
requiring a user-selected target model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple


PROMPT_TYPES = [
    "code",
    "analysis",
    "explain",
    "creative",
    "summarize",
    "troubleshoot",
    "other",
]


@dataclass
class PromptClassification:
    prompt_type: str
    confidence: float
    reasons: List[str]


def _match_any(text: str, patterns: List[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def classify_prompt(raw_prompt: str) -> PromptClassification:
    """Classify a prompt into a coarse type using lightweight heuristics."""
    text = raw_prompt.strip()
    reasons: List[str] = []

    code_patterns = [r"\bcode\b", r"\bsnippet\b", r"\bpython\b", r"\bjavascript\b", r"\bbug\b", r"\brefactor\b"]
    analysis_patterns = [r"compare", r"evaluate", r"analy[sz]e", r"trade[- ]?offs", r"pros and cons"]
    explain_patterns = [r"explain", r"what is", r"how does", r"walk me through", r"overview"]
    creative_patterns = [r"story", r"poem", r"creative", r"brainstorm", r"ideas", r"tagline", r"slogan"]
    summarize_patterns = [r"summari[sz]e", r"tl;dr", r"condense", r"bullet points", r"key points"]
    troubleshoot_patterns = [r"error", r"exception", r"why fails", r"not working", r"fix", r"diagnos"]

    if _match_any(text, code_patterns):
        reasons.append("code-related keywords detected")
        prompt_type = "code"
    elif _match_any(text, troubleshoot_patterns):
        reasons.append("troubleshooting keywords detected")
        prompt_type = "troubleshoot"
    elif _match_any(text, analysis_patterns):
        reasons.append("analysis/comparison keywords detected")
        prompt_type = "analysis"
    elif _match_any(text, summarize_patterns):
        reasons.append("summarization keywords detected")
        prompt_type = "summarize"
    elif _match_any(text, creative_patterns):
        reasons.append("creative/ideation keywords detected")
        prompt_type = "creative"
    elif _match_any(text, explain_patterns):
        reasons.append("explanation keywords detected")
        prompt_type = "explain"
    else:
        prompt_type = "other"
        reasons.append("no strong keyword match; defaulting to 'other'")

    confidence = 0.55 if prompt_type != "other" else 0.35
    return PromptClassification(prompt_type=prompt_type, confidence=confidence, reasons=reasons)


def list_prompt_types() -> Tuple[str, ...]:
    """Return supported prompt types."""
    return tuple(PROMPT_TYPES)
