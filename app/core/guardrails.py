"""
Guardrails — content-level input/output safety checks for LLM calls.
This is separate from (and complements) the Cypher-injection allowlist in
app.services.graph_service, which only protects the graph layer.
"""

import re
from typing import Tuple

_PROMPT_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous",
    "you are now",
    "system:",
    "as an ai language model",
]

_PII_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "SSN"),
    (re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I), "EMAIL"),
    (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "CREDIT_CARD"),
]

MAX_INPUT_LENGTH = 10_000


def check_input(text: str) -> Tuple[bool, str]:
    lower = text.lower()
    for pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern in lower:
            return False, f"Prompt injection detected: {pattern!r}"
    if len(text) > MAX_INPUT_LENGTH:
        return False, "Input exceeds maximum length"
    return True, ""


def redact_output(text: str) -> str:
    for pattern, label in _PII_PATTERNS:
        text = pattern.sub(f"[REDACTED_{label}]", text)
    return text
