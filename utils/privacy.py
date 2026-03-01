"""Privacy helpers for PII redaction and data-handling controls."""

from __future__ import annotations

import re
from typing import Dict, Tuple

PII_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"),
    "linkedin": re.compile(r"https?://(?:www\.)?linkedin\.com/[^\s]+|linkedin\.com/[^\s]+"),
    "github": re.compile(r"https?://(?:www\.)?github\.com/[^\s]+|github\.com/[^\s]+"),
}


def redact_pii(text: str) -> Tuple[str, Dict[str, int]]:
    """Redact common personally identifiable information from free text."""
    redacted = text or ""
    counters: Dict[str, int] = {key: 0 for key in PII_PATTERNS}

    for label, pattern in PII_PATTERNS.items():
        matches = pattern.findall(redacted)
        counters[label] = len(matches)
        if matches:
            token = f"[REDACTED_{label.upper()}]"
            redacted = pattern.sub(token, redacted)

    return redacted, counters

