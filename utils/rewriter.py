"""Resume rewriting helpers with before/after diff rendering."""

from __future__ import annotations

import difflib
import re
from typing import Dict, List, Sequence

STRONG_VERBS = [
    "Designed",
    "Built",
    "Optimized",
    "Automated",
    "Delivered",
    "Led",
    "Implemented",
    "Scaled",
]


def extract_bullets(text: str, max_items: int = 12) -> List[str]:
    bullets = []
    for line in (text or "").splitlines():
        if re.match(r"^\s*[-*•]\s+", line):
            clean = re.sub(r"^\s*[-*•]\s+", "", line).strip()
            if clean:
                bullets.append(clean)
    return bullets[:max_items]


def rewrite_bullet_rule_based(bullet: str, target_keywords: Sequence[str]) -> str:
    """Rewrite bullet to be action-first and keyword-aligned."""
    text = bullet.strip()
    if not text:
        return text

    if not re.match(r"^[A-Z][a-z]+", text):
        verb = STRONG_VERBS[hash(text) % len(STRONG_VERBS)]
        text = f"{verb} {text[0].lower()}{text[1:]}" if len(text) > 1 else f"{verb} {text}"

    lower = text.lower()
    missing_keywords = [keyword for keyword in target_keywords if keyword.lower() not in lower][:2]
    if missing_keywords:
        text = f"{text} using {', '.join(missing_keywords)}"

    if "%" not in text and not re.search(r"\b\d+\b", text):
        text = f"{text}, improving key KPI by 15%."
    elif not text.endswith("."):
        text = f"{text}."
    return text


def rewrite_resume_bullets(resume_text: str, target_keywords: Sequence[str]) -> Dict[str, List[str]]:
    original = extract_bullets(resume_text)
    rewritten = [rewrite_bullet_rule_based(bullet, target_keywords) for bullet in original]
    return {"original": original, "rewritten": rewritten}


def render_bullet_diff_html(original: Sequence[str], rewritten: Sequence[str]) -> str:
    diff = difflib.HtmlDiff(wrapcolumn=110)
    original_lines = [f"- {item}" for item in original] or ["No bullets found."]
    rewritten_lines = [f"- {item}" for item in rewritten] or ["No rewritten bullets generated."]
    return diff.make_table(
        original_lines,
        rewritten_lines,
        fromdesc="Original Bullets",
        todesc="Rewritten Bullets",
        context=True,
        numlines=2,
    )

