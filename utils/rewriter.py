"""Resume rewriting helpers with before/after diff rendering."""

from __future__ import annotations

import html
import re
from difflib import SequenceMatcher
from itertools import zip_longest
from typing import Dict, List, Sequence, Tuple

from utils.skills_db import STOPWORDS as TEXT_STOPWORDS
from utils.skills_db import get_flat_skills

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

GENERIC_KEYWORDS = {
    "senior",
    "junior",
    "engineer",
    "developer",
    "specialist",
    "candidate",
    "role",
    "position",
    "team",
    "company",
    "organization",
    "business",
    "work",
    "working",
    "experience",
    "responsibility",
    "responsibilities",
    "requirement",
    "requirements",
    "strong",
    "excellent",
    "good",
}

ACTION_PREFIX_RE = re.compile(
    r"^(?:built|designed|developed|implemented|optimized|automated|led|launched|deployed|improved|delivered|scaled)\b",
    flags=re.IGNORECASE,
)
BULLET_PREFIX_RE = re.compile(r"^\s*[-*\u2022]\s+")


def extract_bullets(text: str, max_items: int = 12) -> List[str]:
    bullets: List[str] = []
    for line in (text or "").splitlines():
        if BULLET_PREFIX_RE.match(line):
            clean = BULLET_PREFIX_RE.sub("", line).strip()
            if clean:
                bullets.append(clean)
    return bullets[:max_items]


def _normalize_keyword(keyword: str) -> str:
    return re.sub(r"\s+", " ", (keyword or "").strip())


def _looks_skill_like(keyword: str, known_skills: set[str]) -> bool:
    key = _normalize_keyword(keyword)
    if not key:
        return False

    lowered = key.lower()
    if lowered in known_skills:
        return True

    tokens = [token for token in re.split(r"[\s/]+", lowered) if token]
    if not tokens or len(tokens) > 4:
        return False
    if all(token in TEXT_STOPWORDS or token in GENERIC_KEYWORDS for token in tokens):
        return False

    if re.search(r"[+#./]", key):
        return True
    if re.search(r"\d", key):
        return True
    if key.isupper() and 2 <= len(key) <= 8:
        return True

    tech_stems = (
        "api",
        "ml",
        "ai",
        "llm",
        "rag",
        "sql",
        "data",
        "cloud",
        "devops",
        "docker",
        "kubernetes",
        "terraform",
        "aws",
        "azure",
        "gcp",
        "fastapi",
        "streamlit",
        "pytorch",
        "tensorflow",
        "langchain",
        "faiss",
        "pinecone",
    )
    return any(stem in lowered for stem in tech_stems)


def filter_skill_like_keywords(target_keywords: Sequence[str], limit: int = 10) -> List[str]:
    """Keep only technical/skill-like terms and drop generic role words."""
    known_skills = get_flat_skills()
    filtered: List[str] = []
    seen: set[str] = set()

    for raw in target_keywords or []:
        key = _normalize_keyword(raw)
        if not key:
            continue
        lowered = key.lower()
        if lowered in seen:
            continue
        if not _looks_skill_like(key, known_skills):
            continue
        seen.add(lowered)
        filtered.append(key)
        if len(filtered) >= limit:
            break
    return filtered


def _sanitize_using_clause(text: str) -> str:
    """Normalize trailing 'using ...' clauses by keeping only skill-like terms."""
    match = re.search(r"^(?P<prefix>.*?)(?:,\s*)?using\s+(?P<terms>[^.]+)\.?\s*$", text, flags=re.IGNORECASE)
    if not match:
        return text

    prefix = match.group("prefix").strip(" ,;")
    terms = [
        part.strip()
        for part in re.split(r",|/|\band\b", match.group("terms"), flags=re.IGNORECASE)
        if part.strip()
    ]
    skill_terms = filter_skill_like_keywords(terms, limit=2)
    if not skill_terms:
        return prefix
    return f"{prefix} using {', '.join(skill_terms)}"


def rewrite_bullet_rule_based(bullet: str, target_keywords: Sequence[str]) -> str:
    """Rewrite bullet to be action-first without forced keyword/KPI suffixes."""
    text = bullet.strip()
    if not text:
        return text

    # `target_keywords` is pre-filtered upstream to skill-like terms only.
    del target_keywords

    if not ACTION_PREFIX_RE.match(text):
        verb = STRONG_VERBS[hash(text) % len(STRONG_VERBS)]
        text = f"{verb} {text[0].lower()}{text[1:]}" if len(text) > 1 else f"{verb} {text}"

    text = _sanitize_using_clause(text)
    text = re.sub(r"\s+", " ", text).strip().rstrip(";")
    if not text.endswith("."):
        text += "."
    return text


def rewrite_resume_bullets(resume_text: str, target_keywords: Sequence[str]) -> Dict[str, List[str]]:
    original = extract_bullets(resume_text)
    filtered_keywords = filter_skill_like_keywords(target_keywords, limit=10)
    rewritten = [rewrite_bullet_rule_based(bullet, filtered_keywords) for bullet in original]
    return {"original": original, "rewritten": rewritten}


def _word_level_diff_html(original_text: str, rewritten_text: str) -> Tuple[str, str]:
    original_words = (original_text or "").split()
    rewritten_words = (rewritten_text or "").split()
    matcher = SequenceMatcher(None, original_words, rewritten_words)

    original_html: List[str] = []
    rewritten_html: List[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            original_html.extend(f"<span class='rw-same'>{html.escape(w)}</span>" for w in original_words[i1:i2])
            rewritten_html.extend(f"<span class='rw-same'>{html.escape(w)}</span>" for w in rewritten_words[j1:j2])
        elif tag == "delete":
            original_html.extend(f"<span class='rw-del'>{html.escape(w)}</span>" for w in original_words[i1:i2])
        elif tag == "insert":
            rewritten_html.extend(f"<span class='rw-add'>{html.escape(w)}</span>" for w in rewritten_words[j1:j2])
        else:
            original_html.extend(f"<span class='rw-del'>{html.escape(w)}</span>" for w in original_words[i1:i2])
            rewritten_html.extend(f"<span class='rw-add'>{html.escape(w)}</span>" for w in rewritten_words[j1:j2])

    return " ".join(original_html), " ".join(rewritten_html)


def render_bullet_diff_html(original: Sequence[str], rewritten: Sequence[str]) -> str:
    rows: List[str] = []
    for idx, (orig, rew) in enumerate(zip_longest(original, rewritten, fillvalue=""), start=1):
        orig_html, rew_html = _word_level_diff_html(orig, rew)
        rows.append(
            f"""
            <div class="rw-row">
              <div class="rw-num">{idx}</div>
              <div class="rw-cell">{orig_html or "<span class='rw-muted'>(empty)</span>"}</div>
              <div class="rw-cell">{rew_html or "<span class='rw-muted'>(empty)</span>"}</div>
            </div>
            """
        )

    if not rows:
        rows.append(
            """
            <div class="rw-row">
              <div class="rw-num">1</div>
              <div class="rw-cell"><span class="rw-muted">No bullets found.</span></div>
              <div class="rw-cell"><span class="rw-muted">No rewritten bullets generated.</span></div>
            </div>
            """
        )

    return f"""
    <style>
      .rw-shell {{
        font-family: "Segoe UI", "Inter", Arial, sans-serif;
        color: #e6f2ff;
        background: #0b1220;
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 12px;
        padding: 10px;
      }}
      .rw-head {{
        display: grid;
        grid-template-columns: 54px 1fr 1fr;
        gap: 10px;
        margin-bottom: 8px;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #9fb3c8;
        font-weight: 700;
      }}
      .rw-row {{
        display: grid;
        grid-template-columns: 54px 1fr 1fr;
        gap: 10px;
        align-items: start;
        margin-bottom: 8px;
      }}
      .rw-num {{
        background: rgba(148,163,184,0.18);
        border: 1px solid rgba(148,163,184,0.35);
        border-radius: 999px;
        min-height: 28px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        color: #dbeafe;
      }}
      .rw-cell {{
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        background: rgba(15,23,42,0.72);
        padding: 9px 10px;
        line-height: 1.5;
        min-height: 42px;
        color: #dbeafe;
        word-break: break-word;
      }}
      .rw-same {{ color: #dbeafe; }}
      .rw-add {{
        background: rgba(16,185,129,0.25);
        border-radius: 5px;
        padding: 0 3px;
        color: #86efac;
        font-weight: 600;
      }}
      .rw-del {{
        background: rgba(239,68,68,0.22);
        border-radius: 5px;
        padding: 0 3px;
        color: #fca5a5;
        text-decoration: line-through;
      }}
      .rw-muted {{ color: #93a4b8; font-style: italic; }}
      @media (max-width: 820px) {{
        .rw-head {{
          grid-template-columns: 44px 1fr;
        }}
        .rw-head div:last-child {{
          display: none;
        }}
        .rw-row {{
          grid-template-columns: 44px 1fr;
        }}
        .rw-row .rw-cell:last-child {{
          margin-top: 6px;
          grid-column: 2 / 3;
        }}
      }}
    </style>
    <div class="rw-shell">
      <div class="rw-head">
        <div>#</div>
        <div>Original</div>
        <div>Rewritten</div>
      </div>
      {''.join(rows)}
    </div>
    """
