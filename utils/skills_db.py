"""Skill ontology and text keyword helpers for resume-job matching."""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Set, Tuple

SKILL_ONTOLOGY: Dict[str, List[str]] = {
    "programming_languages": [
        "python",
        "java",
        "javascript",
        "typescript",
        "c",
        "c++",
        "c#",
        "go",
        "rust",
        "sql",
        "r",
        "scala",
        "kotlin",
        "swift",
    ],
    "frameworks_libraries": [
        "django",
        "flask",
        "fastapi",
        "streamlit",
        "react",
        "next.js",
        "angular",
        "vue",
        "node.js",
        "express",
        "spring boot",
        "pytorch",
        "tensorflow",
        "scikit-learn",
        "pandas",
        "numpy",
        "langchain",
    ],
    "cloud_devops": [
        "aws",
        "azure",
        "gcp",
        "docker",
        "kubernetes",
        "terraform",
        "jenkins",
        "github actions",
        "ci/cd",
        "linux",
        "nginx",
        "serverless",
    ],
    "data_ai": [
        "machine learning",
        "deep learning",
        "nlp",
        "computer vision",
        "llm",
        "rag",
        "vector database",
        "faiss",
        "pinecone",
        "data analysis",
        "data visualization",
        "tableau",
        "power bi",
        "etl",
    ],
    "databases": [
        "postgresql",
        "mysql",
        "mongodb",
        "redis",
        "elasticsearch",
        "sqlite",
        "snowflake",
        "bigquery",
    ],
    "product_business": [
        "agile",
        "scrum",
        "stakeholder management",
        "product management",
        "a/b testing",
        "communication",
        "problem solving",
        "leadership",
    ],
}

STOPWORDS: Set[str] = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "to",
    "for",
    "in",
    "on",
    "at",
    "of",
    "with",
    "by",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "that",
    "this",
    "it",
    "its",
    "you",
    "your",
    "we",
    "our",
    "their",
    "they",
    "will",
    "can",
    "could",
    "should",
    "must",
    "have",
    "has",
    "had",
    "if",
    "but",
    "about",
    "into",
    "across",
    "within",
    "role",
    "job",
    "candidate",
    "experience",
    "skills",
    "ability",
    "work",
    "team",
    "years",
    "year",
    "using",
    "strong",
    "good",
    "excellent",
    "required",
    "preferred",
    "responsibilities",
    "qualification",
}


def normalize_text(text: str) -> str:
    """Lowercase and normalize whitespace while retaining skill characters."""
    if not text:
        return ""
    text = text.lower()
    text = text.replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"[^\w\s\+\#\.\-/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_flat_skills() -> Set[str]:
    """Return the full canonical set of skills from the ontology."""
    flat_skills: Set[str] = set()
    for skill_group in SKILL_ONTOLOGY.values():
        flat_skills.update(skill.lower() for skill in skill_group)
    return flat_skills


def extract_skills_from_text(text: str, custom_skills: Sequence[str] | None = None) -> Set[str]:
    """Extract skill mentions by exact phrase matching with safe boundaries."""
    normalized = normalize_text(text)
    if not normalized:
        return set()

    skill_pool = set(skill.lower().strip() for skill in (custom_skills or get_flat_skills()))
    found: Set[str] = set()

    for skill in skill_pool:
        if not skill:
            continue
        pattern = rf"(?<![a-z0-9]){re.escape(skill)}(?![a-z0-9])"
        if re.search(pattern, normalized):
            found.add(skill)

    return found


def extract_keywords(text: str, top_n: int = 20, min_len: int = 3) -> List[Tuple[str, int]]:
    """Extract top keywords by frequency after lightweight stop-word filtering."""
    normalized = normalize_text(text)
    if not normalized:
        return []

    tokens = re.findall(r"[a-z][a-z0-9\+\#\.\-/]{1,}", normalized)
    filtered = [
        token
        for token in tokens
        if len(token) >= min_len and token not in STOPWORDS and not token.isdigit()
    ]

    counts = Counter(filtered)
    return counts.most_common(top_n)


def format_skill_list(skills: Iterable[str]) -> List[str]:
    """Format skill list with deterministic ordering and title-cased labels."""
    return [skill.title() for skill in sorted(set(skills))]
