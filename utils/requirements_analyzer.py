"""Job requirement extraction and evidence mapping against resume text."""

from __future__ import annotations

import re
from typing import Dict, List

from utils.skills_db import extract_keywords


def extract_jd_requirements(jd_text: str) -> Dict[str, List[str]]:
    """Extract must-have and nice-to-have requirements from JD text."""
    lines = [line.strip() for line in (jd_text or "").splitlines() if line.strip()]
    mode = "must_have"
    buckets: Dict[str, List[str]] = {"must_have": [], "nice_to_have": []}

    for line in lines:
        lower = line.lower()
        if any(marker in lower for marker in ("nice to have", "preferred", "good to have")):
            mode = "nice_to_have"
            continue
        if any(marker in lower for marker in ("requirements", "must have", "qualifications")):
            mode = "must_have"
            continue
        if re.match(r"^[-*•\d\.)]+\s+", line):
            clean = re.sub(r"^[-*•\d\.)]+\s+", "", line).strip()
            if len(clean) > 4:
                buckets[mode].append(clean)

    # Fallback: derive requirements from JD sentences.
    if not buckets["must_have"]:
        sentences = re.split(r"(?<=[.!?])\s+", jd_text or "")
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            lower = sentence.lower()
            if any(token in lower for token in ("require", "must", "experience with", "strong in")):
                buckets["must_have"].append(sentence)
            elif any(token in lower for token in ("prefer", "nice to have")):
                buckets["nice_to_have"].append(sentence)

    buckets["must_have"] = buckets["must_have"][:12]
    buckets["nice_to_have"] = buckets["nice_to_have"][:10]
    return buckets


def _extract_evidence_snippet(resume_text: str, requirement: str) -> str:
    resume_lines = [line.strip() for line in (resume_text or "").splitlines() if line.strip()]
    keywords = [token for token, _ in extract_keywords(requirement, top_n=5) if len(token) >= 3]
    for line in resume_lines:
        lower = line.lower()
        if any(keyword in lower for keyword in keywords):
            return line
    return ""


def map_requirements_to_evidence(resume_text: str, requirements: Dict[str, List[str]]) -> List[Dict[str, str]]:
    """Map each requirement to present/missing status and a supporting snippet."""
    rows: List[Dict[str, str]] = []
    resume_lower = (resume_text or "").lower()

    for category in ("must_have", "nice_to_have"):
        for requirement in requirements.get(category, []):
            req_keywords = [token for token, _ in extract_keywords(requirement, top_n=6) if len(token) >= 3]
            is_present = any(keyword in resume_lower for keyword in req_keywords)
            evidence = _extract_evidence_snippet(resume_text, requirement) if is_present else ""
            rows.append(
                {
                    "category": "Must-Have" if category == "must_have" else "Nice-to-Have",
                    "requirement": requirement,
                    "status": "Covered" if is_present else "Gap",
                    "evidence": evidence or "No direct evidence found.",
                }
            )

    return rows

