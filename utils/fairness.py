"""Bias and fairness red-flag checks for resume text."""

from __future__ import annotations

import re
from typing import Dict, List

GENDER_CODED = {
    "masculine": ["rockstar", "ninja", "dominant", "aggressive", "fearless"],
    "feminine": ["supportive", "nurturing", "empathetic", "interpersonal"],
}
AGE_SIGNALS = ["young", "recent graduate", "digital native", "over 20 years", "seasoned veteran"]
PROTECTED_TERMS = ["married", "single", "nationality", "religion", "date of birth", "dob"]


def analyze_bias_risks(text: str) -> Dict[str, List[str] | int]:
    """Detect basic wording risks that may hurt fairness/compliance."""
    lowered = (text or "").lower()
    findings: List[str] = []

    for bucket, terms in GENDER_CODED.items():
        hits = [term for term in terms if re.search(rf"\b{re.escape(term)}\b", lowered)]
        if hits:
            findings.append(f"Potential {bucket}-coded wording: {', '.join(hits)}")

    age_hits = [term for term in AGE_SIGNALS if term in lowered]
    if age_hits:
        findings.append(f"Potential age-related phrasing: {', '.join(age_hits)}")

    protected_hits = [term for term in PROTECTED_TERMS if term in lowered]
    if protected_hits:
        findings.append(f"Potential sensitive personal info references: {', '.join(protected_hits)}")

    recommendations = []
    if findings:
        recommendations.extend(
            [
                "Use neutral, competency-based language.",
                "Avoid unnecessary personal or demographic details.",
                "Focus descriptions on measurable achievements and role-relevant skills.",
            ]
        )
    else:
        recommendations.append("No major fairness red flags detected in the current text.")

    return {
        "risk_count": len(findings),
        "findings": findings,
        "recommendations": recommendations,
    }

