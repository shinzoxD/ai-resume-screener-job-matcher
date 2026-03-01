"""Tailored resume draft generation utilities."""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from utils.skills_db import extract_keywords


def build_tailored_resume_draft(
    resume_text: str,
    jd_text: str,
    role_template: str,
    matching_skills: Sequence[str],
    missing_skills: Sequence[str],
) -> str:
    """Build an ATS-safe tailored resume draft in markdown format."""
    jd_keywords = [token for token, _ in extract_keywords(jd_text, top_n=12)]
    priority_keywords = [keyword for keyword in jd_keywords if keyword in (resume_text or "").lower()][:8]

    draft = [
        "# Tailored Resume Draft",
        "",
        f"Target Role Template: **{role_template}**",
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d')}",
        "",
        "## Professional Summary",
        (
            "Results-driven professional aligned to the target job profile, with proven delivery in "
            f"{', '.join(priority_keywords[:4]) if priority_keywords else 'core role competencies'}."
        ),
        "",
        "## Core Skills",
        ", ".join(matching_skills[:10]) if matching_skills else "Add role-relevant technical skills from JD.",
        "",
        "## Experience Highlights",
        "- Quantified impact bullet #1 tailored to JD priorities.",
        "- Quantified impact bullet #2 aligned to core requirements.",
        "- Quantified impact bullet #3 showcasing ownership and cross-functional results.",
        "",
        "## Projects",
        "- Add one project directly mapped to top JD requirements.",
        "- Include stack, problem statement, measurable outcome, and deployment context.",
        "",
        "## Gap Closure Notes",
        ", ".join(missing_skills[:8]) if missing_skills else "No critical skill gaps detected.",
        "",
        "## ATS Checklist",
        "- Use standard section headers.",
        "- Keep role keywords naturally distributed across summary, skills, and bullets.",
        "- Keep formatting clean: no text boxes/images for critical content.",
        "",
    ]
    return "\n".join(draft)

