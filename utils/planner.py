"""Gap-to-action planning utilities for 30-day improvement plans."""

from __future__ import annotations

from typing import Dict, List, Sequence

SKILL_RESOURCES: Dict[str, str] = {
    "python": "Build one API-focused mini-project in Python and publish code on GitHub.",
    "sql": "Complete advanced SQL practice covering joins, windows, CTEs, and optimization.",
    "aws": "Deploy a small app on AWS (EC2/Lambda + S3) and document architecture.",
    "docker": "Containerize one project and write a production-ready Dockerfile + compose setup.",
    "kubernetes": "Deploy your containerized app to a local cluster (kind/minikube).",
    "fastapi": "Create a FastAPI microservice with auth, validation, and tests.",
    "nlp": "Implement an NLP pipeline (classification or retrieval) with evaluation metrics.",
    "scikit-learn": "Train and compare 2-3 baseline models with cross-validation and feature importance.",
    "machine learning": "Ship an end-to-end ML project with data prep, training, and monitoring notes.",
}


def build_30_day_plan(missing_skills: Sequence[str], role_template: str) -> List[Dict[str, str]]:
    """Turn missing skills into a practical four-week plan."""
    selected = list(missing_skills[:8])
    if not selected:
        selected = ["portfolio polish", "resume positioning"]

    week_1 = selected[:2]
    week_2 = selected[2:4]
    week_3 = selected[4:6]
    week_4 = selected[6:8]

    def compose(week_skills: List[str], generic: str) -> str:
        if not week_skills:
            return generic
        actions = [SKILL_RESOURCES.get(skill.lower(), f"Create a focused mini-project showcasing {skill}.") for skill in week_skills]
        return " ".join(actions)

    plan = [
        {
            "week": "Week 1",
            "focus": ", ".join(week_1) if week_1 else "Baseline alignment",
            "actions": compose(week_1, "Audit JD gaps and rewrite summary/skills for target role alignment."),
        },
        {
            "week": "Week 2",
            "focus": ", ".join(week_2) if week_2 else "Project acceleration",
            "actions": compose(week_2, "Add one impact-oriented project section with quantified outcomes."),
        },
        {
            "week": "Week 3",
            "focus": ", ".join(week_3) if week_3 else "Interview readiness",
            "actions": compose(week_3, "Prepare STAR stories and technical walkthroughs for your strongest projects."),
        },
        {
            "week": "Week 4",
            "focus": ", ".join(week_4) if week_4 else "Final optimization",
            "actions": compose(week_4, f"Tailor resume for {role_template} and run final ATS + match checks."),
        },
    ]
    return plan

