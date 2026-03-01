"""Role templates that tune scoring strategy and skill expectations."""

from __future__ import annotations

from typing import Any, Dict, List

from utils.skills_db import get_flat_skills

ROLE_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "General": {
        "description": "Balanced template for most software/analytics roles.",
        "weights_no_reranker": {"semantic": 0.62, "lexical": 0.14, "skill_alignment": 0.24},
        "weights_with_reranker": {
            "semantic": 0.48,
            "lexical": 0.12,
            "skill_alignment": 0.20,
            "reranker": 0.20,
        },
        "extra_skills": [],
    },
    "Data Scientist": {
        "description": "Prioritizes ML/NLP skill coverage and semantic relevance.",
        "weights_no_reranker": {"semantic": 0.55, "lexical": 0.10, "skill_alignment": 0.35},
        "weights_with_reranker": {
            "semantic": 0.42,
            "lexical": 0.08,
            "skill_alignment": 0.30,
            "reranker": 0.20,
        },
        "extra_skills": [
            "feature engineering",
            "model evaluation",
            "xgboost",
            "lightgbm",
            "airflow",
            "statistics",
            "experimentation",
        ],
    },
    "Backend Engineer": {
        "description": "Emphasizes architecture, APIs, infra, and platform reliability.",
        "weights_no_reranker": {"semantic": 0.50, "lexical": 0.20, "skill_alignment": 0.30},
        "weights_with_reranker": {
            "semantic": 0.40,
            "lexical": 0.16,
            "skill_alignment": 0.24,
            "reranker": 0.20,
        },
        "extra_skills": [
            "microservices",
            "rest api",
            "grpc",
            "system design",
            "observability",
            "distributed systems",
            "message queue",
        ],
    },
    "Frontend Engineer": {
        "description": "Focuses on UI stack fit and product delivery language.",
        "weights_no_reranker": {"semantic": 0.52, "lexical": 0.20, "skill_alignment": 0.28},
        "weights_with_reranker": {
            "semantic": 0.40,
            "lexical": 0.16,
            "skill_alignment": 0.24,
            "reranker": 0.20,
        },
        "extra_skills": [
            "redux",
            "webpack",
            "tailwind",
            "responsive design",
            "accessibility",
            "design systems",
            "figma",
        ],
    },
    "Product Manager": {
        "description": "Increases weight on semantic fit and stakeholder language.",
        "weights_no_reranker": {"semantic": 0.64, "lexical": 0.16, "skill_alignment": 0.20},
        "weights_with_reranker": {
            "semantic": 0.50,
            "lexical": 0.14,
            "skill_alignment": 0.16,
            "reranker": 0.20,
        },
        "extra_skills": [
            "roadmap",
            "go-to-market",
            "product discovery",
            "user research",
            "prioritization",
            "kpi",
        ],
    },
}


def get_role_template_names() -> List[str]:
    return list(ROLE_TEMPLATES.keys())


def get_role_template(name: str) -> Dict[str, Any]:
    return ROLE_TEMPLATES.get(name, ROLE_TEMPLATES["General"])


def get_template_weights(role_name: str, use_reranker: bool) -> Dict[str, float]:
    template = get_role_template(role_name)
    weights = (
        template["weights_with_reranker"] if use_reranker else template["weights_no_reranker"]
    )
    total = sum(float(value) for value in weights.values()) or 1.0
    return {key: float(value) / total for key, value in weights.items()}


def get_template_skill_pool(role_name: str) -> List[str]:
    base = set(get_flat_skills())
    extra = set(get_role_template(role_name).get("extra_skills", []))
    return sorted(base | extra)

