from __future__ import annotations

from utils.confidence import compute_confidence
from utils.multilingual import choose_embedding_model, detect_language
from utils.planner import build_30_day_plan
from utils.requirements_analyzer import extract_jd_requirements, map_requirements_to_evidence
from utils.role_templates import get_template_weights


def test_role_template_weights_normalized() -> None:
    weights = get_template_weights("Data Scientist", use_reranker=True)
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_requirements_extraction_and_mapping() -> None:
    jd = """
    Requirements:
    - Python and FastAPI experience
    - AWS and Docker
    Nice to Have:
    - Kubernetes
    """
    resume = "Built FastAPI services in Python on AWS using Docker."
    req = extract_jd_requirements(jd)
    mapping = map_requirements_to_evidence(resume, req)
    assert len(req["must_have"]) >= 1
    assert any(row["status"] == "Covered" for row in mapping)


def test_confidence_output_shape() -> None:
    confidence = compute_confidence(
        {"semantic": 80, "lexical": 60, "skill_alignment": 75, "overall": 74}
    )
    assert confidence["band"] in {"High", "Medium", "Low"}
    assert 0 <= confidence["confidence_pct"] <= 100


def test_multilingual_router() -> None:
    lang = detect_language("Experiencia en Python y aprendizaje automático.")
    model = choose_embedding_model("all-MiniLM-L6-v2", lang, auto_multilingual=True)
    assert model in {"all-MiniLM-L6-v2", "paraphrase-multilingual-MiniLM-L12-v2"}


def test_planner_has_four_weeks() -> None:
    plan = build_30_day_plan(["python", "aws", "docker"], "Backend Engineer")
    assert len(plan) == 4

