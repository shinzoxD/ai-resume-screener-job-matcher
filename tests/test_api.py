from __future__ import annotations

from fastapi.testclient import TestClient

import backend.main as api


def test_health_endpoint() -> None:
    client = TestClient(api.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_endpoint_with_monkeypatched_pipeline(monkeypatch) -> None:
    monkeypatch.setattr(api, "load_embedding_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(api, "load_reranker_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        api,
        "calculate_match_score",
        lambda **_kwargs: {
            "semantic": 80.0,
            "lexical": 70.0,
            "skill_alignment": 65.0,
            "reranker": None,
            "overall": 74.9,
            "weights": {"semantic": 0.6, "lexical": 0.2, "skill_alignment": 0.2},
        },
    )
    monkeypatch.setattr(
        api,
        "analyze_skill_alignment",
        lambda *_args, **_kwargs: {
            "resume_skills": ["python", "fastapi"],
            "jd_skills": ["python", "fastapi", "aws"],
            "matching_skills": ["python", "fastapi"],
            "missing_skills": ["aws"],
            "extra_skills": [],
            "coverage_score": 66.67,
        },
    )
    monkeypatch.setattr(
        api,
        "calculate_ats_compatibility",
        lambda *_args, **_kwargs: {
            "ats_score": 78.2,
            "keyword_score": 66.67,
            "section_score": 80.0,
            "contact_score": 100.0,
            "length_score": 75.0,
            "bullet_score": 80.0,
            "tips": ["Keep iterating"],
        },
    )
    monkeypatch.setattr(
        api,
        "calculate_keyword_density",
        lambda *_args, **_kwargs: [
            {
                "keyword": "python",
                "jd_frequency": 3,
                "resume_frequency": 2,
                "density_pct": 1.2,
                "status": "Matched",
            }
        ],
    )
    monkeypatch.setattr(
        api,
        "build_match_explanation",
        lambda *_args, **_kwargs: {
            "feature_contributions": [],
            "matched_keywords": ["python"],
            "missing_keywords": ["aws"],
            "evidence_points": ["Good alignment."],
        },
    )
    monkeypatch.setattr(api, "build_strong_points", lambda *_args, **_kwargs: ["Strong profile."])
    monkeypatch.setattr(
        api,
        "generate_improvement_suggestions",
        lambda **_kwargs: {
            "provider": "rule-based",
            "suggestions": ["Add AWS project bullet."],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "estimated_cost_usd": 0.0},
        },
    )

    client = TestClient(api.app)
    response = client.post(
        "/analyze",
        json={
            "resume_text": "Python FastAPI engineer",
            "jd_text": "Need Python FastAPI AWS",
            "include_suggestions": True,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert "score_details" in payload
    assert payload["score_details"]["overall"] == 74.9

