from __future__ import annotations

from typing import List

import numpy as np

from utils import matcher


class DummyEmbeddingModel:
    def encode(self, texts: List[str], normalize_embeddings: bool = True):  # noqa: ARG002
        return np.array([[1.0, 0.0, 0.0], [0.8, 0.2, 0.0]])


def test_calculate_match_score_bounds(monkeypatch) -> None:
    monkeypatch.setattr(matcher, "load_embedding_model", lambda model_name: DummyEmbeddingModel())
    score = matcher.calculate_match_score(
        resume_text="Python FastAPI AWS NLP",
        jd_text="Need Python NLP with FastAPI and AWS",
        model_name="fake-model",
        use_reranker=False,
    )
    assert 0 <= score["semantic"] <= 100
    assert 0 <= score["lexical"] <= 100
    assert 0 <= score["overall"] <= 100
    assert "weights" in score


def test_build_match_explanation_has_evidence(monkeypatch) -> None:
    monkeypatch.setattr(matcher, "load_embedding_model", lambda model_name: DummyEmbeddingModel())
    score = matcher.calculate_match_score(
        resume_text="Python FastAPI scikit-learn AWS",
        jd_text="Python FastAPI AWS machine learning",
    )
    skills = matcher.analyze_skill_alignment(
        resume_text="Python FastAPI scikit-learn AWS",
        jd_text="Python FastAPI AWS machine learning",
    )
    explanation = matcher.build_match_explanation(
        resume_text="Python FastAPI scikit-learn AWS",
        jd_text="Python FastAPI AWS machine learning",
        score_details=score,
        skill_details=skills,
    )
    assert "feature_contributions" in explanation
    assert len(explanation["evidence_points"]) >= 1

