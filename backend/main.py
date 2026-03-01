"""FastAPI service for resume-job matching analysis."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException

from backend.schemas import AnalyzeRequest, AnalyzeResponse
from utils.llm_suggestions import generate_improvement_suggestions
from utils.matcher import (
    analyze_skill_alignment,
    build_match_explanation,
    build_strong_points,
    calculate_ats_compatibility,
    calculate_keyword_density,
    calculate_match_score,
    load_embedding_model,
    load_reranker_model,
)
from utils.observability import AnalysisMetrics
from utils.privacy import redact_pii

app = FastAPI(
    title="Resume Screener API",
    version="1.0.0",
    description="Production API for AI resume screening and JD matching",
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    resume_text = payload.resume_text.strip()
    jd_text = payload.jd_text.strip()
    if not resume_text or not jd_text:
        raise HTTPException(status_code=400, detail="Both resume_text and jd_text are required.")

    metrics = AnalysisMetrics()
    privacy_summary: Dict[str, Any] = {}

    if payload.redact_pii:
        resume_text, resume_pii = redact_pii(resume_text)
        jd_text, jd_pii = redact_pii(jd_text)
        privacy_summary = {"resume_pii_redacted": resume_pii, "jd_pii_redacted": jd_pii}

    metrics.start_stage("model_warmup")
    load_embedding_model(payload.embedding_model)
    if payload.use_reranker:
        load_reranker_model(payload.reranker_model)
    metrics.end_stage("model_warmup")

    metrics.start_stage("scoring")
    score_details = calculate_match_score(
        resume_text=resume_text,
        jd_text=jd_text,
        model_name=payload.embedding_model,
        use_reranker=payload.use_reranker,
        reranker_model=payload.reranker_model,
    )
    metrics.end_stage("scoring")

    metrics.start_stage("skills")
    skill_details = analyze_skill_alignment(resume_text, jd_text)
    metrics.end_stage("skills")

    metrics.start_stage("ats_keyword")
    ats_details = calculate_ats_compatibility(resume_text, jd_text, skill_details)
    keyword_rows = calculate_keyword_density(resume_text, jd_text, top_n=15)
    metrics.end_stage("ats_keyword")

    metrics.start_stage("explainability")
    explanation = build_match_explanation(
        resume_text=resume_text,
        jd_text=jd_text,
        score_details=score_details,
        skill_details=skill_details,
        top_n_keywords=15,
    )
    strong_points = build_strong_points(resume_text, score_details, skill_details)
    metrics.end_stage("explainability")

    if payload.include_suggestions:
        metrics.start_stage("llm_suggestions")
        suggestions_payload = generate_improvement_suggestions(
            resume_text=resume_text,
            jd_text=jd_text,
            match_score=float(score_details["overall"]),
            missing_skills=skill_details["missing_skills"],
            matching_skills=skill_details["matching_skills"],
            groq_api_key=payload.groq_api_key,
            model_name=payload.llm_model,
        )
        usage = suggestions_payload.get("usage", {})
        metrics.set_llm_usage(
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            estimated_cost_usd=float(usage.get("estimated_cost_usd", 0.0)),
        )
        metrics.end_stage("llm_suggestions")
    else:
        suggestions_payload = {
            "provider": "disabled",
            "suggestions": [],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "estimated_cost_usd": 0.0},
        }

    return AnalyzeResponse(
        score_details=score_details,
        skill_details=skill_details,
        ats_details=ats_details,
        keyword_rows=keyword_rows,
        strong_points=strong_points,
        explanation=explanation,
        suggestions_payload=suggestions_payload,
        privacy=privacy_summary,
        observability=metrics.to_dict(),
    )

