"""Pydantic schemas for the resume matching API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    resume_text: str = Field(..., min_length=1)
    jd_text: str = Field(..., min_length=1)
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    use_reranker: bool = Field(default=False)
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    redact_pii: bool = Field(default=False)
    include_suggestions: bool = Field(default=True)
    groq_api_key: Optional[str] = Field(default=None)
    llm_model: str = Field(default="llama-3.1-70b-versatile")


class AnalyzeResponse(BaseModel):
    score_details: Dict[str, Any]
    skill_details: Dict[str, Any]
    ats_details: Dict[str, Any]
    keyword_rows: List[Dict[str, Any]]
    strong_points: List[str]
    explanation: Dict[str, Any]
    suggestions_payload: Dict[str, Any]
    privacy: Dict[str, Any]
    observability: Dict[str, Any]

