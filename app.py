"""ResumePilot AI with advanced recruiter/candidate features."""

from __future__ import annotations

import io
import json
import os
import re
import textwrap
from datetime import datetime
from typing import Any, Dict, List, Sequence

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from utils.confidence import compute_confidence
from utils.extractor import detect_sections, extract_document, extract_text_from_input
from utils.fairness import analyze_bias_risks
from utils.llm_suggestions import build_suggestions_markdown, generate_improvement_suggestions
from utils.matcher import (
    DEFAULT_RERANKER_MODEL,
    analyze_skill_alignment,
    build_match_explanation,
    build_strong_points,
    calculate_ats_compatibility,
    calculate_keyword_density,
    calculate_match_score,
    calculate_section_scores,
    load_embedding_model,
    load_reranker_model,
)
from utils.multilingual import choose_embedding_model, detect_language, language_label
from utils.observability import AnalysisMetrics
from utils.pdf_viewer import highlight_keyword_snippets, render_pdf_preview
from utils.planner import build_30_day_plan
from utils.privacy import redact_pii
from utils.requirements_analyzer import extract_jd_requirements, map_requirements_to_evidence
from utils.rewriter import render_bullet_diff_html, rewrite_resume_bullets
from utils.role_templates import get_role_template, get_role_template_names, get_template_skill_pool, get_template_weights
from utils.skills_db import extract_skills_from_text, format_skill_list
from utils.tailor import build_tailored_resume_draft

SAMPLE_RESUME_TEXT = """
Alex Morgan
Senior Data Scientist | Seattle, WA
Email: alex.morgan@email.com | Phone: +1 206-555-0145
LinkedIn: linkedin.com/in/alexmorgan | GitHub: github.com/alexmorgan-ai

Summary
Data Scientist with 6+ years of experience building machine learning products and analytics platforms.
Led cross-functional projects in NLP, recommendation systems, and forecasting for B2B SaaS.
Strong in Python, SQL, scikit-learn, PyTorch, AWS, and MLOps practices.

Technical Skills
Python, SQL, JavaScript, PyTorch, scikit-learn, pandas, NumPy, FastAPI, Docker, Kubernetes,
AWS (S3, Lambda, SageMaker), PostgreSQL, GitHub Actions, Tableau

Experience
Senior Data Scientist, CloudPulse Analytics | 2022-Present
- Built a lead-scoring model that improved qualified pipeline conversion by 27%.
- Developed NLP resume parser and semantic ranking service with sentence-transformers and FAISS.
- Deployed model APIs with FastAPI on AWS, reducing inference latency by 35%.
- Partnered with product and sales teams to define KPI dashboards in Tableau.
""".strip()

SAMPLE_RESUME_TEXT_2 = """
Jordan Lee
Backend Engineer | Remote
Email: jordan.lee@email.com | Phone: +1 312-555-0109
LinkedIn: linkedin.com/in/jordanlee | GitHub: github.com/jordanlee-dev

Summary
Backend engineer with 5 years of experience building scalable API platforms and data workflows.
Focused on Python, FastAPI, PostgreSQL, Docker, and cloud deployment on AWS.

Experience
Backend Engineer, DeltaGrid | 2021-Present
- Built microservices in FastAPI and improved API throughput by 42%.
- Designed PostgreSQL schema optimization reducing query latency by 33%.
- Implemented CI/CD with GitHub Actions and containerized deployments with Docker.
""".strip()

SAMPLE_JD_TEXT = """
Senior AI Engineer - Resume Intelligence Platform

We are hiring a Senior AI Engineer to build intelligent hiring solutions for enterprise customers.
You will design and ship services that parse resumes, extract skills, score candidate-job fit,
and deliver actionable recommendations to recruiters and applicants.

Requirements:
- 4+ years experience in Python and SQL for production applications
- Strong hands-on experience with NLP and machine learning
- Experience with sentence-transformers, vector search, and semantic similarity systems
- Familiarity with scikit-learn, pandas, and model evaluation metrics
- Ability to build APIs with FastAPI or Flask
- Experience deploying on AWS with Docker and CI/CD pipelines
- Strong communication and stakeholder collaboration

Nice to Have:
- Streamlit for internal tooling
- Experience with LLM APIs (Groq, OpenAI, Anthropic)
- Knowledge of ATS optimization and keyword matching workflows
""".strip()

MODEL_OPTIONS = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
LLM_MODEL_OPTIONS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
]


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
            .block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
            .app-hero {
                border: 1px solid rgba(0, 184, 148, 0.25);
                padding: 1rem 1.2rem;
                border-radius: 14px;
                background: linear-gradient(120deg, rgba(0, 184, 148, 0.12), rgba(9, 132, 227, 0.08));
                margin-bottom: 1rem;
            }
            .app-hero h2 {
                margin: 0;
                letter-spacing: 0.2px;
            }
            .app-hero p {
                margin: 0.35rem 0 0 0;
                color: #d8e7f7;
            }
            .pill {
                display: inline-block;
                padding: 0.32rem 0.72rem;
                margin: 0.2rem 0.26rem 0.2rem 0;
                border-radius: 999px;
                border: 1px solid rgba(0, 184, 148, 0.35);
                background-color: rgba(0, 184, 148, 0.14);
                font-size: 0.84rem;
            }
            .muted {
                color: #9AA5B1;
                font-size: 0.9rem;
            }
            .report-card {
                border: 1px solid rgba(255,255,255,0.09);
                border-radius: 16px;
                padding: 16px;
                background: linear-gradient(180deg, rgba(22,28,40,0.98), rgba(15,20,30,0.98));
                box-shadow: 0 12px 24px rgba(0, 0, 0, 0.16);
            }
            .report-title {
                font-size: 1.05rem;
                font-weight: 700;
                color: #EAF5FF;
                margin-bottom: 0.65rem;
            }
            .score-value {
                font-size: 2.3rem;
                font-weight: 800;
                line-height: 1.1;
                color: #F7FBFF;
            }
            .score-value span {
                font-size: 1rem;
                opacity: 0.7;
                margin-left: 4px;
                font-weight: 600;
            }
            .score-sub {
                margin-top: 0.25rem;
                color: #AEBBD0;
                font-size: 0.88rem;
            }
            .score-meta {
                margin-top: 0.55rem;
                color: #90a6c2;
                font-size: 0.82rem;
            }
            .score-sep {
                margin: 12px 0;
                border: 0;
                border-top: 1px solid rgba(255,255,255,0.08);
            }
            .section-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 0.46rem;
                font-size: 0.9rem;
            }
            .section-row .label {
                color: #DCE8F9;
                font-weight: 600;
            }
            .section-sub {
                color: #90a6c2;
                font-size: 0.78rem;
                margin: 0 0 0.42rem 0;
            }
            .badge {
                border-radius: 999px;
                padding: 2px 10px;
                font-size: 0.76rem;
                font-weight: 700;
                border: 1px solid transparent;
                white-space: nowrap;
            }
            .badge.good { background: rgba(16,185,129,0.18); color: #86efac; border-color: rgba(16,185,129,0.35); }
            .badge.warn { background: rgba(245,158,11,0.16); color: #fcd34d; border-color: rgba(245,158,11,0.35); }
            .badge.bad { background: rgba(239,68,68,0.14); color: #fca5a5; border-color: rgba(239,68,68,0.35); }
            .audit-panel {
                border: 1px solid rgba(255,255,255,0.09);
                border-radius: 16px;
                padding: 14px;
                background: linear-gradient(180deg, rgba(20,27,40,0.94), rgba(14,19,30,0.94));
            }
            .audit-item {
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 12px;
                padding: 10px 12px;
                margin-bottom: 10px;
                background: rgba(10, 14, 24, 0.74);
            }
            .audit-item h4 {
                margin: 0 0 4px 0;
                font-size: 0.94rem;
                color: #EAF4FF;
            }
            .audit-item p {
                margin: 0;
                color: #91A4BF;
                font-size: 0.83rem;
                margin-top: 6px;
            }
            .hint-note {
                color: #9ab1cb;
                font-size: 0.82rem;
                margin-top: 0.15rem;
            }
            .simple-subtab-note {
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 10px;
                padding: 0.55rem 0.7rem;
                background: rgba(27, 38, 54, 0.5);
                color: #b7c9df;
                font-size: 0.86rem;
                margin-bottom: 0.6rem;
            }
            .stTabs [data-baseweb="tab-list"] { gap: 0.25rem; }
            .stTabs [data-baseweb="tab"] {
                height: 2.1rem;
                border-radius: 8px 8px 0 0;
                padding-left: 0.65rem;
                padding-right: 0.65rem;
            }
            .stButton > button[kind="primary"] {
                border-radius: 12px;
                border: 0;
                min-height: 3rem;
                font-size: 1rem;
                font-weight: 700;
                letter-spacing: 0.2px;
                color: #f7fbff;
                background: linear-gradient(100deg, #00c6a7, #00a8ff);
                box-shadow: 0 12px 26px rgba(0, 168, 255, 0.24);
                transition: transform 0.14s ease, box-shadow 0.14s ease, filter 0.14s ease;
            }
            .stButton > button[kind="primary"]:hover {
                transform: translateY(-1px);
                filter: brightness(1.02);
                box-shadow: 0 16px 30px rgba(0, 168, 255, 0.28);
            }
            .home-shell {
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 16px;
                background: linear-gradient(140deg, rgba(18, 28, 43, 0.9), rgba(13, 20, 31, 0.95));
                padding: 1rem;
                margin-bottom: 0.95rem;
            }
            .home-title {
                font-size: 1.04rem;
                font-weight: 700;
                color: #eaf4ff;
                margin-bottom: 0.2rem;
            }
            .home-sub {
                color: #9fb3cc;
                font-size: 0.9rem;
                margin-bottom: 0.65rem;
            }
            .home-step {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                margin-bottom: 0.35rem;
                padding: 0.34rem 0.62rem;
                border-radius: 12px;
                border: 1px solid rgba(255,255,255,0.12);
                background: rgba(15, 25, 38, 0.55);
                color: #d6e6fa;
                font-size: 0.8rem;
                font-weight: 600;
            }
            .home-steps {
                display: flex;
                flex-wrap: wrap;
                gap: 0.35rem 0.4rem;
                margin-top: 0.2rem;
            }
            .home-step-num {
                width: 1.2rem;
                height: 1.2rem;
                border-radius: 999px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-size: 0.72rem;
                font-weight: 800;
                color: #dff4ff;
                background: rgba(45, 212, 191, 0.2);
                border: 1px solid rgba(45, 212, 191, 0.45);
            }
            .home-step-label {
                line-height: 1.15;
                display: inline-block;
            }
            .home-ready.good {
                border: 1px solid rgba(52,211,153,0.35);
                border-radius: 10px;
                padding: 0.45rem 0.62rem;
                background: rgba(52,211,153,0.11);
                color: #86efac;
                font-size: 0.87rem;
                margin-top: 0.5rem;
            }
            .home-ready.warn {
                border: 1px solid rgba(245,158,11,0.35);
                border-radius: 10px;
                padding: 0.45rem 0.62rem;
                background: rgba(245,158,11,0.11);
                color: #fcd34d;
                font-size: 0.87rem;
                margin-top: 0.5rem;
            }

            .eh-side-card {
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 18px;
                padding: 1rem;
                background: linear-gradient(180deg, rgba(24,30,43,0.98), rgba(16,22,34,0.98));
                box-shadow: 0 14px 30px rgba(0, 0, 0, 0.2);
                width: 100%;
                box-sizing: border-box;
            }
            .eh-side-title {
                font-size: 1.05rem;
                font-weight: 700;
                color: #EAF4FF;
                margin-bottom: 0.75rem;
            }
            .eh-score-ring {
                width: 128px;
                height: 128px;
                margin: 0 auto;
                border-radius: 999px;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .eh-score-ring-inner {
                width: 96px;
                height: 96px;
                border-radius: 999px;
                display: flex;
                align-items: center;
                justify-content: center;
                flex-direction: column;
                background: rgba(9, 14, 24, 0.95);
                border: 1px solid rgba(255,255,255,0.10);
                color: #F7FBFF;
                font-weight: 800;
                font-size: 1.9rem;
                line-height: 1;
            }
            .eh-score-ring-inner span {
                font-size: 0.8rem;
                color: #9fb2cb;
                font-weight: 600;
                margin-top: 0.25rem;
            }
            .eh-issues {
                text-align: center;
                color: #9fb2cb;
                font-size: 0.86rem;
                margin-top: 0.65rem;
                margin-bottom: 0.75rem;
            }
            .eh-divider {
                border-top: 1px solid rgba(255,255,255,0.09);
                margin: 0.65rem 0 0.85rem 0;
            }
            .eh-cat-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0.45rem 0.55rem;
                border-radius: 10px;
                border: 1px solid transparent;
                margin-bottom: 0.12rem;
            }
            .eh-cat-row.active {
                border-color: rgba(45, 212, 191, 0.35);
                background: rgba(45, 212, 191, 0.08);
            }
            .eh-cat-name {
                color: #d8e8fb;
                font-size: 0.9rem;
                font-weight: 600;
                min-width: 0;
            }
            .eh-cat-meta {
                color: #8ea3bf;
                font-size: 0.76rem;
                margin-bottom: 0.3rem;
                margin-left: 0.58rem;
            }
            .eh-flow-card {
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 14px;
                padding: 0.8rem 0.95rem;
                background: rgba(24, 35, 52, 0.60);
                margin-bottom: 0.65rem;
            }
            .eh-flow-item {
                display: flex;
                align-items: center;
                gap: 0.55rem;
                color: #d6e7fb;
                font-size: 0.92rem;
                margin: 0.2rem 0;
            }
            .eh-flow-item.dim {
                color: #97adc8;
            }
            .eh-flow-dot {
                width: 10px;
                height: 10px;
                border-radius: 999px;
                flex: 0 0 10px;
                display: inline-block;
            }
            .eh-flow-dot.done {
                background: #22c55e;
                box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.22);
            }
            .eh-flow-dot.warn {
                background: #f59e0b;
                box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.2);
            }
            .eh-focus-meta {
                border: 1px solid rgba(255,255,255,0.09);
                border-radius: 10px;
                padding: 0.55rem 0.7rem;
                background: rgba(27, 38, 54, 0.5);
                color: #b7c9df;
                font-size: 0.86rem;
                margin-bottom: 0.6rem;
            }
            .eh-item-title {
                color: #eaf4ff;
                font-size: 0.95rem;
                font-weight: 700;
                margin-bottom: 0.1rem;
            }
            .eh-item-status {
                color: #9fb2cb;
                font-size: 0.80rem;
                margin-bottom: 0.35rem;
            }
            @media (max-width: 768px) {
                .block-container {
                    padding-left: 0.9rem;
                    padding-right: 0.9rem;
                }
                .score-value { font-size: 2rem; }
                .eh-score-ring {
                    width: 110px;
                    height: 110px;
                }
                .eh-score-ring-inner {
                    width: 82px;
                    height: 82px;
                    font-size: 1.5rem;
                }
                .stButton > button[kind="primary"] {
                    min-height: 2.85rem;
                    font-size: 0.95rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_get_secret(key: str) -> str:
    try:
        value = str(st.secrets.get(key, "")).strip()
        if value:
            return value
    except Exception:
        pass
    return str(os.getenv(key, "")).strip()


def render_skill_pills(skills: Sequence[str]) -> None:
    if not skills:
        st.caption("No skills found.")
        return
    html = "".join(f"<span class='pill'>{skill}</span>" for skill in skills)
    st.markdown(html, unsafe_allow_html=True)


def render_copy_suggestions_widget(suggestions: Sequence[str], *, key_suffix: str = "default") -> None:
    lines = [f"{idx}. {text}" for idx, text in enumerate(suggestions, start=1)]
    payload_text = "\n".join(lines) if lines else "No suggestions generated."
    payload_json = json.dumps(payload_text)

    components.html(
        f"""
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
          <button id="copy-btn-{key_suffix}" style="
            background:#00B894; color:#fff; border:0; border-radius:8px;
            padding:8px 12px; cursor:pointer; font-weight:600;">
            Copy All Suggestions
          </button>
          <span id="copy-status-{key_suffix}" style="color:#9AA5B1; font-size:12px;"></span>
        </div>
        <script>
          const text = {payload_json};
          const btn = document.getElementById("copy-btn-{key_suffix}");
          const status = document.getElementById("copy-status-{key_suffix}");
          btn.addEventListener("click", async () => {{
            try {{
              await navigator.clipboard.writeText(text);
              status.textContent = "Copied to clipboard.";
            }} catch (err) {{
              status.textContent = "Clipboard blocked in this browser.";
            }}
          }});
        </script>
        """,
        height=62,
    )


def render_next_steps_checklist(
    missing_skills: Sequence[str],
    target_score: float = 70.0,
    key_prefix: str = "default",
) -> None:
    top_missing = ", ".join(list(missing_skills)[:3]) if missing_skills else "top missing skills"
    st.markdown("#### What To Do Next")
    st.checkbox(
        f"Add measurable evidence bullets for: {top_missing}.",
        key=f"{key_prefix}_next_step_1",
    )
    st.checkbox(
        "Mirror 5 exact JD keywords in your Summary and Skills sections.",
        key=f"{key_prefix}_next_step_2",
    )
    st.checkbox(
        f"Re-run analysis and target at least {target_score:.0f}% overall match.",
        key=f"{key_prefix}_next_step_3",
    )


def _item_status(score: float) -> str:
    if score >= 80:
        return "No issues"
    if score >= 60:
        return "Needs work"
    return "High priority"


def _score_tone(score: float) -> str:
    if score >= 80:
        return "good"
    if score >= 60:
        return "warn"
    return "bad"


def _status_icon(score: float) -> str:
    if score >= 80:
        return "OK"
    if score >= 60:
        return "WARN"
    return "FAIL"


def _quantification_score(results: Dict[str, Any]) -> float:
    bullets = results.get("rewriter", {}).get("original_bullets", []) or []
    if not bullets:
        bullets = [
            line.strip()
            for line in results.get("resume_text", "").splitlines()
            if line.strip().startswith("-")
        ]
    if not bullets:
        return 50.0
    quantified = sum(1 for bullet in bullets if re.search(r"\d|%|\$", bullet))
    return round((quantified / max(len(bullets), 1)) * 100, 2)


def build_interactive_audit(results: Dict[str, Any]) -> Dict[str, Any]:
    score = results["score_details"]
    ats = results["ats_details"]
    requirement_map = results.get("requirement_map", [])

    must_have = [row for row in requirement_map if row.get("category") == "Must-Have"]
    must_covered = [row for row in must_have if row.get("status") == "Covered"]
    must_have_coverage = (
        round((len(must_covered) / max(len(must_have), 1)) * 100, 2) if must_have else 70.0
    )
    quant_score = _quantification_score(results)

    sections = [
        {
            "name": "Content",
            "items": [
                {
                    "name": "ATS Parse Readiness",
                    "score": float(ats.get("section_score", 0)),
                    "hint": "Use clear section headers and clean structure.",
                },
                {
                    "name": "Quantified Impact",
                    "score": quant_score,
                    "hint": "Add metrics like percentages, revenue, or time saved.",
                },
                {
                    "name": "Keyword Match",
                    "score": float(ats.get("keyword_score", 0)),
                    "hint": "Mirror high-value JD phrases in Summary and Skills.",
                },
            ],
        },
        {
            "name": "Sections",
            "items": [
                {
                    "name": "Section Completeness",
                    "score": float(ats.get("section_score", 0)),
                    "hint": "Include Summary, Experience, Skills, Projects, Education.",
                },
                {
                    "name": "Contact Completeness",
                    "score": float(ats.get("contact_score", 0)),
                    "hint": "Keep email, phone, and LinkedIn visible and valid.",
                },
                {
                    "name": "Must-Have Requirement Coverage",
                    "score": must_have_coverage,
                    "hint": "Align bullets to mandatory job requirements.",
                },
            ],
        },
        {
            "name": "ATS Essentials",
            "items": [
                {
                    "name": "ATS Compatibility",
                    "score": float(ats.get("ats_score", 0)),
                    "hint": "Keep formatting ATS-safe and text-selectable.",
                },
                {
                    "name": "Bullet and Layout Quality",
                    "score": float(ats.get("bullet_score", 0)),
                    "hint": "Use concise bullet points and consistent spacing.",
                },
                {
                    "name": "Resume Length Quality",
                    "score": float(ats.get("length_score", 0)),
                    "hint": "Stay concise and prioritize relevant experience.",
                },
            ],
        },
        {
            "name": "Tailoring",
            "items": [
                {
                    "name": "Semantic Alignment",
                    "score": float(score.get("semantic", 0)),
                    "hint": "Match role language and responsibilities from the JD.",
                },
                {
                    "name": "Skill Alignment",
                    "score": float(score.get("skill_alignment", 0)),
                    "hint": "Close high-impact missing skills with project evidence.",
                },
                {
                    "name": "Lexical Match",
                    "score": float(score.get("lexical", 0)),
                    "hint": "Use exact terms from the job description where honest.",
                },
            ],
        },
    ]

    total_issues = 0
    for section in sections:
        section_scores = [item["score"] for item in section["items"]]
        section["score"] = round(sum(section_scores) / max(len(section_scores), 1), 2)
        section["issues"] = sum(1 for value in section_scores if value < 70)
        total_issues += section["issues"]

    return {
        "overall": float(score.get("overall", 0)),
        "sections": sections,
        "issues": total_issues,
    }


def render_interactive_audit_report(results: Dict[str, Any]) -> None:
    audit = build_interactive_audit(results)
    sections = audit["sections"]
    focus_options = [section["name"] for section in sections]
    focus_key = "audit_focus_section"
    if focus_key not in st.session_state or st.session_state[focus_key] not in focus_options:
        st.session_state[focus_key] = focus_options[0]
    selected_section = next((item for item in sections if item["name"] == st.session_state[focus_key]), sections[0])
    left_col, right_col = st.columns([1.0, 2.2], gap="large")

    with left_col:
        ring_value = min(max(audit["overall"], 0.0), 100.0)
        ring_degrees = int(round((ring_value / 100.0) * 360))
        section_rows = []
        for section in sections:
            tone = _score_tone(section["score"])
            active_cls = "active" if section["name"] == selected_section["name"] else ""
            icon = _status_icon(section["score"])
            issue_word = "issue" if section["issues"] == 1 else "issues"
            section_rows.append(
                f'<div class="eh-cat-row {active_cls}">'
                f'<span class="eh-cat-name">{icon} {section["name"]}</span>'
                f'<span class="badge {tone}">{section["score"]:.0f}%</span>'
                f"</div>"
                f'<div class="eh-cat-meta">{section["issues"]} {issue_word} | {_item_status(section["score"])}</div>'
            )

        st.markdown(
            (
                f'<div class="eh-side-card">'
                f'<div class="eh-side-title">Your Score</div>'
                f'<div class="eh-score-ring" style="background: conic-gradient(#22d3ee {ring_degrees}deg, rgba(255,255,255,0.12) 0deg);">'
                f'<div class="eh-score-ring-inner">{ring_value:.0f}<span>/100</span></div>'
                f"</div>"
                f'<div class="eh-issues">{audit["issues"]} issues found</div>'
                f'<div class="eh-divider"></div>'
                f"{''.join(section_rows)}"
                f"</div>"
            ),
            unsafe_allow_html=True,
        )

    with right_col:
        st.markdown("### Interactive Resume Audit")
        st.selectbox("Focus section", focus_options, key=focus_key)
        selected_section = next((item for item in sections if item["name"] == st.session_state[focus_key]), sections[0])

        has_llm_suggestions = bool((results.get("suggestions_payload") or {}).get("suggestions"))
        flow_rows = [
            ("Parsing your resume", True),
            ("Analyzing your experience", True),
            ("Extracting your skills", True),
            ("Generating recommendations", has_llm_suggestions),
        ]
        flow_html = "".join(
            f'<div class="eh-flow-item{" dim" if not done else ""}"><span class="eh-flow-dot {"done" if done else "warn"}"></span>{label}</div>'
            for label, done in flow_rows
        )
        st.markdown(
            f"""
            <div class="eh-flow-card">
                {flow_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="eh-focus-meta">
                <strong>{selected_section["name"]}</strong> audit view:
                score <strong>{selected_section["score"]:.0f}%</strong>,
                status <strong>{_item_status(selected_section["score"])}</strong>,
                issues <strong>{selected_section["issues"]}</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )

        low_items: List[Dict[str, Any]] = []

        for item in selected_section["items"]:
            tone = _score_tone(item["score"])
            if item["score"] < 70:
                low_items.append(item)
            with st.container(border=True):
                c1, c2 = st.columns([4.0, 1.1])
                c1.markdown(f"<div class='eh-item-title'>{item['name']}</div>", unsafe_allow_html=True)
                c1.markdown(
                    f"<div class='eh-item-status'>{_status_icon(item['score'])} {_item_status(item['score'])}</div>",
                    unsafe_allow_html=True,
                )
                c2.markdown(
                    f"<span class='badge {tone}'>{item['score']:.0f}%</span>",
                    unsafe_allow_html=True,
                )
                st.progress(min(max(item["score"] / 100.0, 0.0), 1.0))
                st.markdown(f"<div class='hint-note'>{item['hint']}</div>", unsafe_allow_html=True)

        if low_items:
            st.markdown("#### Priority Fixes")
            for item in low_items[:3]:
                st.markdown(f"- **{item['name']}**: {item['hint']}")

        other_sections = [section for section in sections if section["name"] != selected_section["name"]]
        if other_sections:
            with st.expander("Other sections", expanded=False):
                for section in other_sections:
                    st.markdown(
                        f"- **{section['name']}**: {section['score']:.0f}% ({section['issues']} issues, {_item_status(section['score'])})"
                    )


def markdown_to_pdf_bytes(markdown_text: str) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    _, height = letter
    text_obj = pdf.beginText(40, height - 40)
    text_obj.setFont("Helvetica", 10)
    for line in markdown_text.splitlines():
        wrapped_lines = textwrap.wrap(line, width=105) or [""]
        for wrapped in wrapped_lines:
            if text_obj.getY() <= 40:
                pdf.drawText(text_obj)
                pdf.showPage()
                text_obj = pdf.beginText(40, height - 40)
                text_obj.setFont("Helvetica", 10)
            text_obj.textLine(wrapped)
    pdf.drawText(text_obj)
    pdf.save()
    buffer.seek(0)
    return buffer.read()


def _extract_uploaded_bytes(uploaded_file: Any | None) -> bytes | None:
    if uploaded_file is None:
        return None
    data = uploaded_file.read()
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)
    return data


def _prepare_text(
    uploaded_file: Any | None,
    pasted_text: str | None,
    use_ocr_fallback: bool,
    redact_enabled: bool,
) -> Dict[str, Any]:
    doc = extract_document(uploaded_file, pasted_text=pasted_text, use_ocr_fallback=use_ocr_fallback)
    text = doc.text
    pii_counts = {"email": 0, "phone": 0, "linkedin": 0, "github": 0}
    if redact_enabled:
        text, pii_counts = redact_pii(text)
    return {"text": text, "metadata": doc.metadata, "pii_counts": pii_counts}


def _has_nonzero_privacy_redactions(summary: Dict[str, Any]) -> bool:
    if not isinstance(summary, dict) or not summary:
        return False
    for key in ("resume_pii_redacted", "jd_pii_redacted"):
        counts = summary.get(key, {})
        if isinstance(counts, dict):
            for value in counts.values():
                if isinstance(value, (int, float)) and value > 0:
                    return True
    return False


def run_analysis(
    resume_text: str,
    jd_text: str,
    base_model_name: str,
    auto_multilingual: bool,
    role_template_name: str,
    use_reranker: bool,
    reranker_model: str,
    groq_api_key: str,
    llm_model_name: str,
    include_suggestions: bool = True,
) -> Dict[str, Any]:
    metrics = AnalysisMetrics()

    metrics.start_stage("language_detection")
    resume_language = detect_language(resume_text)
    jd_language = detect_language(jd_text)
    dominant_language = jd_language if jd_language != "unknown" else resume_language
    effective_model = choose_embedding_model(base_model_name, dominant_language, auto_multilingual)
    metrics.end_stage("language_detection")

    role_template = get_role_template(role_template_name)
    custom_weights = get_template_weights(role_template_name, use_reranker)
    custom_skill_pool = get_template_skill_pool(role_template_name)

    metrics.start_stage("model_warmup")
    load_embedding_model(effective_model)
    if use_reranker:
        load_reranker_model(reranker_model)
    metrics.end_stage("model_warmup")

    metrics.start_stage("scoring")
    score_details = calculate_match_score(
        resume_text=resume_text,
        jd_text=jd_text,
        model_name=effective_model,
        use_reranker=use_reranker,
        reranker_model=reranker_model,
        custom_weights=custom_weights,
        custom_skills=custom_skill_pool,
    )
    metrics.end_stage("scoring")

    metrics.start_stage("skills")
    skill_details = analyze_skill_alignment(
        resume_text=resume_text,
        jd_text=jd_text,
        custom_skills=custom_skill_pool,
    )
    metrics.end_stage("skills")

    metrics.start_stage("section_scoring")
    section_scores = calculate_section_scores(
        resume_text=resume_text,
        jd_text=jd_text,
        model_name=effective_model,
        use_reranker=use_reranker,
        reranker_model=reranker_model,
        custom_weights=custom_weights,
        custom_skills=custom_skill_pool,
    )
    metrics.end_stage("section_scoring")

    metrics.start_stage("ats_keyword")
    ats_details = calculate_ats_compatibility(resume_text, jd_text, skill_details)
    keyword_rows = calculate_keyword_density(resume_text, jd_text, top_n=18)
    metrics.end_stage("ats_keyword")

    metrics.start_stage("requirements")
    requirements = extract_jd_requirements(jd_text)
    requirement_map = map_requirements_to_evidence(resume_text, requirements)
    metrics.end_stage("requirements")

    metrics.start_stage("explainability")
    explanation = build_match_explanation(
        resume_text=resume_text,
        jd_text=jd_text,
        score_details=score_details,
        skill_details=skill_details,
        top_n_keywords=18,
    )
    strong_points = build_strong_points(resume_text, score_details, skill_details)
    confidence = compute_confidence(score_details)
    fairness = analyze_bias_risks(resume_text)
    metrics.end_stage("explainability")

    metrics.start_stage("rewriter")
    target_keywords = [row["keyword"] for row in keyword_rows][:10]
    rewrite_payload = rewrite_resume_bullets(resume_text, target_keywords=target_keywords)
    rewrite_diff_html = render_bullet_diff_html(
        rewrite_payload.get("original", []),
        rewrite_payload.get("rewritten", []),
    )
    metrics.end_stage("rewriter")

    metrics.start_stage("planner_draft")
    plan_30_day = build_30_day_plan(skill_details["missing_skills"], role_template_name)
    tailored_resume_draft = build_tailored_resume_draft(
        resume_text=resume_text,
        jd_text=jd_text,
        role_template=role_template_name,
        matching_skills=format_skill_list(skill_details["matching_skills"]),
        missing_skills=format_skill_list(skill_details["missing_skills"]),
    )
    metrics.end_stage("planner_draft")

    if include_suggestions:
        metrics.start_stage("llm_suggestions")
        suggestions_payload = generate_improvement_suggestions(
            resume_text=resume_text,
            jd_text=jd_text,
            match_score=float(score_details["overall"]),
            missing_skills=skill_details["missing_skills"],
            matching_skills=skill_details["matching_skills"],
            groq_api_key=groq_api_key,
            model_name=llm_model_name,
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

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "role_template": role_template_name,
        "role_template_description": role_template.get("description", ""),
        "language": {
            "resume": resume_language,
            "jd": jd_language,
            "effective_model": effective_model,
        },
        "score_details": score_details,
        "skill_details": skill_details,
        "section_scores": section_scores,
        "requirements": requirements,
        "requirement_map": requirement_map,
        "ats_details": ats_details,
        "keyword_rows": keyword_rows,
        "suggestions_payload": suggestions_payload,
        "strong_points": strong_points,
        "explanation": explanation,
        "confidence": confidence,
        "fairness": fairness,
        "rewriter": {
            "original_bullets": rewrite_payload.get("original", []),
            "rewritten_bullets": rewrite_payload.get("rewritten", []),
            "diff_html": rewrite_diff_html,
        },
        "planner_30_day": plan_30_day,
        "tailored_resume_draft": tailored_resume_draft,
        "observability": metrics.to_dict(),
        "resume_text": resume_text,
        "jd_text": jd_text,
    }


def run_resume_health_analysis(
    resume_text: str,
    role_hint: str,
    groq_api_key: str,
    llm_model_name: str,
) -> Dict[str, Any]:
    metrics = AnalysisMetrics()

    metrics.start_stage("resume_health_scoring")
    sections_presence = detect_sections(resume_text)
    section_hits = sum(1 for present in sections_presence.values() if present)
    section_score = round((section_hits / max(len(sections_presence), 1)) * 100, 2)

    contact_patterns = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "phone": r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b",
        "linkedin": r"(?:linkedin\.com/in/|linkedin\.com/company/)",
        "github": r"(?:github\.com/)",
    }
    contact_hits = sum(
        1 for pattern in contact_patterns.values() if re.search(pattern, resume_text or "", flags=re.IGNORECASE)
    )
    contact_score = round((contact_hits / max(len(contact_patterns), 1)) * 100, 2)

    word_count = len(re.findall(r"\b\w+\b", resume_text))
    if 350 <= word_count <= 900:
        length_score = 100.0
    elif 250 <= word_count <= 1100:
        length_score = 78.0
    elif word_count > 0:
        length_score = 50.0
    else:
        length_score = 0.0

    bullet_lines = len(re.findall(r"^\s*[-*]\s+", resume_text or "", flags=re.MULTILINE))
    bullet_score = 100.0 if bullet_lines >= 4 else (75.0 if bullet_lines >= 2 else 45.0)

    quantification_score = _quantification_score({"resume_text": resume_text})
    resume_skills = sorted(extract_skills_from_text(resume_text))
    skill_spread_score = min(len(resume_skills) * 8, 100)

    overall = round(
        (0.24 * section_score)
        + (0.22 * contact_score)
        + (0.18 * length_score)
        + (0.14 * bullet_score)
        + (0.16 * quantification_score)
        + (0.06 * skill_spread_score),
        2,
    )
    metrics.end_stage("resume_health_scoring")

    fixes: List[str] = []
    quality_gaps: List[str] = []
    if section_score < 80:
        fixes.append("Add standard headings: Summary, Experience, Skills, Projects, Education.")
        quality_gaps.append("section structure")
    if contact_score < 75:
        fixes.append("Include complete contact details: email, phone, LinkedIn, and GitHub.")
        quality_gaps.append("contact details")
    if quantification_score < 60:
        fixes.append("Rewrite bullets with measurable impact (percentages, time, revenue, scale).")
        quality_gaps.append("quantified impact")
    if bullet_score < 70:
        fixes.append("Use concise bullet points in experience sections for ATS readability.")
        quality_gaps.append("bullet formatting")
    if length_score < 70:
        fixes.append("Tighten resume length to stay focused and role-relevant.")
        quality_gaps.append("resume length")
    if skill_spread_score < 40:
        fixes.append("Surface core technical skills clearly in a dedicated Skills section.")
        quality_gaps.append("skill visibility")
    if not fixes:
        fixes.append("Resume structure is strong. Prioritize role tailoring and quantified wins.")

    strong_points: List[str] = []
    if section_score >= 80:
        strong_points.append("Section structure is ATS-friendly and easy to parse.")
    if contact_score >= 75:
        strong_points.append("Contact details are complete and recruiter-ready.")
    if quantification_score >= 70:
        strong_points.append("Experience bullets show measurable impact.")
    if bullet_score >= 70:
        strong_points.append("Bullet formatting supports quick recruiter scanning.")
    if skill_spread_score >= 50:
        strong_points.append(f"Strong visible skill coverage: {', '.join(resume_skills[:6])}.")
    if not strong_points:
        strong_points.append("Good foundation; focused edits can raise interview readiness quickly.")

    suggestion_context = (
        f"Resume-only quality audit for role target: {role_hint}."
        if role_hint.strip()
        else "Resume-only quality audit. Improve ATS readiness, impact, and clarity."
    )
    metrics.start_stage("llm_suggestions")
    suggestions_payload = generate_improvement_suggestions(
        resume_text=resume_text,
        jd_text=suggestion_context,
        match_score=overall,
        missing_skills=quality_gaps,
        matching_skills=resume_skills[:8],
        groq_api_key=groq_api_key,
        model_name=llm_model_name,
    )
    usage = suggestions_payload.get("usage", {})
    metrics.set_llm_usage(
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
        estimated_cost_usd=float(usage.get("estimated_cost_usd", 0.0)),
    )
    metrics.end_stage("llm_suggestions")

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "mode": "resume_health",
        "health_metrics": {
            "overall": overall,
            "section_score": round(section_score, 2),
            "contact_score": round(contact_score, 2),
            "length_score": round(length_score, 2),
            "bullet_score": round(bullet_score, 2),
            "quantification_score": round(quantification_score, 2),
            "skill_spread_score": round(skill_spread_score, 2),
            "word_count": word_count,
        },
        "sections_presence": sections_presence,
        "resume_skills": resume_skills,
        "quality_gaps": quality_gaps,
        "fixes": fixes,
        "strong_points": strong_points,
        "suggestions_payload": suggestions_payload,
        "observability": metrics.to_dict(),
        "resume_text": resume_text,
        "role_hint": role_hint.strip(),
    }


def render_resume_health_results(results: Dict[str, Any], resume_pdf_bytes: bytes | None = None) -> None:
    hm = results.get("health_metrics", {})
    overall = float(hm.get("overall", 0))
    suggestions_payload = results.get("suggestions_payload", {})
    suggestions = suggestions_payload.get("suggestions", [])
    section_score = float(hm.get("section_score", 0))
    contact_score = float(hm.get("contact_score", 0))
    quantification_score = float(hm.get("quantification_score", 0))
    bullet_score = float(hm.get("bullet_score", 0))
    sections_presence = results.get("sections_presence", {})
    missing_sections = sum(1 for present in sections_presence.values() if not present)

    score_sections = [
        {
            "name": "Section Quality",
            "score": section_score,
            "issues": missing_sections if missing_sections > 0 else (1 if section_score < 70 else 0),
            "hint": "Use standard headings: Summary, Experience, Skills, Projects, Education.",
        },
        {
            "name": "Contact Completeness",
            "score": contact_score,
            "issues": 0 if contact_score >= 95 else 1,
            "hint": "Keep email, phone, and LinkedIn clear in the header.",
        },
        {
            "name": "Quantified Impact",
            "score": quantification_score,
            "issues": 0 if quantification_score >= 70 else 1,
            "hint": "Add numbers in bullets: %, time saved, revenue, or scale.",
        },
        {
            "name": "Bullet Quality",
            "score": bullet_score,
            "issues": 0 if bullet_score >= 70 else 1,
            "hint": "Use concise action-result bullets for ATS and recruiter readability.",
        },
    ]
    total_issues = sum(section["issues"] for section in score_sections)

    st.markdown("### Resume Health Report")
    left_col, right_col = st.columns([1.0, 2.2], gap="large")

    with left_col:
        ring_value = min(max(overall, 0.0), 100.0)
        ring_degrees = int(round((ring_value / 100.0) * 360))
        section_rows: List[str] = []
        for section in score_sections:
            tone = _score_tone(section["score"])
            icon = _status_icon(section["score"])
            issue_word = "issue" if section["issues"] == 1 else "issues"
            section_rows.append(
                f'<div class="eh-cat-row">'
                f'<span class="eh-cat-name">{icon} {section["name"]}</span>'
                f'<span class="badge {tone}">{section["score"]:.0f}%</span>'
                f"</div>"
                f'<div class="eh-cat-meta">{section["issues"]} {issue_word} | {_item_status(section["score"])}</div>'
            )

        st.markdown(
            (
                f'<div class="eh-side-card">'
                f'<div class="eh-side-title">Your Score</div>'
                f'<div class="eh-score-ring" style="background: conic-gradient(#22d3ee {ring_degrees}deg, rgba(255,255,255,0.12) 0deg);">'
                f'<div class="eh-score-ring-inner">{ring_value:.0f}<span>/100</span></div>'
                f"</div>"
                f'<div class="eh-issues">{total_issues} issues found</div>'
                f'<div class="eh-divider"></div>'
                f"{''.join(section_rows)}"
                f"</div>"
            ),
            unsafe_allow_html=True,
        )

    with right_col:
        flow_rows = [
            ("Parsing your resume", True),
            ("Reviewing structure and sections", True),
            ("Checking skills and impact quality", True),
            ("Generating recommendations", bool(suggestions)),
        ]
        flow_html = "".join(
            f'<div class="eh-flow-item{" dim" if not done else ""}"><span class="eh-flow-dot {"done" if done else "warn"}"></span>{label}</div>'
            for label, done in flow_rows
        )
        st.markdown(
            f"""
            <div class="eh-flow-card">
                {flow_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        metric_cols = st.columns(4)
        metric_cols[0].metric("Resume Score", f"{overall:.2f}%")
        metric_cols[1].metric("Section Quality", f"{section_score:.2f}%")
        metric_cols[2].metric("Contact", f"{contact_score:.2f}%")
        metric_cols[3].metric("Impact", f"{quantification_score:.2f}%")
        st.progress(min(max(overall / 100, 0.0), 1.0))
        st.caption(
            f"Word Count: {hm.get('word_count', 0)} | Skill Visibility: {hm.get('skill_spread_score', 0):.2f}% | Bullet Quality: {bullet_score:.2f}%"
        )

        for section in score_sections:
            tone = _score_tone(section["score"])
            with st.container(border=True):
                c1, c2 = st.columns([4.0, 1.1])
                c1.markdown(f"<div class='eh-item-title'>{section['name']}</div>", unsafe_allow_html=True)
                c1.markdown(
                    f"<div class='eh-item-status'>{_status_icon(section['score'])} {_item_status(section['score'])}</div>",
                    unsafe_allow_html=True,
                )
                c2.markdown(
                    f"<span class='badge {tone}'>{section['score']:.0f}%</span>",
                    unsafe_allow_html=True,
                )
                st.progress(min(max(section["score"] / 100.0, 0.0), 1.0))
                st.markdown(f"<div class='hint-note'>{section['hint']}</div>", unsafe_allow_html=True)

    resume_skills = format_skill_list(results.get("resume_skills", []))
    quality_gaps = format_skill_list(results.get("quality_gaps", []))
    fixes = results.get("fixes", [])
    strong_points = results.get("strong_points", [])
    role_hint = (results.get("role_hint") or "General").strip() or "General"
    observability = results.get("observability", {})
    resume_text = results.get("resume_text", "")

    tabs = st.tabs(
        [
            "Overview",
            "Section Scores",
            "JD Requirements",
            "Skills & ATS",
            "Rewriter Diff",
            "Action Plan + Tailored Draft",
            "Explainability + Fairness",
            "PDF + Highlights",
            "Observability",
        ]
    )

    with tabs[0]:
        st.markdown("#### Top Fixes")
        for idx, fix in enumerate(fixes, start=1):
            st.markdown(f"{idx}. {fix}")
        st.markdown("#### Strong Points")
        for point in strong_points:
            st.markdown(f"- {point}")

        st.markdown("#### Personalized Suggestions")
        provider = suggestions_payload.get("provider", "unknown")
        st.caption(f"Suggestion source: `{provider}`")
        if provider == "rule-based":
            diagnostics = suggestions_payload.get("diagnostics", {})
            if diagnostics:
                with st.expander("LLM Diagnostics", expanded=False):
                    st.json(diagnostics)
        for idx, suggestion in enumerate(suggestions or ["No suggestions generated."], start=1):
            st.markdown(f"{idx}. {suggestion}")

        render_copy_suggestions_widget(suggestions or ["No suggestions generated."], key_suffix="resume_health_full")
        render_next_steps_checklist(quality_gaps, target_score=75.0, key_prefix="resume_health")

    with tabs[1]:
        section_df = pd.DataFrame(
            [
                {
                    "section": section["name"],
                    "score": round(float(section["score"]), 2),
                    "issues": int(section["issues"]),
                    "status": _item_status(float(section["score"])),
                }
                for section in score_sections
            ]
        )
        st.dataframe(section_df, use_container_width=True, hide_index=True)
        st.bar_chart(section_df.set_index("section")[["score"]])

        presence = results.get("sections_presence", {})
        if presence:
            st.markdown("#### Section Presence")
            rows = [{"section": key.title(), "present": "Yes" if value else "No"} for key, value in presence.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tabs[2]:
        st.markdown("#### Requirement-Style Gap View (Resume Health)")
        req_rows = [{"requirement": gap, "category": "Resume Quality Gap", "status": "Needs Work"} for gap in quality_gaps]
        req_df = pd.DataFrame(req_rows)
        if req_df.empty:
            st.caption("No critical gaps detected in resume-only mode.")
        else:
            st.dataframe(req_df, use_container_width=True, hide_index=True)
            st.metric("Gap Coverage", f"{max(0.0, 100.0 - (len(req_rows) * 12.5)):.2f}%")
        st.caption("For strict JD requirement coverage, run JD Match mode.")

    with tabs[3]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Detected Skills")
            render_skill_pills(resume_skills)
        with c2:
            st.markdown("#### Missing Focus Areas")
            render_skill_pills(quality_gaps)

        st.markdown("#### ATS Components")
        ats_proxy = {
            "Keyword alignment": float(hm.get("skill_spread_score", 0)),
            "Section completeness": section_score,
            "Contact completeness": contact_score,
            "Length quality": float(hm.get("length_score", 0)),
            "Bullet formatting": bullet_score,
        }
        for label, value in ats_proxy.items():
            st.caption(label)
            st.progress(min(max(value / 100, 0.0), 1.0))

        st.markdown("#### ATS Tips")
        for tip in fixes[:5]:
            st.markdown(f"- {tip}")

    with tabs[4]:
        st.markdown("#### Rewriter Diff")
        st.info("Resume Health mode does not run JD-based bullet rewrite diff.")
        st.caption("Use JD Match mode to see original vs rewritten bullets and interactive diff.")

    with tabs[5]:
        st.markdown("#### 30-Day Gap-to-Action Plan")
        plan_rows = build_30_day_plan(quality_gaps, role_hint)
        st.dataframe(pd.DataFrame(plan_rows), use_container_width=True, hide_index=True)

        st.markdown("#### Tailored Resume Draft (Resume Health)")
        summary_skills = ", ".join(resume_skills[:10]) if resume_skills else "Add role-relevant skills"
        draft_md = (
            f"# Tailored Resume Draft ({role_hint})\n\n"
            "## Professional Summary\n"
            "Results-driven professional focused on measurable impact, ATS-friendly structure, and role alignment.\n\n"
            "## Core Skills\n"
            f"{summary_skills}\n\n"
            "## Priority Improvements\n"
            + "\n".join(f"- {fix}" for fix in fixes[:6])
        )
        st.text_area("Tailored Draft (ATS-safe)", value=draft_md, height=260)
        draft_pdf = markdown_to_pdf_bytes(draft_md)
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Download Tailored Draft (Markdown)",
                data=draft_md,
                file_name="resume_health_tailored_draft.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "Download Tailored Draft (PDF)",
                data=draft_pdf,
                file_name="resume_health_tailored_draft.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        report_md = build_suggestions_markdown(
            match_score=overall,
            matching_skills=resume_skills[:10],
            missing_skills=quality_gaps,
            strong_points=strong_points,
            suggestions=suggestions or ["Prioritize measurable impact and ATS-safe formatting."],
            ats_score=float(hm.get("section_score", 0)),
        )
        report_pdf = markdown_to_pdf_bytes(report_md)
        st.download_button(
            "Download Resume Health Report (PDF)",
            data=report_pdf,
            file_name="resume_health_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    with tabs[6]:
        st.markdown("#### Feature Contribution Breakdown")
        contribution_rows = [
            {"feature": "Section Quality", "weight": 0.28, "score": section_score},
            {"feature": "Contact Completeness", "weight": 0.20, "score": contact_score},
            {"feature": "Quantified Impact", "weight": 0.22, "score": quantification_score},
            {"feature": "Bullet Quality", "weight": 0.15, "score": bullet_score},
            {"feature": "Skill Visibility", "weight": 0.15, "score": float(hm.get("skill_spread_score", 0))},
        ]
        st.dataframe(pd.DataFrame(contribution_rows), use_container_width=True, hide_index=True)

        st.markdown("#### Evidence")
        for point in strong_points:
            st.markdown(f"- {point}")
        for fix in fixes[:3]:
            st.markdown(f"- Improvement signal: {fix}")

        st.markdown("#### Fairness and Bias Checks")
        fairness = analyze_bias_risks(resume_text)
        if fairness["risk_count"] > 0:
            st.warning(f"Detected {fairness['risk_count']} potential fairness/compliance flags.")
            for finding in fairness["findings"]:
                st.markdown(f"- {finding}")
        else:
            st.success("No major fairness risks detected.")
        st.markdown("#### Recommendations")
        for rec in fairness["recommendations"]:
            st.markdown(f"- {rec}")

    with tabs[7]:
        render_pdf_preview(resume_pdf_bytes, "Resume PDF Preview")

        keyword_options = sorted(set(resume_skills + quality_gaps))
        if not keyword_options:
            keyword_options = ["python"]

        selected_keyword = st.selectbox("Keyword Evidence", keyword_options, key="resume_health_keyword")
        snippets = highlight_keyword_snippets(resume_text, selected_keyword)
        st.markdown("#### Keyword Evidence in Resume")
        st.caption("These are extracted text snippets where the selected keyword appears in your resume.")
        if snippets:
            for snippet in snippets:
                st.markdown(snippet, unsafe_allow_html=True)
        else:
            st.caption("No evidence snippet found for selected keyword.")

    with tabs[8]:
        st.markdown("#### Latency Metrics")
        timing_df = pd.DataFrame(observability.get("stage_timings", []))
        if not timing_df.empty:
            st.dataframe(timing_df, use_container_width=True, hide_index=True)
        st.metric("Total Latency", f"{observability.get('total_elapsed_ms', 0)} ms")
        st.markdown("#### LLM Usage")
        u1, u2, u3 = st.columns(3)
        u1.metric("Prompt Tokens", observability.get("llm_prompt_tokens", 0))
        u2.metric("Completion Tokens", observability.get("llm_completion_tokens", 0))
        u3.metric("Estimated Cost (USD)", observability.get("llm_estimated_cost_usd", 0.0))


def run_batch_analysis(
    resume_files: Sequence[Any],
    jd_text: str,
    base_model_name: str,
    auto_multilingual: bool,
    role_template_name: str,
    use_reranker: bool,
    reranker_model: str,
    use_ocr_fallback: bool,
    redact_enabled: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    progress = st.progress(0, text="Starting batch analysis...")
    total = max(len(resume_files), 1)

    for index, resume_file in enumerate(resume_files, start=1):
        prep = _prepare_text(
            uploaded_file=resume_file,
            pasted_text=None,
            use_ocr_fallback=use_ocr_fallback,
            redact_enabled=redact_enabled,
        )
        resume_text = prep["text"]
        if not resume_text.strip():
            continue

        result = run_analysis(
            resume_text=resume_text,
            jd_text=jd_text,
            base_model_name=base_model_name,
            auto_multilingual=auto_multilingual,
            role_template_name=role_template_name,
            use_reranker=use_reranker,
            reranker_model=reranker_model,
            groq_api_key="",
            llm_model_name="",
            include_suggestions=False,
        )

        score = result["score_details"]
        skills = result["skill_details"]
        ats = result["ats_details"]
        confidence = result["confidence"]
        missing_count = len(skills["missing_skills"])

        if score["overall"] >= 78 and ats["ats_score"] >= 70 and missing_count <= 3:
            reason = "High fit: strong score, ATS readiness, and low skill gaps."
        elif score["overall"] >= 65:
            reason = "Moderate fit: shortlist after targeted screening."
        else:
            reason = "Lower fit: major gaps relative to JD requirements."

        rows.append(
            {
                "resume_name": getattr(resume_file, "name", f"resume_{index}.pdf"),
                "overall_match": score["overall"],
                "calibrated_overall": confidence["calibrated_overall"],
                "confidence_band": confidence["band"],
                "semantic": score["semantic"],
                "lexical": score["lexical"],
                "skill_alignment": score["skill_alignment"],
                "ats_score": ats["ats_score"],
                "missing_count": missing_count,
                "matched_skills": ", ".join(skills["matching_skills"][:8]),
                "missing_skills": ", ".join(skills["missing_skills"][:8]),
                "shortlist_reason": reason,
            }
        )
        progress.progress(int((index / total) * 100), text=f"Processed {index}/{total} resumes")

    progress.progress(100, text="Batch analysis complete.")
    return rows

def render_single_results(
    results: Dict[str, Any],
    resume_pdf_bytes: bytes | None,
    jd_pdf_bytes: bytes | None,
    simple_mode: bool = False,
) -> None:
    score = results["score_details"]
    skills = results["skill_details"]
    ats = results["ats_details"]
    keyword_rows = results["keyword_rows"]
    suggestions_payload = results["suggestions_payload"]
    suggestions = suggestions_payload.get("suggestions", [])
    strong_points = results["strong_points"]
    explanation = results["explanation"]
    observability = results["observability"]
    confidence = results["confidence"]
    fairness = results["fairness"]
    section_scores = results["section_scores"]
    requirement_map = results["requirement_map"]
    planner_30_day = results["planner_30_day"]
    tailored_draft = results["tailored_resume_draft"]

    st.markdown("### Match Overview")
    if simple_mode:
        metric_cols = st.columns(4)
        metric_cols[0].metric("Overall Match", f"{score['overall']}%")
        metric_cols[1].metric("ATS Score", f"{ats['ats_score']}%")
        metric_cols[2].metric("Skill Alignment", f"{score['skill_alignment']}%")
        metric_cols[3].metric("Confidence", f"{confidence['confidence_pct']}%")
    else:
        metric_cols = st.columns(4)
        metric_cols[0].metric("Overall Match", f"{score['overall']}%")
        metric_cols[1].metric("ATS Score", f"{ats['ats_score']}%")
        metric_cols[2].metric("Skill Alignment", f"{score['skill_alignment']}%")
        metric_cols[3].metric("Confidence", f"{confidence['confidence_pct']}%")
        st.caption(
            f"Calibrated: **{confidence['calibrated_overall']}%** | "
            f"Confidence Band: **{confidence['band']}**"
        )
    st.progress(min(max(score["overall"] / 100, 0.0), 1.0))

    language = results.get("language", {})
    if simple_mode:
        st.caption(
            f"Role Template: **{results.get('role_template', 'General')}** | "
            f"Model: `{language.get('effective_model', 'n/a')}`"
        )
    else:
        st.caption(
            f"Detected Languages -> Resume: {language_label(language.get('resume', 'unknown'))}, "
            f"JD: {language_label(language.get('jd', 'unknown'))}; "
            f"Model Used: `{language.get('effective_model', 'n/a')}`; "
            f"Role Template: **{results.get('role_template', 'General')}**"
        )

    tabs = st.tabs(
        [
            "Overview",
            "Section Scores",
            "JD Requirements",
            "Skills & ATS",
            "Rewriter Diff",
            "Action Plan + Tailored Draft",
            "Explainability + Fairness",
            "PDF + Highlights",
            "Observability",
        ]
    )

    with tabs[0]:
        render_interactive_audit_report(results)
        st.markdown("#### Strong Points")
        for point in strong_points:
            st.markdown(f"- {point}")
        st.markdown("#### Personalized Suggestions")
        provider = suggestions_payload.get("provider", "unknown")
        st.caption(f"Suggestion source: `{provider}`")
        if provider == "rule-based":
            st.warning("LLM call failed for this run. Showing fallback suggestions.")
            diagnostics = suggestions_payload.get("diagnostics", {})
            if diagnostics:
                with st.expander("LLM Diagnostics", expanded=False):
                    st.json(diagnostics)
        for idx, suggestion in enumerate(suggestions or ["No suggestions generated in this mode."], start=1):
            st.markdown(f"{idx}. {suggestion}")
        render_copy_suggestions_widget(
            suggestions or ["No suggestions generated in this mode."],
            key_suffix="simple" if simple_mode else "advanced",
        )
        render_next_steps_checklist(
            format_skill_list(skills["missing_skills"]),
            key_prefix="simple" if simple_mode else "advanced",
        )

    with tabs[1]:
        section_df = pd.DataFrame(section_scores)
        if section_df.empty:
            st.caption("No section-level scores available.")
        else:
            st.dataframe(section_df, use_container_width=True, hide_index=True)
            st.bar_chart(section_df.set_index("section")[["overall"]])

    with tabs[2]:
        req_df = pd.DataFrame(requirement_map)
        if req_df.empty:
            st.caption("No explicit JD requirements extracted.")
        else:
            st.dataframe(req_df, use_container_width=True, hide_index=True)
            must_cov = (
                (req_df[req_df["category"] == "Must-Have"]["status"] == "Covered").mean() * 100
                if (req_df["category"] == "Must-Have").any()
                else 0
            )
            st.metric("Must-Have Coverage", f"{must_cov:.2f}%")

    with tabs[3]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Matching Skills")
            render_skill_pills(format_skill_list(skills["matching_skills"]))
            st.markdown("#### Resume Skills")
            render_skill_pills(format_skill_list(skills["resume_skills"]))
        with c2:
            st.markdown("#### Missing Skills")
            render_skill_pills(format_skill_list(skills["missing_skills"]))
            st.markdown("#### JD Skills")
            render_skill_pills(format_skill_list(skills["jd_skills"]))

        st.markdown("#### ATS Components")
        for label, key in [
            ("Keyword alignment", "keyword_score"),
            ("Section completeness", "section_score"),
            ("Contact completeness", "contact_score"),
            ("Length quality", "length_score"),
            ("Bullet formatting", "bullet_score"),
        ]:
            st.caption(label)
            st.progress(min(max(ats[key] / 100, 0.0), 1.0))
        st.markdown("#### ATS Tips")
        for tip in ats["tips"]:
            st.markdown(f"- {tip}")

        keyword_df = pd.DataFrame(keyword_rows)
        st.markdown("#### Keyword Density")
        if keyword_df.empty:
            st.caption("No keywords detected.")
        else:
            st.dataframe(keyword_df, use_container_width=True, hide_index=True)

    with tabs[4]:
        rewrite = results["rewriter"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Original Bullets")
            for bullet in rewrite.get("original_bullets", []):
                st.markdown(f"- {bullet}")
        with c2:
            st.markdown("#### Rewritten Bullets")
            for bullet in rewrite.get("rewritten_bullets", []):
                st.markdown(f"- {bullet}")
        st.markdown("#### Interactive Diff")
        diff_rows = max(
            len(rewrite.get("original_bullets", []) or []),
            len(rewrite.get("rewritten_bullets", []) or []),
            1,
        )
        diff_height = min(1100, max(340, 140 + diff_rows * 95))
        components.html(rewrite.get("diff_html", "<p>No diff available.</p>"), height=diff_height, scrolling=True)

    with tabs[5]:
        st.markdown("#### 30-Day Gap-to-Action Plan")
        plan_df = pd.DataFrame(planner_30_day)
        st.dataframe(plan_df, use_container_width=True, hide_index=True)

        st.markdown("#### One-Click Tailored Resume Draft")
        st.text_area("Tailored Draft (ATS-safe)", value=tailored_draft, height=320)
        draft_pdf = markdown_to_pdf_bytes(tailored_draft)
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Download Tailored Draft (Markdown)",
                data=tailored_draft,
                file_name="tailored_resume_draft.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "Download Tailored Draft (PDF)",
                data=draft_pdf,
                file_name="tailored_resume_draft.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        report_md = build_suggestions_markdown(
            match_score=score["overall"],
            matching_skills=format_skill_list(skills["matching_skills"]),
            missing_skills=format_skill_list(skills["missing_skills"]),
            strong_points=strong_points,
            suggestions=suggestions or ["Use the planner to close top skill gaps."],
            ats_score=ats["ats_score"],
        )
        report_pdf = markdown_to_pdf_bytes(report_md)
        st.download_button(
            "Download Improvement Report (PDF)",
            data=report_pdf,
            file_name="improvement_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    with tabs[6]:
        st.markdown("#### Feature Contribution Breakdown")
        contribution_df = pd.DataFrame(explanation["feature_contributions"])
        if not contribution_df.empty:
            st.dataframe(contribution_df, use_container_width=True, hide_index=True)
        st.markdown("#### Evidence")
        for point in explanation["evidence_points"]:
            st.markdown(f"- {point}")
        st.markdown("#### Fairness and Bias Checks")
        if fairness["risk_count"] > 0:
            st.warning(f"Detected {fairness['risk_count']} potential fairness/compliance flags.")
            for finding in fairness["findings"]:
                st.markdown(f"- {finding}")
        else:
            st.success("No major fairness risks detected.")
        st.markdown("#### Recommendations")
        for rec in fairness["recommendations"]:
            st.markdown(f"- {rec}")

    with tabs[7]:
        preview_c1, preview_c2 = st.columns(2)
        with preview_c1:
            render_pdf_preview(resume_pdf_bytes, "Resume PDF Preview")
        with preview_c2:
            render_pdf_preview(jd_pdf_bytes, "JD PDF Preview")

        keyword_options = sorted(
            set(explanation.get("matched_keywords", []) + explanation.get("missing_keywords", []))
        )
        if not keyword_options:
            keyword_options = [row["keyword"] for row in keyword_rows][:12]

        selected_keyword = st.selectbox("Keyword Evidence", keyword_options or ["python"])
        st.markdown("#### Keyword Evidence in Resume")
        st.caption("These snippets show where the selected keyword appears in your resume text.")
        resume_snippets = highlight_keyword_snippets(results.get("resume_text", ""), selected_keyword)
        if resume_snippets:
            for snippet in resume_snippets:
                st.markdown(snippet, unsafe_allow_html=True)
        else:
            st.caption("No resume evidence found for selected keyword.")

        st.markdown("#### Keyword Evidence in Job Description")
        st.caption("These snippets show where the selected keyword appears in the job description.")
        jd_snippets = highlight_keyword_snippets(results.get("jd_text", ""), selected_keyword)
        if jd_snippets:
            for snippet in jd_snippets:
                st.markdown(snippet, unsafe_allow_html=True)
        else:
            st.caption("No job-description evidence found for selected keyword.")

    with tabs[8]:
        st.markdown("#### Latency Metrics")
        timing_df = pd.DataFrame(observability.get("stage_timings", []))
        if not timing_df.empty:
            st.dataframe(timing_df, use_container_width=True, hide_index=True)
        st.metric("Total Latency", f"{observability.get('total_elapsed_ms', 0)} ms")
        st.markdown("#### LLM Usage")
        u1, u2, u3 = st.columns(3)
        u1.metric("Prompt Tokens", observability.get("llm_prompt_tokens", 0))
        u2.metric("Completion Tokens", observability.get("llm_completion_tokens", 0))
        u3.metric("Estimated Cost (USD)", observability.get("llm_estimated_cost_usd", 0.0))


def render_batch_results(batch_rows: Sequence[Dict[str, Any]]) -> None:
    if not batch_rows:
        st.warning("No valid resumes were processed.")
        return

    full_df = pd.DataFrame(batch_rows).sort_values(by="overall_match", ascending=False).reset_index(drop=True)
    st.markdown("### Recruiter Batch Dashboard")

    f1, f2, f3 = st.columns(3)
    min_overall = f1.slider("Min Overall Match", 0, 100, 70)
    min_ats = f2.slider("Min ATS Score", 0, 100, 65)
    max_missing = f3.slider("Max Missing Skills", 0, 20, 4)

    filtered = full_df[
        (full_df["overall_match"] >= min_overall)
        & (full_df["ats_score"] >= min_ats)
        & (full_df["missing_count"] <= max_missing)
    ].copy()
    filtered["shortlisted"] = "Yes"

    st.markdown("#### Full Leaderboard")
    st.dataframe(full_df, use_container_width=True, hide_index=True)
    st.markdown("#### Filtered Shortlist")
    if filtered.empty:
        st.info("No resumes meet current shortlist thresholds.")
    else:
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.bar_chart(filtered.set_index("resume_name")[["overall_match", "ats_score"]])

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download Full Results (CSV)",
            data=full_df.to_csv(index=False),
            file_name="batch_results_full.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "Download Shortlist (CSV)",
            data=filtered.to_csv(index=False),
            file_name="batch_results_shortlist.csv",
            mime="text/csv",
            use_container_width=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="ResumePilot AI",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()

    defaults = {
        "use_sample": False,
        "analysis_results": None,
        "resume_health_results": None,
        "batch_results": None,
        "privacy_summary": {},
        "resume_pdf_bytes": None,
        "jd_pdf_bytes": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    st.markdown(
        """
        <div class="app-hero">
            <h2 style="margin:0;">ResumePilot AI</h2>
            <p style="margin:0.35rem 0 0 0;">
                Hybrid AI matching with role templates, section scoring, requirement evidence mapping, rewrite diff,
                30-day action plans, tailored resume drafts, fairness checks, and multilingual support.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    resume_pdf = None
    resume_pdfs = []
    jd_mode = "Paste Text"
    jd_upload = None
    jd_pasted = ""
    base_model_name = MODEL_OPTIONS[0]
    auto_multilingual = True
    use_reranker = True
    reranker_model = DEFAULT_RERANKER_MODEL
    use_ocr_fallback = True
    redact_enabled = False
    llm_model_name = LLM_MODEL_OPTIONS[0]
    secret_groq_key = safe_get_secret("GROQ_API_KEY")
    groq_api_key = secret_groq_key
    simple_mode = True
    screening_mode = "JD Match"
    role_hint = ""

    with st.sidebar:
        st.subheader("Quick Setup")
        ui_mode = st.radio("Experience", ["Simple (Recommended)", "Advanced"], index=0)
        simple_mode = ui_mode == "Simple (Recommended)"
        if simple_mode:
            analysis_mode = "Single Resume"
            role_template_name = "General"
            screening_mode = st.radio("Screening Mode", ["JD Match", "Resume Health"], index=0)
            st.caption("Simple mode keeps only core controls.")

            source_options = ["Live Upload/Paste", "Use Built-in Sample"]
            source_index = 1 if st.session_state.use_sample else 0
            input_source = st.radio("Input Source", source_options, index=source_index)
            st.session_state.use_sample = input_source == "Use Built-in Sample"
            st.caption("Inputs are on the main screen below.")

            st.markdown("### LLM Suggestions")
            groq_api_key = secret_groq_key
            if groq_api_key:
                st.success("LLM suggestions enabled from Space Secret.")
            else:
                st.warning("Add `GROQ_API_KEY` in Space Secrets to use AI suggestions.")
        else:
            analysis_mode = st.radio("Mode", ["Single Resume", "Batch Screening"], index=0)
            if analysis_mode == "Single Resume":
                screening_mode = st.radio("Screening Mode", ["JD Match", "Resume Health"], index=0)
            else:
                screening_mode = "JD Match"
                st.caption("Batch Screening currently runs JD Match mode.")

            source_options = ["Live Upload/Paste", "Use Built-in Sample"]
            source_index = 1 if st.session_state.use_sample else 0
            input_source = st.radio("Input Source", source_options, index=source_index)
            st.session_state.use_sample = input_source == "Use Built-in Sample"

            role_template_name = st.selectbox("Role Template", get_role_template_names(), index=0)
            st.caption(get_role_template(role_template_name).get("description", ""))
            st.caption("Input files are on the main page below.")

            if st.session_state.use_sample:
                if analysis_mode == "Single Resume" and screening_mode == "Resume Health":
                    st.info("Sample mode is ON. Built-in resume will be used.")
                else:
                    st.info("Sample mode is ON. Built-in resume and JD will be used.")
                st.caption("Switch Input Source to `Live Upload/Paste` to use your own files.")

            with st.expander("LLM Suggestions", expanded=True):
                llm_mode = st.radio("Suggestion Engine", ["Auto (Groq if key)", "Rule-based only"], index=0)
                llm_model_name = st.selectbox("Groq Model", LLM_MODEL_OPTIONS, index=0)
                groq_api_key_input = st.text_input("Groq API Key (Optional)", value="", type="password")
                groq_api_key = groq_api_key_input.strip() or secret_groq_key
                if llm_mode == "Rule-based only":
                    groq_api_key = ""

            with st.expander("Advanced Settings", expanded=False):
                base_model_name = st.selectbox("Embedding Model", MODEL_OPTIONS, index=0)
                auto_multilingual = st.checkbox("Auto Multilingual Routing", value=True)
                use_reranker = st.checkbox("Enable Cross-Encoder Reranker", value=True)
                reranker_model = st.text_input("Reranker Model", value=DEFAULT_RERANKER_MODEL)
                use_ocr_fallback = st.checkbox("OCR Fallback for Scanned PDFs", value=True)
                redact_enabled = st.checkbox("Redact PII Before Analysis", value=False)

    if simple_mode:
        home_title = "Start Your JD Match" if screening_mode == "JD Match" else "Start Your Resume Health Check"
        home_sub = (
            "Upload resume + target JD to get fit score, skills gap, and AI suggestions."
            if screening_mode == "JD Match"
            else "Upload your resume only to get quality score, ATS health, and concrete fixes."
        )
        step_2 = "Add Job Description" if screening_mode == "JD Match" else "Run Health Audit"
        step_3 = "Analyze Match and Fixes" if screening_mode == "JD Match" else "Get Score and Fixes"

        st.markdown(
            f"""
            <div class="home-shell">
                <div class="home-title">{home_title}</div>
                <div class="home-sub">{home_sub}</div>
                <div class="home-steps">
                    <span class="home-step"><span class="home-step-num">1</span><span class="home-step-label">Upload Resume</span></span>
                    <span class="home-step"><span class="home-step-num">2</span><span class="home-step-label">{step_2}</span></span>
                    <span class="home-step"><span class="home-step-num">3</span><span class="home-step-label">{step_3}</span></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.use_sample:
            st.markdown(
                (
                    "<div class='home-ready good'>Sample mode active. Built-in resume and JD will be analyzed when you click Analyze.</div>"
                    if screening_mode == "JD Match"
                    else "<div class='home-ready good'>Sample mode active. Built-in resume will be analyzed when you click Analyze.</div>"
                ),
                unsafe_allow_html=True,
            )
        else:
            if screening_mode == "JD Match":
                i1, i2 = st.columns([1, 1], gap="large")
                with i1:
                    with st.container(border=True):
                        st.markdown("#### Resume")
                        st.caption("Upload one resume PDF (max 20MB).")
                        resume_pdf = st.file_uploader(
                            "Upload Resume PDF",
                            type=["pdf"],
                            accept_multiple_files=False,
                            key="main_resume_pdf",
                        )
                with i2:
                    with st.container(border=True):
                        st.markdown("#### Job Description")
                        jd_mode_pick = st.radio(
                            "JD Input Type",
                            ["Paste Text", "Upload File"],
                            horizontal=True,
                            key="main_jd_mode_pick",
                        )
                        if jd_mode_pick == "Upload File":
                            jd_mode = "Upload File"
                            jd_upload = st.file_uploader(
                                "Upload JD (PDF or TXT)",
                                type=["pdf", "txt"],
                                key="main_jd_upload",
                            )
                        else:
                            jd_mode = "Paste Text"
                            jd_pasted = st.text_area(
                                "Paste Job Description",
                                height=190,
                                key="main_jd_pasted",
                                placeholder="Paste the full job description here...",
                            )

                resume_ready = bool(resume_pdf)
                jd_ready = bool(jd_upload) if jd_mode == "Upload File" else bool(jd_pasted.strip())
                ready_cls = "good" if (resume_ready and jd_ready) else "warn"
                ready_text = (
                    "Inputs ready. Click Analyze."
                    if (resume_ready and jd_ready)
                    else "Upload both Resume and JD to enable a full analysis."
                )
            else:
                i1, i2 = st.columns([1.35, 1], gap="large")
                with i1:
                    with st.container(border=True):
                        st.markdown("#### Resume")
                        st.caption("Upload one resume PDF (max 20MB).")
                        resume_pdf = st.file_uploader(
                            "Upload Resume PDF",
                            type=["pdf"],
                            accept_multiple_files=False,
                            key="main_resume_pdf_health",
                        )
                with i2:
                    with st.container(border=True):
                        st.markdown("#### Target Role (Optional)")
                        role_hint = st.text_input(
                            "Role Hint",
                            value="",
                            key="main_role_hint",
                            placeholder="Example: Data Scientist, Backend Engineer, AI Engineer",
                        )
                        st.caption("Used to tailor resume-only suggestions.")

                ready_cls = "good" if bool(resume_pdf) else "warn"
                ready_text = (
                    "Resume ready. Click Analyze."
                    if bool(resume_pdf)
                    else "Upload a resume PDF to run Resume Health mode."
                )

            st.markdown(f"<div class='home-ready {ready_cls}'>{ready_text}</div>", unsafe_allow_html=True)
    else:
        if analysis_mode == "Single Resume" and screening_mode == "Resume Health":
            adv_title = "Advanced Resume Health Analysis"
            adv_sub = "Tune model, OCR, privacy, and LLM settings, then run resume-only quality scoring."
            adv_step_2 = "Upload Resume"
        elif analysis_mode == "Single Resume":
            adv_title = "Advanced JD Match Analysis"
            adv_sub = "Tune model, reranker, OCR, privacy, and LLM settings before scoring resume vs JD."
            adv_step_2 = "Upload Resume + JD"
        else:
            adv_title = "Advanced Batch Screening"
            adv_sub = "Upload multiple resumes, set strict shortlist thresholds, and rank candidates against one JD."
            adv_step_2 = "Upload Resumes + JD"
        st.markdown(
            f"""
            <div class="home-shell">
                <div class="home-title">{adv_title}</div>
                <div class="home-sub">{adv_sub}</div>
                <div class="home-steps">
                    <span class="home-step"><span class="home-step-num">1</span><span class="home-step-label">Set Analysis Options</span></span>
                    <span class="home-step"><span class="home-step-num">2</span><span class="home-step-label">{adv_step_2}</span></span>
                    <span class="home-step"><span class="home-step-num">3</span><span class="home-step-label">Run and Review Results</span></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.use_sample:
            if analysis_mode == "Single Resume" and screening_mode == "Resume Health":
                st.markdown(
                    "<div class='home-ready good'>Sample mode active. Built-in resume will be analyzed when you click Analyze.</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='home-ready good'>Sample mode active. Built-in inputs will be analyzed when you click Analyze.</div>",
                    unsafe_allow_html=True,
                )
        else:
            if analysis_mode == "Single Resume" and screening_mode == "JD Match":
                i1, i2 = st.columns([1, 1], gap="large")
                with i1:
                    with st.container(border=True):
                        st.markdown("#### Resume")
                        st.caption("Upload one resume PDF (max 20MB).")
                        resume_pdf = st.file_uploader(
                            "Upload Resume PDF",
                            type=["pdf"],
                            accept_multiple_files=False,
                            key="adv_main_resume_pdf",
                        )
                with i2:
                    with st.container(border=True):
                        st.markdown("#### Job Description")
                        jd_mode_pick = st.radio(
                            "JD Input Type",
                            ["Paste Text", "Upload File"],
                            horizontal=True,
                            key="adv_main_jd_mode_pick",
                        )
                        if jd_mode_pick == "Upload File":
                            jd_mode = "Upload File"
                            jd_upload = st.file_uploader(
                                "Upload JD (PDF or TXT)",
                                type=["pdf", "txt"],
                                key="adv_main_jd_upload",
                            )
                        else:
                            jd_mode = "Paste Text"
                            jd_pasted = st.text_area(
                                "Paste Job Description",
                                height=190,
                                key="adv_main_jd_pasted",
                                placeholder="Paste the full job description here...",
                            )

                resume_ready = bool(resume_pdf)
                jd_ready = bool(jd_upload) if jd_mode == "Upload File" else bool(jd_pasted.strip())
                ready_cls = "good" if (resume_ready and jd_ready) else "warn"
                ready_text = (
                    "Inputs ready. Click Analyze."
                    if (resume_ready and jd_ready)
                    else "Upload both Resume and JD to enable a full analysis."
                )
                st.markdown(f"<div class='home-ready {ready_cls}'>{ready_text}</div>", unsafe_allow_html=True)

            elif analysis_mode == "Single Resume":
                i1, i2 = st.columns([1.35, 1], gap="large")
                with i1:
                    with st.container(border=True):
                        st.markdown("#### Resume")
                        st.caption("Upload one resume PDF (max 20MB).")
                        resume_pdf = st.file_uploader(
                            "Upload Resume PDF",
                            type=["pdf"],
                            accept_multiple_files=False,
                            key="adv_main_resume_pdf_health",
                        )
                with i2:
                    with st.container(border=True):
                        st.markdown("#### Target Role (Optional)")
                        role_hint = st.text_input(
                            "Role Hint",
                            value="",
                            key="adv_main_role_hint",
                            placeholder="Example: Data Scientist, Backend Engineer, AI Engineer",
                        )
                        st.caption("Used to tailor resume-only suggestions.")

                ready_cls = "good" if bool(resume_pdf) else "warn"
                ready_text = (
                    "Resume ready. Click Analyze."
                    if bool(resume_pdf)
                    else "Upload a resume PDF to run Resume Health mode."
                )
                st.markdown(f"<div class='home-ready {ready_cls}'>{ready_text}</div>", unsafe_allow_html=True)

            else:
                i1, i2 = st.columns([1, 1], gap="large")
                with i1:
                    with st.container(border=True):
                        st.markdown("#### Resumes")
                        st.caption("Upload multiple resume PDFs for batch screening.")
                        resume_pdfs = st.file_uploader(
                            "Upload Resume PDFs",
                            type=["pdf"],
                            accept_multiple_files=True,
                            key="adv_main_resume_pdfs_batch",
                        )
                with i2:
                    with st.container(border=True):
                        st.markdown("#### Job Description")
                        jd_mode_pick = st.radio(
                            "JD Input Type",
                            ["Paste Text", "Upload File"],
                            horizontal=True,
                            key="adv_main_batch_jd_mode_pick",
                        )
                        if jd_mode_pick == "Upload File":
                            jd_mode = "Upload File"
                            jd_upload = st.file_uploader(
                                "Upload JD (PDF or TXT)",
                                type=["pdf", "txt"],
                                key="adv_main_batch_jd_upload",
                            )
                        else:
                            jd_mode = "Paste Text"
                            jd_pasted = st.text_area(
                                "Paste Job Description",
                                height=190,
                                key="adv_main_batch_jd_pasted",
                                placeholder="Paste the full job description here...",
                            )

                resume_ready = bool(resume_pdfs)
                jd_ready = bool(jd_upload) if jd_mode == "Upload File" else bool(jd_pasted.strip())
                ready_cls = "good" if (resume_ready and jd_ready) else "warn"
                ready_text = (
                    "Inputs ready. Click Analyze."
                    if (resume_ready and jd_ready)
                    else "Upload resumes and JD to enable batch screening."
                )
                st.markdown(f"<div class='home-ready {ready_cls}'>{ready_text}</div>", unsafe_allow_html=True)

    def _jd_input_ready() -> bool:
        return bool(jd_upload) if jd_mode == "Upload File" else bool(jd_pasted.strip())

    if simple_mode:
        if screening_mode == "Resume Health":
            inputs_ready = st.session_state.use_sample or bool(resume_pdf)
        else:
            inputs_ready = st.session_state.use_sample or (bool(resume_pdf) and _jd_input_ready())
    else:
        if st.session_state.use_sample:
            inputs_ready = True
        elif analysis_mode == "Single Resume":
            if screening_mode == "Resume Health":
                inputs_ready = bool(resume_pdf)
            else:
                inputs_ready = bool(resume_pdf) and _jd_input_ready()
        else:
            inputs_ready = bool(resume_pdfs) and _jd_input_ready()

    if simple_mode:
        action_label = (
            "Analyze Resume with AI Suggestions"
            if screening_mode == "JD Match"
            else "Run Resume Health Analysis"
        )
    elif st.session_state.use_sample:
        if analysis_mode == "Single Resume" and screening_mode == "Resume Health":
            action_label = "Run Sample Resume Health Analysis"
        else:
            action_label = "Run Sample Analysis" if analysis_mode == "Single Resume" else "Run Sample Batch Screening"
    else:
        if analysis_mode == "Single Resume" and screening_mode == "Resume Health":
            action_label = "Run Resume Health Analysis"
        else:
            action_label = "Analyze Resume Match" if analysis_mode == "Single Resume" else "Run Batch Screening"
    if not inputs_ready:
        st.caption("Complete required inputs to enable Analyze.")
    run_clicked = st.button(action_label, type="primary", use_container_width=True)

    if run_clicked and not inputs_ready:
        if analysis_mode == "Single Resume" and screening_mode == "Resume Health":
            st.warning("Missing: Resume. Upload a resume PDF, then click Analyze.")
        elif analysis_mode == "Single Resume":
            st.warning("Missing: Resume or Job Description. Upload both inputs, then click Analyze.")
        else:
            st.warning("Missing: Resume files or Job Description. Upload all required inputs, then click Analyze.")
    elif run_clicked:
        if simple_mode and not groq_api_key.strip():
            st.warning("No `GROQ_API_KEY` found. Running with rule-based suggestions for this analysis.")

        if analysis_mode == "Single Resume" and screening_mode == "Resume Health":
            if st.session_state.use_sample:
                resume_text = SAMPLE_RESUME_TEXT
                resume_pii = {}
                resume_pdf_bytes = None
                if redact_enabled:
                    resume_text, resume_pii = redact_pii(resume_text)
            else:
                prep = _prepare_text(
                    uploaded_file=resume_pdf,
                    pasted_text=None,
                    use_ocr_fallback=use_ocr_fallback,
                    redact_enabled=redact_enabled,
                )
                resume_text = prep["text"]
                resume_pii = prep["pii_counts"]
                resume_pdf_bytes = _extract_uploaded_bytes(resume_pdf) if resume_pdf else None

            if not resume_text.strip():
                st.error("Resume text is empty. Please upload a valid PDF or use sample mode.")
            else:
                try:
                    health_results = run_resume_health_analysis(
                        resume_text=resume_text,
                        role_hint=role_hint,
                        groq_api_key=groq_api_key,
                        llm_model_name=llm_model_name,
                    )
                    health_results["privacy"] = {"resume_pii_redacted": resume_pii, "jd_pii_redacted": {}}
                    st.session_state.resume_health_results = health_results
                    st.session_state.analysis_results = None
                    st.session_state.batch_results = None
                    st.session_state.privacy_summary = (
                        health_results["privacy"] if _has_nonzero_privacy_redactions(health_results["privacy"]) else {}
                    )
                    st.session_state.resume_pdf_bytes = resume_pdf_bytes
                    st.session_state.jd_pdf_bytes = None
                    st.success("Resume health analysis complete.")
                except Exception as exc:
                    st.session_state.resume_health_results = None
                    st.exception(exc)
        else:
            with st.spinner("Preparing inputs..."):
                if st.session_state.use_sample:
                    jd_text = SAMPLE_JD_TEXT
                    st.session_state.jd_pdf_bytes = None
                else:
                    if jd_mode == "Upload File":
                        jd_text = (
                            extract_text_from_input(jd_upload, use_ocr_fallback=use_ocr_fallback)
                            if jd_upload is not None
                            else ""
                        )
                    else:
                        jd_text = jd_pasted.strip()

                    st.session_state.jd_pdf_bytes = _extract_uploaded_bytes(jd_upload) if jd_upload else None

                jd_pii = {}
                if redact_enabled and jd_text:
                    jd_text, jd_pii = redact_pii(jd_text)

            if not jd_text.strip():
                st.error("Job description is empty. Please provide a valid JD.")
            elif analysis_mode == "Single Resume":
                if st.session_state.use_sample:
                    resume_text = SAMPLE_RESUME_TEXT
                    resume_meta = {"word_count": len(SAMPLE_RESUME_TEXT.split()), "sections": {}}
                    resume_pii = {}
                    resume_pdf_bytes = None
                    if redact_enabled:
                        resume_text, resume_pii = redact_pii(resume_text)
                else:
                    prep = _prepare_text(
                        uploaded_file=resume_pdf,
                        pasted_text=None,
                        use_ocr_fallback=use_ocr_fallback,
                        redact_enabled=redact_enabled,
                    )
                    resume_text = prep["text"]
                    resume_meta = prep["metadata"]
                    resume_pii = prep["pii_counts"]
                    resume_pdf_bytes = _extract_uploaded_bytes(resume_pdf) if resume_pdf else None

                if not resume_text.strip():
                    st.error("Resume text is empty. Please upload a valid PDF or use sample mode.")
                else:
                    try:
                        analysis = run_analysis(
                            resume_text=resume_text,
                            jd_text=jd_text,
                            base_model_name=base_model_name,
                            auto_multilingual=auto_multilingual,
                            role_template_name=role_template_name,
                            use_reranker=use_reranker,
                            reranker_model=reranker_model,
                            groq_api_key=groq_api_key,
                            llm_model_name=llm_model_name,
                            include_suggestions=True,
                        )
                        analysis["input_metadata"] = {
                            "resume_word_count": resume_meta.get("word_count", 0),
                            "jd_word_count": len(jd_text.split()),
                        }
                        analysis["privacy"] = {"resume_pii_redacted": resume_pii, "jd_pii_redacted": jd_pii}

                        st.session_state.analysis_results = analysis
                        st.session_state.resume_health_results = None
                        st.session_state.batch_results = None
                        st.session_state.privacy_summary = (
                            analysis["privacy"] if _has_nonzero_privacy_redactions(analysis["privacy"]) else {}
                        )
                        st.session_state.resume_pdf_bytes = resume_pdf_bytes

                        st.success("Single resume analysis complete.")
                    except Exception as exc:
                        st.session_state.analysis_results = None
                        st.exception(exc)
            else:
                try:
                    if st.session_state.use_sample and not resume_pdfs:
                        class _SampleUpload:
                            def __init__(self, name: str, text: str) -> None:
                                self.name = name
                                self._bytes = text.encode("utf-8")

                            def read(self) -> bytes:
                                return self._bytes

                            def seek(self, _: int) -> None:
                                return None

                        sample_files = [
                            _SampleUpload("sample_resume_1.txt", SAMPLE_RESUME_TEXT),
                            _SampleUpload("sample_resume_2.txt", SAMPLE_RESUME_TEXT_2),
                        ]
                        rows = run_batch_analysis(
                            resume_files=sample_files,
                            jd_text=jd_text,
                            base_model_name=base_model_name,
                            auto_multilingual=auto_multilingual,
                            role_template_name=role_template_name,
                            use_reranker=use_reranker,
                            reranker_model=reranker_model,
                            use_ocr_fallback=use_ocr_fallback,
                            redact_enabled=redact_enabled,
                        )
                    else:
                        if not resume_pdfs:
                            st.error("Please upload at least one resume PDF for batch mode.")
                            rows = []
                        else:
                            rows = run_batch_analysis(
                                resume_files=resume_pdfs,
                                jd_text=jd_text,
                                base_model_name=base_model_name,
                                auto_multilingual=auto_multilingual,
                                role_template_name=role_template_name,
                                use_reranker=use_reranker,
                                reranker_model=reranker_model,
                                use_ocr_fallback=use_ocr_fallback,
                                redact_enabled=redact_enabled,
                            )
                    st.session_state.batch_results = rows
                    st.session_state.analysis_results = None
                    st.session_state.resume_health_results = None
                    if rows:
                        st.success(f"Batch screening complete for {len(rows)} resumes.")
                except Exception as exc:
                    st.session_state.batch_results = None
                    st.exception(exc)

    if analysis_mode == "Single Resume":
        if screening_mode == "Resume Health":
            has_output = st.session_state.resume_health_results is not None
        else:
            has_output = st.session_state.analysis_results is not None
    else:
        has_output = bool(st.session_state.batch_results)
    if simple_mode:
        if screening_mode == "Resume Health":
            if not st.session_state.resume_health_results:
                st.caption("Upload a resume and click Analyze to see resume quality score and fixes.")
            else:
                render_resume_health_results(
                    st.session_state.resume_health_results,
                    st.session_state.get("resume_pdf_bytes"),
                )
        else:
            if not st.session_state.analysis_results:
                st.caption("Fill the home inputs above, then click Analyze.")
            else:
                render_single_results(
                    st.session_state.analysis_results,
                    st.session_state.get("resume_pdf_bytes"),
                    st.session_state.get("jd_pdf_bytes"),
                    simple_mode=True,
                )

        with st.expander("Input Preview", expanded=False):
            if st.session_state.use_sample:
                if screening_mode == "JD Match":
                    c1, c2 = st.columns(2)
                    with c1:
                        st.text_area("Sample Resume", SAMPLE_RESUME_TEXT, height=220, disabled=True)
                    with c2:
                        st.text_area("Sample JD", SAMPLE_JD_TEXT, height=220, disabled=True)
                else:
                    st.text_area("Sample Resume", SAMPLE_RESUME_TEXT, height=220, disabled=True)
            else:
                if screening_mode == "JD Match":
                    resume_ready = bool(resume_pdf)
                    jd_ready = bool(jd_upload) if jd_mode == "Upload File" else bool(jd_pasted.strip())
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Resume Input", "Ready" if resume_ready else "Missing")
                    with c2:
                        st.metric("JD Input", "Ready" if jd_ready else "Missing")
                else:
                    st.metric("Resume Input", "Ready" if bool(resume_pdf) else "Missing")
                    if role_hint.strip():
                        st.caption(f"Role Hint: {role_hint.strip()}")
    else:
        if has_output:
            results_tab, input_tab = st.tabs(["Analysis Results", "Input Preview"])
        else:
            input_tab, results_tab = st.tabs(["Input Preview", "Analysis Results"])

        with input_tab:
            if has_output:
                st.info("Latest results are ready in the Analysis Results tab.")

            if st.session_state.use_sample:
                st.info("Sample mode active.")
                if analysis_mode == "Single Resume" and screening_mode == "Resume Health":
                    st.text_area("Sample Resume", SAMPLE_RESUME_TEXT, height=320, disabled=True)
                    st.caption("Resume Health mode in Advanced uses resume-only scoring.")
                elif analysis_mode == "Single Resume":
                    c1, c2 = st.columns(2)
                    with c1:
                        st.text_area("Sample Resume", SAMPLE_RESUME_TEXT, height=320, disabled=True)
                    with c2:
                        st.text_area("Sample JD", SAMPLE_JD_TEXT, height=320, disabled=True)
                else:
                    st.markdown("Using two built-in sample resumes against sample JD in batch mode.")
            else:
                if analysis_mode == "Single Resume" and screening_mode == "Resume Health":
                    resume_ready = bool(resume_pdf)
                    with st.container(border=True):
                        st.markdown("#### Resume Input")
                        if resume_ready:
                            st.success("Ready")
                        else:
                            st.info("Waiting for resume upload")
                            st.caption("Upload one resume PDF in the sidebar.")
                        if role_hint.strip():
                            st.caption(f"Role Hint: {role_hint.strip()}")

                    if not resume_ready:
                        st.warning("Missing: Resume. Upload a resume PDF to run Resume Health analysis.")
                    else:
                        st.success("Inputs look good. Click analyze.")
                else:
                    resume_ready = bool(resume_pdf) if analysis_mode == "Single Resume" else bool(resume_pdfs)
                    jd_ready = bool(jd_upload) if jd_mode == "Upload File" else bool(jd_pasted.strip())

                    c1, c2 = st.columns(2)
                    with c1:
                        with st.container(border=True):
                            st.markdown("#### Resume Input")
                        if resume_ready:
                            st.success("Ready")
                        else:
                            st.info("Waiting for resume upload")
                            if analysis_mode == "Single Resume":
                                st.caption("Upload one resume PDF in the input section above.")
                            else:
                                st.caption("Upload one or more resume PDFs in the input section above.")
                    with c2:
                        with st.container(border=True):
                            st.markdown("#### JD Input")
                        if jd_ready:
                            st.success("Ready")
                        else:
                            st.info("Waiting for job description")
                            st.caption("Paste JD text or upload a JD file in the input section above.")

                    if not resume_ready or not jd_ready:
                        missing_parts = []
                        if not resume_ready:
                            missing_parts.append("Resume")
                        if not jd_ready:
                            missing_parts.append("Job Description")
                        st.warning(f"Missing: {', '.join(missing_parts)}. Complete these inputs to run analysis.")
                    else:
                        st.success("Inputs look good. Click analyze.")

        with results_tab:
            if analysis_mode == "Single Resume":
                if screening_mode == "Resume Health":
                    if not st.session_state.resume_health_results:
                        st.info("No analysis yet. Submit a resume to generate Resume Health results.")
                    else:
                        render_resume_health_results(
                            st.session_state.resume_health_results,
                            st.session_state.get("resume_pdf_bytes"),
                        )
                        privacy_summary = st.session_state.privacy_summary or {}
                        if _has_nonzero_privacy_redactions(privacy_summary):
                            with st.expander("Privacy Redaction Summary", expanded=False):
                                st.json(privacy_summary)
                else:
                    if not st.session_state.analysis_results:
                        st.info("No analysis yet. Submit resume and JD to generate results.")
                    else:
                        render_single_results(
                            st.session_state.analysis_results,
                            st.session_state.get("resume_pdf_bytes"),
                            st.session_state.get("jd_pdf_bytes"),
                            simple_mode=False,
                        )
                        privacy_summary = st.session_state.privacy_summary or {}
                        if _has_nonzero_privacy_redactions(privacy_summary):
                            with st.expander("Privacy Redaction Summary", expanded=False):
                                st.json(privacy_summary)
            else:
                if not st.session_state.batch_results:
                    st.info("No batch output yet. Upload resumes and run screening.")
                else:
                    render_batch_results(st.session_state.batch_results)


if __name__ == "__main__":
    main()
