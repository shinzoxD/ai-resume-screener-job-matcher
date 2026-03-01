"""AI Resume Screener & Job Matcher with advanced recruiter/candidate features."""

from __future__ import annotations

import io
import json
import os
import textwrap
from datetime import datetime
from typing import Any, Dict, List, Sequence

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from utils.confidence import compute_confidence
from utils.extractor import extract_document, extract_text_from_input
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
from utils.skills_db import format_skill_list
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
LLM_MODEL_OPTIONS = ["llama-3.1-70b-versatile", "mixtral-8x7b-32768"]


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
            .block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
            .app-hero {
                border: 1px solid rgba(0, 184, 148, 0.25);
                padding: 1rem 1.2rem; border-radius: 14px;
                background: linear-gradient(120deg, rgba(0, 184, 148, 0.12), rgba(9, 132, 227, 0.08));
                margin-bottom: 1rem;
            }
            .pill {
                display: inline-block; padding: 0.32rem 0.72rem; margin: 0.2rem 0.26rem 0.2rem 0;
                border-radius: 999px; border: 1px solid rgba(0, 184, 148, 0.35);
                background-color: rgba(0, 184, 148, 0.14); font-size: 0.84rem;
            }
            .muted { color: #9AA5B1; font-size: 0.9rem; }
            @media (max-width: 768px) {
                .block-container { padding-left: 0.9rem; padding-right: 0.9rem; }
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
    metric_cols = st.columns(6)
    metric_cols[0].metric("Overall Match", f"{score['overall']}%")
    metric_cols[1].metric("Calibrated", f"{confidence['calibrated_overall']}%")
    metric_cols[2].metric("Confidence", f"{confidence['confidence_pct']}%")
    metric_cols[3].metric("Confidence Band", confidence["band"])
    metric_cols[4].metric("Skill Alignment", f"{score['skill_alignment']}%")
    metric_cols[5].metric("ATS Score", f"{ats['ats_score']}%")
    st.progress(min(max(score["overall"] / 100, 0.0), 1.0))

    language = results.get("language", {})
    st.caption(
        f"Detected Languages -> Resume: {language_label(language.get('resume', 'unknown'))}, "
        f"JD: {language_label(language.get('jd', 'unknown'))}; "
        f"Model Used: `{language.get('effective_model', 'n/a')}`; "
        f"Role Template: **{results.get('role_template', 'General')}**"
    )

    if simple_mode:
        tabs = st.tabs(["Summary", "Skills Gap", "Suggestions"])

        with tabs[0]:
            st.markdown("#### Strong Points")
            for point in strong_points:
                st.markdown(f"- {point}")
            st.markdown("#### Score Breakdown")
            b1, b2, b3 = st.columns(3)
            b1.metric("Semantic", f"{score['semantic']}%")
            b2.metric("Lexical", f"{score['lexical']}%")
            b3.metric("Confidence", f"{confidence['confidence_pct']}%")

        with tabs[1]:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Matching Skills")
                render_skill_pills(format_skill_list(skills["matching_skills"]))
            with c2:
                st.markdown("#### Missing Skills")
                render_skill_pills(format_skill_list(skills["missing_skills"]))
            st.markdown("#### ATS Tips")
            for tip in ats["tips"][:4]:
                st.markdown(f"- {tip}")

        with tabs[2]:
            st.markdown("#### Personalized Suggestions")
            st.caption(f"Suggestion source: `{suggestions_payload.get('provider', 'unknown')}`")
            for idx, suggestion in enumerate(suggestions or ["No suggestions generated in this mode."], start=1):
                st.markdown(f"{idx}. {suggestion}")

            report_md = build_suggestions_markdown(
                match_score=score["overall"],
                matching_skills=format_skill_list(skills["matching_skills"]),
                missing_skills=format_skill_list(skills["missing_skills"]),
                strong_points=strong_points,
                suggestions=suggestions or ["Use the planner to close top skill gaps."],
                ats_score=ats["ats_score"],
            )
            report_pdf = markdown_to_pdf_bytes(report_md)

            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    "Download Suggestions (PDF)",
                    data=report_pdf,
                    file_name="improvement_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            with d2:
                st.download_button(
                    "Download Tailored Draft (Markdown)",
                    data=tailored_draft,
                    file_name="tailored_resume_draft.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

        return

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
        st.markdown("#### Strong Points")
        for point in strong_points:
            st.markdown(f"- {point}")
        st.markdown("#### Personalized Suggestions")
        st.caption(f"Suggestion source: `{suggestions_payload.get('provider', 'unknown')}`")
        for idx, suggestion in enumerate(suggestions or ["No suggestions generated in this mode."], start=1):
            st.markdown(f"{idx}. {suggestion}")

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
        components.html(rewrite.get("diff_html", "<p>No diff available.</p>"), height=420, scrolling=True)

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

        selected_keyword = st.selectbox("Keyword Highlights", keyword_options or ["python"])
        st.markdown("#### Resume Snippets")
        resume_snippets = highlight_keyword_snippets(results.get("resume_text", ""), selected_keyword)
        if resume_snippets:
            for snippet in resume_snippets:
                st.markdown(snippet, unsafe_allow_html=True)
        else:
            st.caption("No resume snippet found for selected keyword.")

        st.markdown("#### JD Snippets")
        jd_snippets = highlight_keyword_snippets(results.get("jd_text", ""), selected_keyword)
        if jd_snippets:
            for snippet in jd_snippets:
                st.markdown(snippet, unsafe_allow_html=True)
        else:
            st.caption("No JD snippet found for selected keyword.")

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
        page_title="AI Resume Screener & Job Matcher",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()

    defaults = {
        "use_sample": False,
        "analysis_results": None,
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
            <h2 style="margin:0;">AI Resume Screener & Job Matcher</h2>
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

    with st.sidebar:
        st.subheader("Quick Setup")
        ui_mode = st.radio("Experience", ["Simple (Recommended)", "Advanced"], index=0)
        simple_mode = ui_mode == "Simple (Recommended)"
        if simple_mode:
            analysis_mode = "Single Resume"
            role_template_name = "General"
            st.caption("Simple mode keeps only core controls.")

            source_options = ["Live Upload/Paste", "Use Built-in Sample"]
            source_index = 1 if st.session_state.use_sample else 0
            input_source = st.radio("Input Source", source_options, index=source_index)
            st.session_state.use_sample = input_source == "Use Built-in Sample"

            st.markdown("### Inputs")
            if st.session_state.use_sample:
                st.info("Sample mode is ON. Built-in resume and JD will be used.")
            else:
                resume_pdf = st.file_uploader("Resume PDF", type=["pdf"], accept_multiple_files=False)
                use_jd_upload = st.checkbox("Upload JD file instead of pasting text", value=False)
                if use_jd_upload:
                    jd_mode = "Upload File"
                    jd_upload = st.file_uploader("Upload JD (PDF or TXT)", type=["pdf", "txt"])
                else:
                    jd_mode = "Paste Text"
                    jd_pasted = st.text_area("Paste Job Description", height=220)

            st.markdown("### LLM Suggestions")
            groq_api_key_input = st.text_input("Groq API Key (Optional override)", value="", type="password")
            groq_api_key = groq_api_key_input.strip() or secret_groq_key
            if groq_api_key:
                st.success("LLM suggestions enabled (Groq).")
            else:
                st.warning("Add `GROQ_API_KEY` in Space Secrets (or paste key above) to use AI suggestions.")
        else:
            analysis_mode = st.radio("Mode", ["Single Resume", "Batch Screening"], index=0)

            source_options = ["Live Upload/Paste", "Use Built-in Sample"]
            source_index = 1 if st.session_state.use_sample else 0
            input_source = st.radio("Input Source", source_options, index=source_index)
            st.session_state.use_sample = input_source == "Use Built-in Sample"

            role_template_name = st.selectbox("Role Template", get_role_template_names(), index=0)
            st.caption(get_role_template(role_template_name).get("description", ""))

            st.markdown("### Inputs")
            if st.session_state.use_sample:
                st.info("Sample mode is ON. Built-in resume and JD will be used.")
                st.caption("Switch Input Source to `Live Upload/Paste` to use your own files.")
            else:
                if analysis_mode == "Single Resume":
                    resume_pdf = st.file_uploader("Resume PDF", type=["pdf"], accept_multiple_files=False)
                    resume_pdfs = []
                else:
                    resume_pdfs = st.file_uploader(
                        "Upload Resume PDFs",
                        type=["pdf"],
                        accept_multiple_files=True,
                        help="Batch mode ranks multiple candidates against one JD.",
                    )
                    resume_pdf = None

                jd_mode = st.radio("Job Description Input", ["Paste Text", "Upload File"], index=0)
                if jd_mode == "Paste Text":
                    jd_pasted = st.text_area("Paste Job Description", height=220)
                else:
                    jd_upload = st.file_uploader("Upload JD (PDF or TXT)", type=["pdf", "txt"])

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
        action_label = "Analyze Resume with AI Suggestions"
    elif st.session_state.use_sample:
        action_label = "Run Sample Analysis" if analysis_mode == "Single Resume" else "Run Sample Batch Screening"
    else:
        action_label = "Analyze Resume Match" if analysis_mode == "Single Resume" else "Run Batch Screening"
    run_clicked = st.button(action_label, type="primary", use_container_width=True)

    if run_clicked and simple_mode and not groq_api_key.strip():
        st.error("Simple mode requires `GROQ_API_KEY` to generate LLM suggestions. Add it in Space Secrets.")
        run_clicked = False

    if run_clicked:
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
                    st.session_state.batch_results = None
                    st.session_state.privacy_summary = analysis["privacy"]
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
                if rows:
                    st.success(f"Batch screening complete for {len(rows)} resumes.")
            except Exception as exc:
                st.session_state.batch_results = None
                st.exception(exc)

    has_output = (
        st.session_state.analysis_results is not None
        if analysis_mode == "Single Resume"
        else bool(st.session_state.batch_results)
    )
    if simple_mode:
        if not st.session_state.analysis_results:
            st.info("Upload resume + JD, then click Analyze to see AI suggestions.")
        else:
            render_single_results(
                st.session_state.analysis_results,
                st.session_state.get("resume_pdf_bytes"),
                st.session_state.get("jd_pdf_bytes"),
                simple_mode=True,
            )

        with st.expander("Input Preview", expanded=False):
            if st.session_state.use_sample:
                c1, c2 = st.columns(2)
                with c1:
                    st.text_area("Sample Resume", SAMPLE_RESUME_TEXT, height=220, disabled=True)
                with c2:
                    st.text_area("Sample JD", SAMPLE_JD_TEXT, height=220, disabled=True)
            else:
                resume_ready = bool(resume_pdf)
                jd_ready = bool(jd_upload) if jd_mode == "Upload File" else bool(jd_pasted.strip())
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Resume Input", "Ready" if resume_ready else "Missing")
                with c2:
                    st.metric("JD Input", "Ready" if jd_ready else "Missing")
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
                if analysis_mode == "Single Resume":
                    c1, c2 = st.columns(2)
                    with c1:
                        st.text_area("Sample Resume", SAMPLE_RESUME_TEXT, height=320, disabled=True)
                    with c2:
                        st.text_area("Sample JD", SAMPLE_JD_TEXT, height=320, disabled=True)
                else:
                    st.markdown("Using two built-in sample resumes against sample JD in batch mode.")
            else:
                resume_ready = bool(resume_pdf) if analysis_mode == "Single Resume" else bool(resume_pdfs)
                jd_ready = bool(jd_upload) if jd_mode == "Upload File" else bool(jd_pasted.strip())

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Resume Input", "Ready" if resume_ready else "Missing")
                with c2:
                    st.metric("JD Input", "Ready" if jd_ready else "Missing")

                if not resume_ready or not jd_ready:
                    st.warning("Upload resume and provide a full job description before running analysis.")
                else:
                    st.success("Inputs look good. Click analyze.")

        with results_tab:
            if analysis_mode == "Single Resume":
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
                    if privacy_summary:
                        st.markdown("#### Privacy Summary")
                        st.json(privacy_summary)
            else:
                if not st.session_state.batch_results:
                    st.info("No batch output yet. Upload resumes and run screening.")
                else:
                    render_batch_results(st.session_state.batch_results)


if __name__ == "__main__":
    main()
