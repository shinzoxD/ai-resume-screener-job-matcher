"""LLM-powered and fallback resume improvement suggestions."""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Dict, List, Sequence

try:
    from groq import Groq
except Exception:  # pragma: no cover - keeps app usable if SDK import fails
    Groq = None  # type: ignore


def _build_fallback_suggestions(
    match_score: float,
    missing_skills: Sequence[str],
    matching_skills: Sequence[str],
) -> List[str]:
    """Rule-based suggestions when no LLM provider is configured."""
    suggestions: List[str] = []

    if missing_skills:
        suggestions.append(
            f"Add evidence-backed bullets for missing skills: {', '.join(missing_skills[:6])}."
        )
    suggestions.append(
        "Rewrite experience bullets using action + metric format (for example, 'Built X, improving Y by Z%')."
    )
    suggestions.append(
        "Move the most relevant projects and tools to the top half of the first page for faster recruiter scan."
    )
    if match_score < 70:
        suggestions.append(
            "Mirror important JD phrases in your summary and skills section to improve semantic alignment."
        )
    if matching_skills:
        suggestions.append(
            f"Highlight your strongest matching skills ({', '.join(matching_skills[:5])}) in the professional summary."
        )
    suggestions.append(
        "Keep formatting ATS-safe with standard section headers, readable fonts, and no text embedded in images."
    )

    return suggestions[:5]


def _parse_suggestion_lines(text: str, min_items: int = 3, max_items: int = 5) -> List[str]:
    """Extract clean bullet points from raw model output."""
    if not text:
        return []

    lines: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*\u2022\d\.\)\(]+\s*", "", line).strip()
        if len(line) < 10:
            continue
        lines.append(line)

    deduped: List[str] = []
    seen = set()
    for line in lines:
        key = line.lower()
        if key not in seen:
            deduped.append(line)
            seen.add(key)

    # Some models return paragraph text instead of bullets.
    if len(deduped) < min_items:
        sentence_parts = re.split(r"(?<=[.!?])\s+", text.strip())
        sentence_candidates: List[str] = []
        for part in sentence_parts:
            cleaned = re.sub(r"^[-*\u2022\d\.\)\(]+\s*", "", part.strip())
            if len(cleaned) >= 18:
                sentence_candidates.append(cleaned.rstrip(".") + ".")

        seen_sentences = set()
        deduped = []
        for sentence in sentence_candidates:
            key = sentence.lower()
            if key not in seen_sentences:
                deduped.append(sentence)
                seen_sentences.add(key)

    if len(deduped) < min_items:
        return []
    return deduped[:max_items]


def _resolve_candidate_models(client: Any, preferred_models: Sequence[str]) -> List[str]:
    """Pick usable Groq models, preferring user-selected models first."""
    deduped_pref: List[str] = []
    seen = set()
    for model in preferred_models:
        m = (model or "").strip()
        if m and m not in seen:
            deduped_pref.append(m)
            seen.add(m)

    try:
        models = client.models.list()
        available = {getattr(m, "id", "") for m in getattr(models, "data", [])}
        filtered = [m for m in deduped_pref if m in available]
        if filtered:
            return filtered
    except Exception:
        pass

    return deduped_pref


def generate_improvement_suggestions(
    resume_text: str,
    jd_text: str,
    match_score: float,
    missing_skills: Sequence[str],
    matching_skills: Sequence[str],
    groq_api_key: str | None = None,
    model_name: str = "llama-3.1-70b-versatile",
) -> Dict[str, Any]:
    """Generate 3-5 personalized suggestions via Groq or deterministic fallback."""
    fallback = _build_fallback_suggestions(match_score, missing_skills, matching_skills)
    fallback_payload: Dict[str, Any] = {
        "provider": "rule-based",
        "suggestions": fallback,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "estimated_cost_usd": 0.0},
        "diagnostics": {"reason": "", "attempted_models": [], "errors": []},
    }

    api_key = (groq_api_key or os.getenv("GROQ_API_KEY") or "").strip()
    if not api_key or Groq is None:
        fallback_payload["diagnostics"]["reason"] = "no_api_key_or_groq_sdk"
        return fallback_payload

    prompt = f"""
You are an expert technical recruiter and resume coach.
Analyze the resume against the job description and provide exactly 5 concise, high-impact suggestions.

Constraints:
- Suggestions must be personalized to the candidate profile and target role.
- Focus on missing skills, quantification, ATS optimization, and stronger positioning.
- Each suggestion must be one sentence and start with a strong action verb.
- Do not include intro/outro text.

Current match score: {match_score:.2f}%
Missing skills: {", ".join(missing_skills) if missing_skills else "None"}
Matching skills: {", ".join(matching_skills) if matching_skills else "None"}

Resume:
{resume_text[:5500]}

Job Description:
{jd_text[:5500]}
""".strip()

    errors: List[str] = []
    attempted_models: List[str] = []

    try:
        client = Groq(api_key=api_key)
        candidate_models = _resolve_candidate_models(
            client,
            [
                model_name,
                "llama-3.3-70b-versatile",
                "llama-3.1-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
            ],
        )

        for candidate_model in candidate_models:
            attempted_models.append(candidate_model)
            try:
                completion = client.chat.completions.create(
                    model=candidate_model,
                    temperature=0.3,
                    max_tokens=500,
                    messages=[
                        {"role": "system", "content": "You provide practical resume improvement advice."},
                        {"role": "user", "content": prompt},
                    ],
                )
                content = completion.choices[0].message.content or ""
                suggestions = _parse_suggestion_lines(content, min_items=3, max_items=5)

                prompt_tokens = int(getattr(completion.usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(completion.usage, "completion_tokens", 0) or 0)
                estimated_cost_usd = ((prompt_tokens / 1_000_000) * 0.59) + ((completion_tokens / 1_000_000) * 0.79)

                if suggestions:
                    return {
                        "provider": f"groq:{candidate_model}",
                        "suggestions": suggestions,
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "estimated_cost_usd": round(estimated_cost_usd, 6),
                        },
                        "diagnostics": {
                            "reason": "",
                            "attempted_models": attempted_models,
                            "errors": errors,
                        },
                    }
                errors.append(f"{candidate_model}: parsed_empty_output")
            except Exception as exc:
                errors.append(f"{candidate_model}: {type(exc).__name__}: {exc}")
    except Exception as exc:
        errors.append(f"client_init: {type(exc).__name__}: {exc}")

    fallback_payload["diagnostics"] = {
        "reason": "groq_request_failed",
        "attempted_models": attempted_models,
        "errors": errors[:4],
    }
    return fallback_payload


def build_suggestions_markdown(
    match_score: float,
    matching_skills: Sequence[str],
    missing_skills: Sequence[str],
    strong_points: Sequence[str],
    suggestions: Sequence[str],
    ats_score: float,
) -> str:
    """Create a markdown report for download."""
    date_label = datetime.utcnow().strftime("%Y-%m-%d")
    lines = [
        "# Improved Resume Suggestions",
        "",
        f"Generated on: {date_label}",
        "",
        "## Summary",
        f"- Match Score: **{match_score:.2f}%**",
        f"- ATS Compatibility Score: **{ats_score:.2f}%**",
        "",
        "## Matching Skills",
        ", ".join(matching_skills) if matching_skills else "No matching skills detected.",
        "",
        "## Missing Skills",
        ", ".join(missing_skills) if missing_skills else "No critical skill gaps found.",
        "",
        "## Strong Points",
    ]

    if strong_points:
        lines.extend([f"- {point}" for point in strong_points])
    else:
        lines.append("- Add a concise summary to highlight role-fit quickly.")

    lines.extend(["", "## Personalized Improvements"])
    lines.extend([f"{idx}. {text}" for idx, text in enumerate(suggestions, start=1)])
    lines.append("")

    return "\n".join(lines)
