"""Resume-to-JD matching, explainability, and ATS analytics."""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Sequence

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.extractor import detect_sections, segment_sections
from utils.skills_db import extract_keywords, extract_skills_from_text

try:
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover - optional runtime path
    CrossEncoder = None  # type: ignore

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

WEIGHTS_NO_RERANKER = {
    "semantic": 0.62,
    "lexical": 0.14,
    "skill_alignment": 0.24,
}
WEIGHTS_WITH_RERANKER = {
    "semantic": 0.48,
    "lexical": 0.12,
    "skill_alignment": 0.20,
    "reranker": 0.20,
}

CONTACT_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "phone": r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b",
    "linkedin": r"(?:linkedin\.com/in/|linkedin\.com/company/)",
    "github": r"(?:github\.com/)",
}


@lru_cache(maxsize=4)
def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    """Load and cache sentence-transformer model."""
    return SentenceTransformer(model_name)


@lru_cache(maxsize=1)
def load_reranker_model(model_name: str = DEFAULT_RERANKER_MODEL) -> Any | None:
    """Load and cache optional cross-encoder reranker."""
    if CrossEncoder is None:
        return None
    try:
        return CrossEncoder(model_name)
    except Exception:
        return None


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def _clamp_pct(value: float) -> float:
    return max(0.0, min(100.0, value))


def _semantic_score(resume_text: str, jd_text: str, model_name: str) -> float:
    model = load_embedding_model(model_name)
    embeddings = model.encode([resume_text, jd_text], normalize_embeddings=True)
    raw = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
    return _clamp_pct(raw * 100)


def _lexical_score(resume_text: str, jd_text: str) -> float:
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=7000)
    try:
        tfidf = vectorizer.fit_transform([resume_text, jd_text])
        raw = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
    except ValueError:
        raw = 0.0
    return _clamp_pct(raw * 100)


def _reranker_score(
    resume_text: str,
    jd_text: str,
    use_reranker: bool,
    reranker_model: str,
) -> float | None:
    if not use_reranker:
        return None
    model = load_reranker_model(reranker_model)
    if model is None:
        return None

    try:
        raw_score = float(model.predict([(resume_text, jd_text)])[0])
        normalized = 1 / (1 + math.exp(-raw_score))
        return _clamp_pct(normalized * 100)
    except Exception:
        return None


def analyze_skill_alignment(
    resume_text: str,
    jd_text: str,
    custom_skills: Sequence[str] | None = None,
) -> Dict[str, object]:
    """Extract and compare skills present in resume and job description."""
    resume_skills = extract_skills_from_text(resume_text, custom_skills=custom_skills)
    jd_skills = extract_skills_from_text(jd_text, custom_skills=custom_skills)

    matching = sorted(resume_skills & jd_skills)
    missing = sorted(jd_skills - resume_skills)
    extra = sorted(resume_skills - jd_skills)
    coverage = _safe_divide(len(matching), len(jd_skills)) * 100

    return {
        "resume_skills": sorted(resume_skills),
        "jd_skills": sorted(jd_skills),
        "matching_skills": matching,
        "missing_skills": missing,
        "extra_skills": extra,
        "coverage_score": round(coverage, 2),
    }


def calculate_match_score(
    resume_text: str,
    jd_text: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    use_reranker: bool = False,
    reranker_model: str = DEFAULT_RERANKER_MODEL,
    custom_weights: Mapping[str, float] | None = None,
    custom_skills: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Compute hybrid resume-to-JD score with explainable feature weights."""
    resume_text = (resume_text or "").strip()
    jd_text = (jd_text or "").strip()

    if not resume_text or not jd_text:
        return {
            "semantic": 0.0,
            "lexical": 0.0,
            "skill_alignment": 0.0,
            "reranker": None,
            "overall": 0.0,
            "weights": dict(WEIGHTS_NO_RERANKER),
        }

    semantic = _semantic_score(resume_text, jd_text, model_name)
    lexical = _lexical_score(resume_text, jd_text)
    skill_details = analyze_skill_alignment(resume_text, jd_text, custom_skills=custom_skills)
    skill_alignment = float(skill_details["coverage_score"])
    reranker = _reranker_score(resume_text, jd_text, use_reranker, reranker_model)

    weights = dict(WEIGHTS_WITH_RERANKER if reranker is not None else WEIGHTS_NO_RERANKER)
    if custom_weights:
        weights = _normalize_weights(custom_weights)
        if reranker is None and "reranker" in weights:
            weights.pop("reranker", None)
            weights = _normalize_weights(weights)
    weighted_scores = {
        "semantic": semantic * weights["semantic"],
        "lexical": lexical * weights["lexical"],
        "skill_alignment": skill_alignment * weights["skill_alignment"],
    }
    if reranker is not None:
        weighted_scores["reranker"] = reranker * weights["reranker"]

    overall = sum(weighted_scores.values())
    return {
        "semantic": round(semantic, 2),
        "lexical": round(lexical, 2),
        "skill_alignment": round(skill_alignment, 2),
        "reranker": None if reranker is None else round(reranker, 2),
        "overall": round(_clamp_pct(overall), 2),
        "weights": weights,
    }


def _normalize_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    filtered = {key: float(value) for key, value in weights.items() if float(value) > 0}
    total = sum(filtered.values()) or 1.0
    return {key: value / total for key, value in filtered.items()}


def calculate_section_scores(
    resume_text: str,
    jd_text: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    use_reranker: bool = False,
    reranker_model: str = DEFAULT_RERANKER_MODEL,
    custom_weights: Mapping[str, float] | None = None,
    custom_skills: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    """Score each detected resume section against the full JD."""
    section_map = segment_sections(resume_text)
    rows: List[Dict[str, Any]] = []
    for section_name, section_text in section_map.items():
        if len(section_text.split()) < 8:
            continue
        score = calculate_match_score(
            resume_text=section_text,
            jd_text=jd_text,
            model_name=model_name,
            use_reranker=use_reranker,
            reranker_model=reranker_model,
            custom_weights=custom_weights,
            custom_skills=custom_skills,
        )
        rows.append(
            {
                "section": section_name.title(),
                "overall": score["overall"],
                "semantic": score["semantic"],
                "lexical": score["lexical"],
                "skill_alignment": score["skill_alignment"],
            }
        )
    rows.sort(key=lambda row: row["overall"], reverse=True)
    return rows


def build_feature_contributions(score_details: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Convert weighted score components into human-readable contributions."""
    weights = score_details.get("weights") or WEIGHTS_NO_RERANKER
    rows: List[Dict[str, Any]] = []
    for feature, weight in weights.items():
        value = score_details.get(feature)
        if value is None:
            continue
        contribution = float(value) * float(weight)
        rows.append(
            {
                "feature": feature,
                "score": round(float(value), 2),
                "weight": round(float(weight), 3),
                "contribution_points": round(contribution, 2),
            }
        )
    rows.sort(key=lambda row: row["contribution_points"], reverse=True)
    return rows


def build_match_explanation(
    resume_text: str,
    jd_text: str,
    score_details: Mapping[str, Any],
    skill_details: Mapping[str, Any],
    top_n_keywords: int = 12,
) -> Dict[str, Any]:
    """Generate plain-language explanation and evidence for a match decision."""
    keyword_rows = calculate_keyword_density(resume_text, jd_text, top_n=top_n_keywords)
    matched_keywords = [row["keyword"] for row in keyword_rows if row["status"] == "Matched"][:6]
    missing_keywords = [row["keyword"] for row in keyword_rows if row["status"] == "Missing"][:6]
    contributions = build_feature_contributions(score_details)

    evidence_points: List[str] = []
    if float(score_details.get("semantic", 0)) >= 75:
        evidence_points.append("Resume language aligns well with the intent and context of the role.")
    if float(skill_details.get("coverage_score", 0)) >= 60:
        evidence_points.append("Strong coverage of required technical skills in the JD.")
    if matched_keywords:
        evidence_points.append(f"Frequent JD keywords found in resume: {', '.join(matched_keywords[:4])}.")
    if missing_keywords:
        evidence_points.append(f"Important terms missing or underrepresented: {', '.join(missing_keywords[:4])}.")
    if not evidence_points:
        evidence_points.append("Core relevance exists, but feature signals are currently moderate.")

    return {
        "feature_contributions": contributions,
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "evidence_points": evidence_points[:4],
    }


def build_strong_points(
    resume_text: str,
    score_details: Dict[str, Any],
    skill_details: Dict[str, Any],
) -> List[str]:
    """Generate concise strengths for UI display."""
    points: List[str] = []
    word_count = _word_count(resume_text)
    matching_skills = skill_details.get("matching_skills", [])
    coverage = float(skill_details.get("coverage_score", 0))

    if score_details.get("overall", 0) >= 78:
        points.append("Strong overall alignment for this role based on hybrid ranking.")
    if score_details.get("semantic", 0) >= 80:
        points.append("Content language and context strongly match the job intent.")
    if coverage >= 65:
        points.append("High overlap in required skills and tools.")
    if len(matching_skills) >= 6:
        points.append(f"Solid skill evidence in resume: {', '.join(matching_skills[:6])}.")
    if 320 <= word_count <= 950:
        points.append("Resume length is in an ATS-friendly range for most companies.")
    if score_details.get("reranker") is not None and score_details.get("reranker", 0) >= 75:
        points.append("Cross-encoder reranker also signals strong fit.")

    if not points:
        points.append("Foundational role fit exists; targeted revisions can raise competitiveness quickly.")

    return points[:5]


def calculate_ats_compatibility(
    resume_text: str,
    jd_text: str,
    skill_details: Dict[str, Any],
) -> Dict[str, Any]:
    """Estimate ATS compatibility from structure, contact info, and keyword alignment."""
    sections = detect_sections(resume_text)
    section_hits = sum(1 for is_present in sections.values() if is_present)
    section_score = _safe_divide(section_hits, len(sections)) * 100

    contact_hits = 0
    for pattern in CONTACT_PATTERNS.values():
        if re.search(pattern, resume_text or "", flags=re.IGNORECASE):
            contact_hits += 1
    contact_score = _safe_divide(contact_hits, len(CONTACT_PATTERNS)) * 100

    jd_skills = skill_details.get("jd_skills", [])
    matching_skills = skill_details.get("matching_skills", [])
    keyword_score = _safe_divide(len(matching_skills), len(jd_skills)) * 100

    words = _word_count(resume_text)
    if 350 <= words <= 900:
        length_score = 100.0
    elif 250 <= words <= 1100:
        length_score = 78.0
    elif words > 0:
        length_score = 50.0
    else:
        length_score = 0.0

    bullet_lines = len(re.findall(r"^\s*[-*]\s+", resume_text or "", flags=re.MULTILINE))
    bullet_score = 100.0 if bullet_lines >= 4 else (75.0 if bullet_lines >= 2 else 45.0)

    overall = (
        (0.40 * keyword_score)
        + (0.22 * section_score)
        + (0.20 * contact_score)
        + (0.10 * length_score)
        + (0.08 * bullet_score)
    )

    tips: List[str] = []
    if keyword_score < 55:
        tips.append("Add more role-specific keywords from the JD into measurable experience bullets.")
    if section_score < 80:
        tips.append("Use standard section headers: Summary, Experience, Skills, Projects, Education.")
    if contact_score < 75:
        tips.append("Include complete contact details (email, phone, LinkedIn, GitHub).")
    if length_score < 70:
        tips.append("Keep the resume concise and scoped to role-relevant accomplishments.")
    if bullet_score < 70:
        tips.append("Use bullet points in experience sections to improve ATS parsing consistency.")
    if not tips:
        tips.append("Resume structure and keyword alignment are ATS-friendly.")

    return {
        "ats_score": round(_clamp_pct(overall), 2),
        "keyword_score": round(keyword_score, 2),
        "section_score": round(section_score, 2),
        "contact_score": round(contact_score, 2),
        "length_score": round(length_score, 2),
        "bullet_score": round(bullet_score, 2),
        "tips": tips[:5],
    }


def calculate_keyword_density(
    resume_text: str,
    jd_text: str,
    top_n: int = 15,
) -> List[Dict[str, Any]]:
    """Return top JD keywords and their frequency/density in the resume."""
    jd_keywords = extract_keywords(jd_text, top_n=top_n * 2)
    resume_lower = (resume_text or "").lower()
    total_words = max(_word_count(resume_text), 1)

    rows: List[Dict[str, Any]] = []
    seen = set()
    for keyword, jd_frequency in jd_keywords:
        if keyword in seen:
            continue
        seen.add(keyword)
        pattern = rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])"
        resume_frequency = len(re.findall(pattern, resume_lower))
        density = (resume_frequency / total_words) * 100
        rows.append(
            {
                "keyword": keyword,
                "jd_frequency": int(jd_frequency),
                "resume_frequency": int(resume_frequency),
                "density_pct": round(density, 3),
                "status": "Matched" if resume_frequency > 0 else "Missing",
            }
        )
        if len(rows) >= top_n:
            break

    return rows
