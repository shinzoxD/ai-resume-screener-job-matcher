"""Language detection and multilingual model routing helpers."""

from __future__ import annotations

import re
from typing import Dict

MULTILINGUAL_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Lightweight keyword hints for common languages.
LANG_HINTS: Dict[str, tuple[str, ...]] = {
    "es": ("experiencia", "habilidades", "proyecto", "educación"),
    "fr": ("expérience", "compétences", "projet", "éducation"),
    "de": ("erfahrung", "fähigkeiten", "projekt", "bildung"),
    "hi": ("अनुभव", "कौशल", "परियोजना", "शिक्षा"),
    "pt": ("experiência", "habilidades", "projeto", "educação"),
}


def detect_language(text: str) -> str:
    """Detect language with simple heuristics to avoid heavyweight dependencies."""
    snippet = (text or "").strip().lower()[:2000]
    if not snippet:
        return "unknown"

    # Basic script detection for Hindi/Devanagari.
    if re.search(r"[\u0900-\u097F]", snippet):
        return "hi"

    for language, hints in LANG_HINTS.items():
        if any(hint in snippet for hint in hints):
            return language

    # Default to English for latin text.
    if re.search(r"[a-zA-Z]", snippet):
        return "en"
    return "unknown"


def choose_embedding_model(default_model: str, language: str, auto_multilingual: bool) -> str:
    """Route to multilingual model when auto mode is enabled and language is non-English."""
    if auto_multilingual and language not in {"en", "unknown"}:
        return MULTILINGUAL_MODEL
    return default_model


def language_label(language: str) -> str:
    labels = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "hi": "Hindi",
        "pt": "Portuguese",
        "unknown": "Unknown",
    }
    return labels.get(language, language.upper())

