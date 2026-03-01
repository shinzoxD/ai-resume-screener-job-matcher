from __future__ import annotations

from utils.extractor import clean_text, detect_sections


def test_clean_text_normalizes_spacing() -> None:
    raw = "Hello   world \n\n\n This\tis  test"
    cleaned = clean_text(raw)
    assert "Hello world" in cleaned
    assert "\n\n\n" not in cleaned


def test_detect_sections_finds_common_headers() -> None:
    text = """
    Summary
    Experience
    Technical Skills
    Education
    """
    sections = detect_sections(text)
    assert sections["summary"] is True
    assert sections["experience"] is True
    assert sections["skills"] is True
    assert sections["education"] is True

