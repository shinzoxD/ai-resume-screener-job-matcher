from __future__ import annotations

from utils.rewriter import (
    extract_bullets,
    filter_skill_like_keywords,
    render_bullet_diff_html,
    rewrite_bullet_rule_based,
)


def test_extract_bullets_handles_unicode_bullet() -> None:
    text = """
    - Built APIs with FastAPI
    * Improved AWS deployment
    • Automated CI/CD checks
    """
    bullets = extract_bullets(text)
    assert len(bullets) == 3


def test_filter_skill_like_keywords_removes_generic_terms() -> None:
    raw_keywords = ["Senior", "Engineer", "AWS", "PyTorch", "communication", "CI/CD"]
    filtered = filter_skill_like_keywords(raw_keywords, limit=10)
    lowered = {item.lower() for item in filtered}

    assert "senior" not in lowered
    assert "engineer" not in lowered
    assert "aws" in lowered
    assert "pytorch" in lowered
    assert "ci/cd" in lowered


def test_rewrite_rule_based_has_no_forced_suffixes() -> None:
    bullet = "built lead-scoring service in python"
    rewritten = rewrite_bullet_rule_based(bullet, ["Senior", "Engineer", "AWS"])

    assert rewritten.endswith(".")
    assert "improving key KPI by 15%" not in rewritten
    assert "using senior" not in rewritten.lower()
    assert "using engineer" not in rewritten.lower()


def test_render_diff_returns_dark_theme_markup() -> None:
    html_blob = render_bullet_diff_html(
        ["Built API service with Python."],
        ["Built API service with Python and FastAPI."],
    )
    assert "rw-shell" in html_blob
    assert "rw-add" in html_blob
    assert "rw-del" in html_blob
