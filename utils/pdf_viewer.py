"""PDF preview and keyword highlight helpers for Streamlit UX."""

from __future__ import annotations

import html
import re
from typing import List

import fitz
import streamlit as st


def render_pdf_preview(uploaded_file, title: str, height: int = 500) -> None:
    """Render uploaded PDF as page images (iframe-free, Space-safe)."""
    del height  # maintained for backward compatibility with existing calls
    if uploaded_file is None:
        st.caption(f"{title}: no file uploaded.")
        return

    st.markdown(f"**{title}**", unsafe_allow_html=False)

    try:
        if isinstance(uploaded_file, (bytes, bytearray)):
            data = bytes(uploaded_file)
        else:
            data = uploaded_file.read()
            uploaded_file.seek(0)
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception:
        st.warning(f"{title}: unable to load this PDF.")
        return

    if doc.page_count == 0:
        st.caption(f"{title}: empty PDF.")
        return

    st.caption(f"Pages: {doc.page_count}")
    key_base = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") or "pdf"
    page_number = st.number_input(
        "Preview page",
        min_value=1,
        max_value=doc.page_count,
        value=1,
        step=1,
        key=f"{key_base}_page_selector",
    )

    try:
        page = doc.load_page(int(page_number) - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(1.4, 1.4), alpha=False)
        st.image(pix.tobytes("png"), use_container_width=True)
    except Exception:
        st.warning(f"{title}: failed to render selected page.")


def highlight_keyword_snippets(text: str, keyword: str, window: int = 140, max_snippets: int = 6) -> List[str]:
    """Return HTML snippets with highlighted keyword matches."""
    if not text or not keyword:
        return []
    pattern = re.compile(re.escape(keyword), flags=re.IGNORECASE)
    matches = list(pattern.finditer(text))
    snippets: List[str] = []
    for match in matches[:max_snippets]:
        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)
        snippet = html.escape(text[start:end])
        snippet = re.sub(
            re.escape(html.escape(match.group(0))),
            lambda m: f"<mark>{m.group(0)}</mark>",
            snippet,
            flags=re.IGNORECASE,
        )
        snippets.append(snippet)
    return snippets
