"""PDF preview and keyword highlight helpers for Streamlit UX."""

from __future__ import annotations

import base64
import html
import re
from typing import List

import streamlit as st


def render_pdf_preview(uploaded_file, title: str, height: int = 500) -> None:
    """Render uploaded PDF inline using an iframe."""
    if uploaded_file is None:
        st.caption(f"{title}: no file uploaded.")
        return
    try:
        if isinstance(uploaded_file, (bytes, bytearray)):
            data = bytes(uploaded_file)
        else:
            data = uploaded_file.read()
            uploaded_file.seek(0)
        encoded = base64.b64encode(data).decode("utf-8")
        iframe = (
            f"<iframe src='data:application/pdf;base64,{encoded}' width='100%' height='{height}' "
            "style='border:1px solid #2f2f2f;border-radius:8px;'></iframe>"
        )
        st.markdown(f"**{title}**", unsafe_allow_html=False)
        st.markdown(iframe, unsafe_allow_html=True)
    except Exception:
        st.warning(f"{title}: unable to preview this PDF in-browser.")


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
