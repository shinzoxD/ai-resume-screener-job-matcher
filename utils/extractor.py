"""Utilities for extracting, cleaning, and structuring resume / JD text."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency path
    Image = None  # type: ignore

try:
    import pytesseract
except Exception:  # pragma: no cover - optional dependency path
    pytesseract = None  # type: ignore

SECTION_PATTERNS: Dict[str, List[str]] = {
    "summary": ["summary", "profile", "objective", "professional summary"],
    "experience": ["experience", "professional experience", "employment history", "work history"],
    "projects": ["projects", "project experience", "portfolio"],
    "skills": ["skills", "technical skills", "core competencies"],
    "education": ["education", "academic background", "qualifications"],
    "certifications": ["certifications", "licenses"],
}


@dataclass
class ExtractionResult:
    """Structured extraction result for downstream scoring and explainability."""

    text: str
    metadata: Dict[str, Any]


def clean_text(text: str) -> str:
    """Normalize whitespace and remove common encoding artifacts."""
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2022", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    return text.strip()


def _read_file_bytes(file_or_bytes: Any) -> bytes:
    """Read bytes from UploadedFile, bytes object, or filesystem path."""
    if file_or_bytes is None:
        return b""
    if isinstance(file_or_bytes, (bytes, bytearray)):
        return bytes(file_or_bytes)
    if isinstance(file_or_bytes, (str, Path)):
        return Path(file_or_bytes).read_bytes()
    if hasattr(file_or_bytes, "read"):
        data = file_or_bytes.read()
        if hasattr(file_or_bytes, "seek"):
            file_or_bytes.seek(0)
        return data or b""
    raise TypeError("Unsupported file input type")


def _extract_text_with_ocr(document: fitz.Document) -> str:
    """Fallback OCR extraction for scanned PDFs when direct text is sparse."""
    if Image is None or pytesseract is None:
        return ""

    ocr_chunks: List[str] = []
    for page in document:
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.open(io.BytesIO(pix.tobytes("png")))
            page_text = pytesseract.image_to_string(image) or ""
            ocr_chunks.append(page_text)
        except Exception:
            continue

    return clean_text("\n".join(ocr_chunks))


def detect_sections(text: str) -> Dict[str, bool]:
    """Detect whether common resume/JD sections exist in extracted text."""
    lowered = (text or "").lower()
    results: Dict[str, bool] = {}
    for section, labels in SECTION_PATTERNS.items():
        results[section] = any(re.search(rf"\b{re.escape(label)}\b", lowered) for label in labels)
    return results


def segment_sections(text: str) -> Dict[str, str]:
    """Split text into approximate named sections based on heading lines."""
    lines = (text or "").splitlines()
    sections: Dict[str, List[str]] = {"other": []}
    current = "other"

    heading_lookup = {
        label: key
        for key, values in SECTION_PATTERNS.items()
        for label in values
    }

    for raw_line in lines:
        line = raw_line.strip()
        lower = line.lower()
        matched_key = None
        for label, key in heading_lookup.items():
            if lower == label or lower.startswith(f"{label}:"):
                matched_key = key
                break

        if matched_key:
            current = matched_key
            sections.setdefault(current, [])
            continue

        sections.setdefault(current, []).append(raw_line)

    return {key: clean_text("\n".join(value)) for key, value in sections.items() if clean_text("\n".join(value))}


def extract_text_from_pdf(file_or_bytes: Any, use_ocr_fallback: bool = True) -> str:
    """Extract full text from a PDF file with optional OCR fallback."""
    data = _read_file_bytes(file_or_bytes)
    if not data:
        return ""

    try:
        document = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ValueError("Unable to parse PDF. Please upload a valid PDF file.") from exc

    try:
        direct_chunks = [page.get_text("text") for page in document]
        direct_text = clean_text("\n".join(direct_chunks))
        if direct_text:
            # If the parser returns meaningful text, keep the fast path.
            if len(direct_text.split()) >= 30 or not use_ocr_fallback:
                return direct_text

        if use_ocr_fallback:
            ocr_text = _extract_text_with_ocr(document)
            if len(ocr_text.split()) > len(direct_text.split()):
                return ocr_text

        return direct_text
    finally:
        document.close()


def extract_text_from_txt(file_or_bytes: Any) -> str:
    """Extract text from plain text bytes/file."""
    data = _read_file_bytes(file_or_bytes)
    if not data:
        return ""

    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return clean_text(data.decode(encoding))
        except UnicodeDecodeError:
            continue

    return clean_text(data.decode("utf-8", errors="ignore"))


def extract_text_from_input(
    uploaded_file: Any | None,
    pasted_text: str | None = None,
    use_ocr_fallback: bool = True,
) -> str:
    """Resolve text either from uploaded PDF/TXT or pasted text input."""
    if uploaded_file is not None:
        filename = getattr(uploaded_file, "name", "").lower()
        if filename.endswith(".pdf"):
            return extract_text_from_pdf(uploaded_file, use_ocr_fallback=use_ocr_fallback)
        return extract_text_from_txt(uploaded_file)

    return clean_text(pasted_text or "")


def extract_document(
    uploaded_file: Any | None,
    pasted_text: str | None = None,
    use_ocr_fallback: bool = True,
) -> ExtractionResult:
    """Extract text and useful metadata in a single helper call."""
    text = extract_text_from_input(
        uploaded_file=uploaded_file,
        pasted_text=pasted_text,
        use_ocr_fallback=use_ocr_fallback,
    )
    metadata = {
        "word_count": len(re.findall(r"\b\w+\b", text)),
        "sections": detect_sections(text),
        "section_text": segment_sections(text),
    }
    return ExtractionResult(text=text, metadata=metadata)
