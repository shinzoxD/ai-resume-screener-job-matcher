"""Confidence and calibration helpers for match score reporting."""

from __future__ import annotations

from statistics import pstdev
from typing import Any, Dict, List


def compute_confidence(score_details: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate confidence from agreement among feature-level scores."""
    feature_values: List[float] = []
    for key in ("semantic", "lexical", "skill_alignment", "reranker"):
        value = score_details.get(key)
        if value is not None:
            feature_values.append(float(value))

    if len(feature_values) < 2:
        return {"confidence_pct": 55.0, "band": "Medium", "calibrated_overall": score_details.get("overall", 0)}

    spread = pstdev(feature_values)
    confidence = max(35.0, min(98.0, 100.0 - (spread * 1.8)))

    if confidence >= 80:
        band = "High"
    elif confidence >= 60:
        band = "Medium"
    else:
        band = "Low"

    overall = float(score_details.get("overall", 0.0))
    calibration_factor = 0.92 + (confidence / 1000)  # mild calibration toward confidence.
    calibrated = max(0.0, min(100.0, overall * calibration_factor))

    return {
        "confidence_pct": round(confidence, 2),
        "band": band,
        "calibrated_overall": round(calibrated, 2),
        "feature_spread": round(spread, 2),
    }

