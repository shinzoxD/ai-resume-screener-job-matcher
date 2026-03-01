"""Offline evaluation for resume-job scoring quality."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.matcher import calculate_match_score


def load_pairs(dataset_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def evaluate(
    dataset_path: Path,
    model_name: str,
    use_reranker: bool,
    reranker_model: str,
) -> Dict[str, Any]:
    pairs = load_pairs(dataset_path)
    if not pairs:
        raise ValueError("Dataset is empty.")

    outputs: List[Dict[str, Any]] = []
    for row in pairs:
        score = calculate_match_score(
            resume_text=row["resume_text"],
            jd_text=row["jd_text"],
            model_name=model_name,
            use_reranker=use_reranker,
            reranker_model=reranker_model,
        )
        outputs.append(
            {
                "id": row.get("id"),
                "human_score": float(row["human_score"]),
                "predicted_score": float(score["overall"]),
                "semantic": float(score["semantic"]),
                "lexical": float(score["lexical"]),
                "skill_alignment": float(score["skill_alignment"]),
            }
        )

    df = pd.DataFrame(outputs)
    mae = mean_absolute_error(df["human_score"], df["predicted_score"])
    rmse = math.sqrt(mean_squared_error(df["human_score"], df["predicted_score"]))
    pearson_val, _ = pearsonr(df["human_score"], df["predicted_score"])
    spearman_val, _ = spearmanr(df["human_score"], df["predicted_score"])

    return {
        "num_pairs": int(len(df)),
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "pearson": round(float(pearson_val), 4),
        "spearman": round(float(spearman_val), 4),
        "details": df,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate resume matching against labeled pairs.")
    parser.add_argument("--dataset", type=Path, default=Path("data/eval_pairs.jsonl"))
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--use-reranker", action="store_true")
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--output", type=Path, default=Path("artifacts/eval_results.csv"))
    args = parser.parse_args()

    results = evaluate(
        dataset_path=args.dataset,
        model_name=args.model,
        use_reranker=args.use_reranker,
        reranker_model=args.reranker_model,
    )
    details = results.pop("details")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(args.output, index=False)

    print("Evaluation Summary")
    for key, value in results.items():
        print(f"{key}: {value}")
    print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()
