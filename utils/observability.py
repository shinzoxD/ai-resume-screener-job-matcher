"""Simple observability helpers for latency and LLM cost tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class StageMetric:
    stage: str
    elapsed_ms: float


@dataclass
class AnalysisMetrics:
    """Collect stage timings and token/cost estimates for one analysis run."""

    started_at: float = field(default_factory=time.perf_counter)
    stage_timings: List[StageMetric] = field(default_factory=list)
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_estimated_cost_usd: float = 0.0
    _stage_start: Dict[str, float] = field(default_factory=dict)

    def start_stage(self, stage: str) -> None:
        self._stage_start[stage] = time.perf_counter()

    def end_stage(self, stage: str) -> None:
        start = self._stage_start.get(stage)
        if start is None:
            return
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.stage_timings.append(StageMetric(stage=stage, elapsed_ms=round(elapsed_ms, 2)))
        self._stage_start.pop(stage, None)

    def set_llm_usage(self, prompt_tokens: int, completion_tokens: int, estimated_cost_usd: float) -> None:
        self.llm_prompt_tokens = int(prompt_tokens)
        self.llm_completion_tokens = int(completion_tokens)
        self.llm_estimated_cost_usd = float(round(max(estimated_cost_usd, 0.0), 6))

    def to_dict(self) -> Dict[str, object]:
        total_ms = (time.perf_counter() - self.started_at) * 1000
        return {
            "total_elapsed_ms": round(total_ms, 2),
            "stage_timings": [
                {"stage": row.stage, "elapsed_ms": row.elapsed_ms} for row in self.stage_timings
            ],
            "llm_prompt_tokens": self.llm_prompt_tokens,
            "llm_completion_tokens": self.llm_completion_tokens,
            "llm_estimated_cost_usd": self.llm_estimated_cost_usd,
        }

