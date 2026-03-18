"""Logprob extraction + trajectory logging helpers for entropy detection experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .entropy import step_entropy_stats, token_entropy_from_logprobs, trajectory_entropy_features


@dataclass
class TopKTokenLogprob:
    token: str
    logprob: float


@dataclass
class GeneratedTokenLogprobs:
    token: str
    logprob: float
    topk: list[TopKTokenLogprob] = field(default_factory=list)


@dataclass
class GenerationStepRecord:
    step_idx: int
    generated_text: str
    tokens: list[GeneratedTokenLogprobs] = field(default_factory=list)


@dataclass
class TrajectoryLogRecord:
    task_id: str
    trajectory_id: str
    label: str
    attack_type: str | None = None
    steps: list[GenerationStepRecord] = field(default_factory=list)


def _get(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def parse_openai_logprobs_content(content: list[Any] | None) -> list[GeneratedTokenLogprobs]:
    """Parse OpenAI/vLLM chat completion `logprobs.content` payload to a normalized schema."""
    if not content:
        return []

    parsed: list[GeneratedTokenLogprobs] = []
    for token_obj in content:
        top_items = _get(token_obj, "top_logprobs", []) or []
        parsed_top = [
            TopKTokenLogprob(token=str(_get(top, "token", "")), logprob=float(_get(top, "logprob", 0.0)))
            for top in top_items
        ]
        parsed.append(
            GeneratedTokenLogprobs(
                token=str(_get(token_obj, "token", "")),
                logprob=float(_get(token_obj, "logprob", 0.0)),
                topk=parsed_top,
            )
        )
    return parsed


def append_generation_step(record: TrajectoryLogRecord, step_idx: int, generated_text: str, content: list[Any] | None) -> None:
    tokens = parse_openai_logprobs_content(content)
    record.steps.append(GenerationStepRecord(step_idx=step_idx, generated_text=generated_text, tokens=tokens))


def compute_entropy_features_from_record(record: TrajectoryLogRecord) -> dict[str, float]:
    token_entropies: list[float] = []
    step_mean_entropies: list[float] = []

    for step in record.steps:
        step_token_entropies: list[float] = []
        for token in step.tokens:
            if token.topk:
                token_topk_logprobs = [candidate.logprob for candidate in token.topk]
            else:
                token_topk_logprobs = [token.logprob]
            entropy = token_entropy_from_logprobs(token_topk_logprobs)
            token_entropies.append(entropy)
            step_token_entropies.append(entropy)

        step_stats = step_entropy_stats(step_token_entropies)
        step_mean_entropies.append(step_stats["step_entropy_mean"])

    base = trajectory_entropy_features(token_entropies, step_mean_entropies)
    return {
        "token_entropy": sum(token_entropies) / len(token_entropies) if token_entropies else 0.0,
        "step_entropy_mean": sum(step_mean_entropies) / len(step_mean_entropies) if step_mean_entropies else 0.0,
        "step_entropy_max": max(step_mean_entropies) if step_mean_entropies else 0.0,
        **base,
    }


def save_trajectory_log(record: TrajectoryLogRecord, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(record)
    payload["entropy_features"] = compute_entropy_features_from_record(record)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
