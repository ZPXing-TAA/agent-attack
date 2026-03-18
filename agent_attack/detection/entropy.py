"""Utilities for entropy-based attack detection from token top-k logprobs."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    max_value = max(values)
    return max_value + math.log(sum(math.exp(v - max_value) for v in values))


def normalize_logprobs(logprobs: Sequence[float]) -> list[float]:
    """Normalize (possibly truncated) logprobs to a probability simplex."""
    if not logprobs:
        return []
    lse = _logsumexp(logprobs)
    return [math.exp(lp - lse) for lp in logprobs]


def token_entropy_from_logprobs(logprobs: Sequence[float]) -> float:
    """Compute entropy over top-k token logprobs.

    Notes:
        This is a truncated-entropy proxy when full-vocabulary probabilities
        are unavailable, which is still useful for consistent comparisons.
    """
    probs = normalize_logprobs(logprobs)
    if not probs:
        return 0.0
    return -sum(p * math.log(p + 1e-12) for p in probs)


def step_entropy_stats(token_entropies: Sequence[float], tail_fraction: float = 0.2) -> dict[str, float]:
    """Aggregate token entropy values for one semantic agent step."""
    if not token_entropies:
        return {
            "step_entropy_mean": 0.0,
            "step_entropy_max": 0.0,
            "step_entropy_std": 0.0,
            "step_entropy_tail_mean": 0.0,
        }

    n = len(token_entropies)
    mean_val = sum(token_entropies) / n
    max_val = max(token_entropies)
    var = sum((x - mean_val) ** 2 for x in token_entropies) / n
    std_val = math.sqrt(var)

    tail_size = max(1, int(math.ceil(n * tail_fraction)))
    tail_vals = sorted(token_entropies)[-tail_size:]
    tail_mean = sum(tail_vals) / len(tail_vals)

    return {
        "step_entropy_mean": mean_val,
        "step_entropy_max": max_val,
        "step_entropy_std": std_val,
        "step_entropy_tail_mean": tail_mean,
    }


def sliding_window_means(values: Sequence[float], window_size: int) -> list[float]:
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if len(values) < window_size:
        return []

    means: list[float] = []
    window_sum = sum(values[:window_size])
    means.append(window_sum / window_size)

    for idx in range(window_size, len(values)):
        window_sum += values[idx] - values[idx - window_size]
        means.append(window_sum / window_size)
    return means


def first_spike_index(values: Sequence[float], threshold: float) -> int:
    """Return the first index where value >= threshold, else -1."""
    for idx, value in enumerate(values):
        if value >= threshold:
            return idx
    return -1


def trajectory_entropy_features(
    token_entropies: Sequence[float],
    step_mean_entropies: Sequence[float],
    window_sizes: Iterable[int] = (8, 16, 32),
    spike_threshold: float | None = None,
) -> dict[str, float]:
    """Compute minimal trajectory-level feature set for entropy detection."""
    if token_entropies:
        trajectory_mean = sum(token_entropies) / len(token_entropies)
    else:
        trajectory_mean = 0.0

    window_entropy_mean_values: list[float] = []
    window_entropy_max_values: list[float] = []
    for w in window_sizes:
        if w <= 0:
            continue
        w_means = sliding_window_means(token_entropies, w)
        if w_means:
            window_entropy_mean_values.append(sum(w_means) / len(w_means))
            window_entropy_max_values.append(max(w_means))

    if spike_threshold is None:
        # Simple adaptive threshold for first-pass feasibility experiments.
        spike_threshold = trajectory_mean + 0.5

    return {
        "trajectory_entropy_mean": trajectory_mean,
        "window_entropy_mean": (
            sum(window_entropy_mean_values) / len(window_entropy_mean_values)
            if window_entropy_mean_values
            else 0.0
        ),
        "window_entropy_max": max(window_entropy_max_values) if window_entropy_max_values else 0.0,
        "first_spike_step": float(first_spike_index(step_mean_entropies, spike_threshold)),
    }
