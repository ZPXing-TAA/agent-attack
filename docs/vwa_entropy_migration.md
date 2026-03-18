# VWA Entropy Detection Migration Guide

This repo now includes a minimal entropy-detection pipeline aligned with `vwa_entropy_detection_summary.md`.

## 1) Hook point in model generation path

For OpenAI-compatible backends, the best hook point is the chat completion response where `choices[0].logprobs.content` is returned.

Implemented wrappers:

- `agent_attack/models/vllm.py`
- `agent_attack/models/gpt4v.py`

Both wrappers now support:

- requesting logprobs by default (`logprobs=True`, `top_logprobs=10`)
- `return_generation_payloads=True` in `generate_answer(...)`
- returning `(responses, payloads)` where `payloads` is token-level top-k logprob content

## 2) Trajectory logging schema

Use `agent_attack/detection/logprob_logging.py`:

- `TrajectoryLogRecord`
- `append_generation_step(...)`
- `save_trajectory_log(...)`

This writes a JSON record compatible with the structure in `vwa_entropy_detection_summary.md` (task id, trajectory id, label, steps, token top-k logprobs).

## 3) Entropy features implemented

`agent_attack/detection/entropy.py` provides:

- token entropy proxy from top-k logprobs
- step aggregation (`mean/max/std/tail mean`)
- sliding-window means
- trajectory summary
- first spike index

`compute_entropy_features_from_record(...)` returns the requested minimal feature set:

- `token_entropy`
- `step_entropy_mean`
- `step_entropy_max`
- `window_entropy_mean`
- `window_entropy_max`
- `trajectory_entropy_mean`
- `first_spike_step`

## 4) Offline feature extraction script

```bash
python scripts/compute_entropy_features.py --input <trajectory_json_or_dir> --output <out.jsonl>
```

- Input can be one JSON or a directory of JSON files.
- Output is JSONL with one trajectory-level row per record.

## 5) Recommended integration in episode runner

When you run VWA/VWA-Adv episode loops, after each agent step generation:

1. call wrapper with `return_generation_payloads=True`
2. map one model call to one `step_idx`
3. append payload via `append_generation_step(...)`
4. at end of episode, call `save_trajectory_log(...)`

That keeps changes localized to generation call sites and avoids touching evaluator logic.
