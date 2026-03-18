"""Compute entropy detection features from saved trajectory logprob JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_attack.detection import TrajectoryLogRecord, compute_entropy_features_from_record
from agent_attack.detection.logprob_logging import (
    GeneratedTokenLogprobs,
    GenerationStepRecord,
    TopKTokenLogprob,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute entropy features for trajectory log files.")
    parser.add_argument("--input", type=str, required=True, help="Path to one trajectory JSON or a directory of JSON files")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSONL file")
    return parser.parse_args()


def load_record(path: Path) -> TrajectoryLogRecord:
    payload = json.loads(path.read_text(encoding="utf-8"))
    steps = []
    for step in payload.get("steps", []):
        tokens = []
        for token in step.get("tokens", []):
            topk = [TopKTokenLogprob(token=item["token"], logprob=float(item["logprob"])) for item in token.get("topk", [])]
            tokens.append(
                GeneratedTokenLogprobs(
                    token=token.get("token", ""),
                    logprob=float(token.get("logprob", 0.0)),
                    topk=topk,
                )
            )
        steps.append(
            GenerationStepRecord(
                step_idx=int(step.get("step_idx", 0)),
                generated_text=step.get("generated_text", ""),
                tokens=tokens,
            )
        )

    return TrajectoryLogRecord(
        task_id=payload.get("task_id", path.stem),
        trajectory_id=payload.get("trajectory_id", path.stem),
        label=payload.get("label", "unknown"),
        attack_type=payload.get("attack_type"),
        steps=steps,
    )


def iter_json_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(p for p in input_path.glob("*.json") if p.is_file())
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    paths = iter_json_files(input_path)
    with output_path.open("w", encoding="utf-8") as f:
        for path in paths:
            record = load_record(path)
            features = compute_entropy_features_from_record(record)
            row = {
                "task_id": record.task_id,
                "trajectory_id": record.trajectory_id,
                "label": record.label,
                "attack_type": record.attack_type,
                **features,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(paths)} trajectories to {output_path}")


if __name__ == "__main__":
    main()
