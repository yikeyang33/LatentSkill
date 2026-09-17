#!/usr/bin/env python3
"""Merge deterministic ALFWorld episode shards and reproduce the summary."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


TASK_ORDER = [
    "pick_and_place",
    "pick_two_and_place",
    "clean",
    "heat",
    "cool",
    "look_at_obj_in_light",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--split", choices=("seen", "unseen"), required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--expected_episodes", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = []
    missing_files = []
    for shard_index in range(args.num_shards):
        path = args.input_dir / (
            f"results_detail_{args.split}_shard{shard_index:03d}-of-{args.num_shards:03d}.jsonl"
        )
        if not path.is_file():
            missing_files.append(str(path))
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("shard_index") != shard_index:
                raise ValueError(f"{path}:{line_number}: wrong shard_index")
            records.append(record)

    if missing_files:
        raise FileNotFoundError("missing shard files:\n  " + "\n  ".join(missing_files))

    records.sort(key=lambda record: record["episode_idx"])
    episode_indices = [record["episode_idx"] for record in records]
    if len(episode_indices) != len(set(episode_indices)):
        raise ValueError("duplicate episode_idx across shards")
    if args.expected_episodes is not None:
        expected = list(range(args.expected_episodes))
        if episode_indices != expected:
            missing = sorted(set(expected) - set(episode_indices))
            extra = sorted(set(episode_indices) - set(expected))
            raise ValueError(f"episode coverage mismatch: missing={missing} extra={extra}")

    detail_path = args.input_dir / f"results_detail_{args.split}.jsonl"
    with detail_path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    task_results = defaultdict(list)
    for record in records:
        task_results[record["task_type"]].append(bool(record["won"]))
    summary = {}
    all_results = []
    for task in TASK_ORDER:
        results = task_results.get(task, [])
        success = sum(results)
        total = len(results)
        summary[task] = {
            "success": success,
            "total": total,
            "rate": round(success / total, 4) if total else None,
        }
        all_results.extend(results)
    summary["overall"] = {
        "success": sum(all_results),
        "total": len(all_results),
        "rate": round(sum(all_results) / len(all_results), 4) if all_results else 0.0,
    }
    summary_path = args.input_dir / f"results_summary_{args.split}.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"merged {len(records)} episodes")
    print(f"detail:  {detail_path}")
    print(f"summary: {summary_path}")
    print(json.dumps(summary["overall"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
