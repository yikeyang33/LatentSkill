#!/usr/bin/env python3
"""Repartition completed ALFWorld JSONL records for a wider resumed run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split", choices=("seen", "unseen"), required=True)
    parser.add_argument("--source_shards", type=int, required=True)
    parser.add_argument("--target_shards", type=int, required=True)
    return parser.parse_args()


def detail_path(root: Path, split: str, shard: int, count: int) -> Path:
    return root / f"results_detail_{split}_shard{shard:03d}-of-{count:03d}.jsonl"


def main() -> int:
    args = parse_args()
    if args.target_shards < 1:
        raise ValueError("target_shards must be positive")
    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output directory: {args.output_dir}"
        )

    records = {}
    for source_shard in range(args.source_shards):
        source = detail_path(
            args.input_dir, args.split, source_shard, args.source_shards
        )
        if not source.is_file():
            raise FileNotFoundError(source)
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            record = json.loads(line)
            episode_idx = record["episode_idx"]
            if episode_idx in records:
                raise ValueError(
                    f"{source}:{line_number}: duplicate episode {episode_idx}"
                )
            records[episode_idx] = record

    args.output_dir.mkdir(parents=True)
    outputs = [
        detail_path(args.output_dir, args.split, shard, args.target_shards).open(
            "w", encoding="utf-8"
        )
        for shard in range(args.target_shards)
    ]
    try:
        for episode_idx, original in sorted(records.items()):
            target_shard = episode_idx % args.target_shards
            record = dict(original)
            record["shard_index"] = target_shard
            record["num_shards"] = args.target_shards
            outputs[target_shard].write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
    finally:
        for output in outputs:
            output.close()

    counts = [0] * args.target_shards
    for episode_idx in records:
        counts[episode_idx % args.target_shards] += 1
    print(f"repartitioned {len(records)} completed episodes")
    print(f"source shards: {args.source_shards}")
    print(f"target shards: {args.target_shards}")
    print("records per target shard:", ",".join(map(str, counts)))
    print(f"output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
