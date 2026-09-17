#!/usr/bin/env bash
# Launch deterministic ALFWorld episode shards, one process per listed GPU.
# Run this launcher inside tmux on the target host.
# GPU groups are separated by semicolons; commas join GPUs used by one worker.
# Usage: bash scripts/eval_alfworld_sft_parallel.sh seen '0,1;2,3;4,5;6,7' all

set -euo pipefail

SPLIT=${1:-seen}
GPU_GROUPS=${2:-0,1;2,3;4,5;6,7}
MAX_GAMES=${3:-all}
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

IFS=';' read -r -a GPU_GROUP_ARRAY <<< "$GPU_GROUPS"
NUM_SHARDS=${#GPU_GROUP_ARRAY[@]}
if [ "$NUM_SHARDS" -lt 1 ]; then
    echo "At least one GPU is required" >&2
    exit 2
fi

if [ "$MAX_GAMES" = "all" ]; then
    RUN_TAG="sft_epoch10_${SPLIT}_parallel_${NUM_SHARDS}way"
else
    RUN_TAG="sft_epoch10_${SPLIT}_${MAX_GAMES}games_parallel_${NUM_SHARDS}way"
fi
OUTPUT_DIR=${OUTPUT_DIR:-evals/alfworld/results/${RUN_TAG}}
LOG_DIR=${LOG_DIR:-evals/alfworld/logs/${RUN_TAG}}
PYTHON_BIN=${PYTHON_BIN:-.shared-runtime/venv/bin/python}
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

pids=()
for shard_index in "${!GPU_GROUP_ARRAY[@]}"; do
    gpu_group=${GPU_GROUP_ARRAY[$shard_index]}
    log_file="$LOG_DIR/shard$(printf '%03d' "$shard_index").log"
    args=(
        -m evals.alfworld.evaluate
        --checkpoint "${CHECKPOINT:-checkpoints/latentskill_sft_qwen3_8b/checkpoint-epoch-10}"
        --config_name models/qwen3_8b
        --split "$SPLIT"
        --alfworld_data "${ALFWORLD_DATA:-alfworld_data/alfworld}"
        --alfworld_config "${ALFWORLD_CONFIG:-evals/alfworld/config_tw.yaml}"
        --skill_context_dir "${SKILL_CONTEXT_DIR:-evals/alfworld/skills}"
        --max_steps "${MAX_STEPS:-50}"
        --max_new_tokens "${MAX_NEW_TOKENS:-2048}"
        --history_length 5
        --context_max_length 4096
        --conversation_max_length 4096
        --output_dir "$OUTPUT_DIR"
        --device cuda
        --dtype "${MODEL_DTYPE:-float32}"
        --num_shards "$NUM_SHARDS"
        --shard_index "$shard_index"
    )
    if [[ "$gpu_group" == *,* ]]; then
        args+=(--model_parallel)
    fi
    if [ "$MAX_GAMES" != "all" ]; then
        args+=(--max_games "$MAX_GAMES")
    fi
    echo "launch shard=$shard_index/$NUM_SHARDS gpus=$gpu_group log=$log_file"
    CUDA_VISIBLE_DEVICES="$gpu_group" MODEL_PATH="${MODEL_PATH:-/media/public/models/huggingface/Qwen/Qwen3-8B}" \
        PYTHONPATH="$PROJECT_ROOT" PYTHONUNBUFFERED=1 \
        "$PYTHON_BIN" "${args[@]}" >"$log_file" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
if [ "$status" -ne 0 ]; then
    echo "At least one shard failed; inspect $LOG_DIR" >&2
    exit "$status"
fi

merge_args=(
    -m evals.alfworld.merge_shards
    --input_dir "$OUTPUT_DIR"
    --split "$SPLIT"
    --num_shards "$NUM_SHARDS"
)
if [ "$MAX_GAMES" != "all" ]; then
    EXPECTED_EPISODES=$MAX_GAMES
elif [ "$SPLIT" = "seen" ]; then
    EXPECTED_EPISODES=${EXPECTED_EPISODES:-140}
else
    EXPECTED_EPISODES=${EXPECTED_EPISODES:-134}
fi
merge_args+=(--expected_episodes "$EXPECTED_EPISODES")
"$PYTHON_BIN" "${merge_args[@]}"
