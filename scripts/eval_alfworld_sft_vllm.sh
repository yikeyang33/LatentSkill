#!/usr/bin/env bash
# Launch many CPU ALFWorld clients against one continuous-batching vLLM server.
set -euo pipefail

SPLIT=${1:-seen}
NUM_CLIENTS=${2:-16}
MAX_GAMES=${3:-all}
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
RUN_TAG="sft_epoch10_${SPLIT}_${MAX_NEW_TOKENS}tok_vllm_${NUM_CLIENTS}clients"
OUTPUT_DIR=${OUTPUT_DIR:-evals/alfworld/results/$RUN_TAG}
LOG_DIR=${LOG_DIR:-evals/alfworld/logs/$RUN_TAG}
PYTHON_BIN=${PYTHON_BIN:-.shared-runtime/venv/bin/python}
VLLM_BASE_URL=${VLLM_BASE_URL:-http://127.0.0.1:8000}
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Refuse to start clients before the adapter models are visible.
"$PYTHON_BIN" - "$VLLM_BASE_URL" <<'PY'
import json, sys, urllib.request
with urllib.request.urlopen(sys.argv[1].rstrip('/') + '/v1/models', timeout=10) as r:
    models = {m['id'] for m in json.load(r)['data']}
required = {f'latentskill-{x}' for x in ['pick_and_place','cool','heat','clean','look_at_obj_in_light']}
missing = required - models
if missing:
    raise SystemExit(f'missing served adapters: {sorted(missing)}')
print('vLLM ready:', ', '.join(sorted(required)))
PY

pids=()
for ((shard=0; shard<NUM_CLIENTS; shard++)); do
    args=(
        -m evals.alfworld.evaluate
        --checkpoint checkpoints/latentskill_sft_qwen3_8b/checkpoint-epoch-10
        --config_name models/qwen3_8b
        --split "$SPLIT"
        --alfworld_data alfworld_data/alfworld
        --alfworld_config evals/alfworld/config_tw.yaml
        --skill_context_dir evals/alfworld/skills
        --max_steps 50
        --max_new_tokens "$MAX_NEW_TOKENS"
        --history_length 5
        --context_max_length 4096
        --conversation_max_length 4096
        --output_dir "$OUTPUT_DIR"
        --device cpu
        --dtype bfloat16
        --num_shards "$NUM_CLIENTS"
        --shard_index "$shard"
        --vllm_base_url "$VLLM_BASE_URL"
    )
    if [ "$MAX_GAMES" != all ]; then
        args+=(--max_games "$MAX_GAMES")
    fi
    PYTHONPATH="$PROJECT_ROOT" PYTHONUNBUFFERED=1 MODEL_PATH="${MODEL_PATH:-/media/public/models/huggingface/Qwen/Qwen3-8B}" \
        "$PYTHON_BIN" "${args[@]}" >"$LOG_DIR/client$(printf '%03d' "$shard").log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
if [ "$status" -ne 0 ]; then
    echo "At least one client failed; inspect $LOG_DIR" >&2
    exit "$status"
fi

expected=$MAX_GAMES
if [ "$MAX_GAMES" = all ]; then
    [ "$SPLIT" = seen ] && expected=140 || expected=134
fi
if [ "$NUM_CLIENTS" -eq 1 ]; then
    echo "Single client wrote the final unsharded result directly: $OUTPUT_DIR"
    exit 0
fi
"$PYTHON_BIN" -m evals.alfworld.merge_shards \
    --input_dir "$OUTPUT_DIR" --split "$SPLIT" \
    --num_shards "$NUM_CLIENTS" --expected_episodes "$expected"
