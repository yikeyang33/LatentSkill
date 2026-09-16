
#!/usr/bin/env bash
# Run the released epoch-10 SFT checkpoint on ALFWorld in the foreground.
# Launch this script inside tmux for remote evaluations.
#
# Usage:
#   bash scripts/eval_alfworld_sft.sh [seen|unseen] [max_games|all] [gpu_id]

set -euo pipefail

SPLIT=${1:-seen}
MAX_GAMES=${2:-all}
GPU_ID=${3:-0}
MAX_STEPS=${MAX_STEPS:-50}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
DEBUG_PROMPT=${DEBUG_PROMPT:-0}

if [ "$SPLIT" != "seen" ] && [ "$SPLIT" != "unseen" ]; then
    echo "split must be seen or unseen" >&2
    exit 2
fi

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

MODEL_PATH=${MODEL_PATH:-/media/public/models/huggingface/Qwen/Qwen3-8B}
CHECKPOINT=${CHECKPOINT:-checkpoints/latentskill_sft_qwen3_8b/checkpoint-epoch-10}
ALFWORLD_DATA=${ALFWORLD_DATA:-alfworld_data/alfworld}
ALFWORLD_CONFIG=${ALFWORLD_CONFIG:-evals/alfworld/config_tw.yaml}
SKILL_CONTEXT_DIR=${SKILL_CONTEXT_DIR:-evals/alfworld/skills}
PYTHON_BIN=${PYTHON_BIN:-.shared-runtime/venv/bin/python}

if [ "$MAX_GAMES" = "all" ]; then
    run_tag="sft_epoch10_${SPLIT}_full"
else
    run_tag="sft_epoch10_${SPLIT}_${MAX_GAMES}games"
fi

OUTPUT_DIR=${OUTPUT_DIR:-evals/alfworld/results/${run_tag}}
LOG_DIR=${LOG_DIR:-evals/alfworld/logs}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/${run_tag}.log}
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

args=(
    -m evals.alfworld.evaluate
    --checkpoint "$CHECKPOINT"
    --config_name models/qwen3_8b
    --split "$SPLIT"
    --alfworld_data "$ALFWORLD_DATA"
    --alfworld_config "$ALFWORLD_CONFIG"
    --skill_context_dir "$SKILL_CONTEXT_DIR"
    --max_steps "$MAX_STEPS"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --history_length 5
    --context_max_length 4096
    --conversation_max_length 4096
    --output_dir "$OUTPUT_DIR"
    --device cuda
)

if [ "$MAX_GAMES" != "all" ]; then
    args+=(--max_games "$MAX_GAMES")
fi

if [ "$DEBUG_PROMPT" = "1" ]; then
    args+=(--debug_prompt --debug_episodes 1 --debug_steps 1)
fi

echo "Model      : $MODEL_PATH"
echo "Checkpoint : $CHECKPOINT"
echo "Split      : $SPLIT"
echo "Max games  : $MAX_GAMES"
echo "Max steps  : $MAX_STEPS"
echo "Max tokens : $MAX_NEW_TOKENS"
echo "GPU        : $GPU_ID"
echo "Output     : $OUTPUT_DIR"
echo "Log        : $LOG_FILE"

export MODEL_PATH
export PYTHONPATH="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

set +e
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" "${args[@]}" 2>&1 | tee "$LOG_FILE"
status=${PIPESTATUS[0]}
set -e
exit "$status"
