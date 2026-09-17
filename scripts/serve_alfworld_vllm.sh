#!/usr/bin/env bash
# Serve Qwen3-8B BF16 with all exported ALFWorld skill adapters.
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_PATH=${MODEL_PATH:-/media/public/models/huggingface/Qwen/Qwen3-8B}
ADAPTER_ROOT=${ADAPTER_ROOT:-$PROJECT_ROOT/artifacts/alfworld_skill_loras}
VLLM_BIN=${VLLM_BIN:-$PROJECT_ROOT/runtime/vllm/.venv/bin/vllm}
PORT=${PORT:-8000}
VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$PROJECT_ROOT/runtime/vllm/cache/vllm}
TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$PROJECT_ROOT/runtime/vllm/cache/torchinductor}
export VLLM_CACHE_ROOT TORCHINDUCTOR_CACHE_DIR
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR"

loras=()
for skill in pick_and_place cool heat clean look_at_obj_in_light; do
    test -f "$ADAPTER_ROOT/$skill/adapter_model.safetensors"
    loras+=("latentskill-$skill=$ADAPTER_ROOT/$skill")
done

exec "$VLLM_BIN" serve "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --dtype bfloat16 \
    --generation-config vllm \
    --max-model-len 8192 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.92}" \
    --enable-prefix-caching \
    --enable-lora \
    --max-loras 5 \
    --max-cpu-loras 5 \
    --max-lora-rank 8 \
    --lora-modules "${loras[@]}"
