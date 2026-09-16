#!/usr/bin/env bash
# Verify the files and runtime required for LatentSkill ALFWorld evaluation.

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

failed=0

check_file() {
    if [ -s "$1" ]; then
        echo "[ok] $1"
    else
        echo "[missing] $1" >&2
        failed=1
    fi
}

check_file uv.lock
check_file alfworld_data/alfworld/logic/alfred.pddl
check_file alfworld_data/alfworld/logic/alfred.twl2
check_file checkpoints/latentskill_pretrain_qwen3_8b/checkpoint-epoch-10/metanetwork.pth
check_file checkpoints/latentskill_pretrain_qwen3_8b/checkpoint-epoch-10/metalora.pth
check_file checkpoints/latentskill_sft_qwen3_8b/checkpoint-epoch-10/metanetwork.pth
check_file checkpoints/latentskill_sft_qwen3_8b/checkpoint-epoch-10/metalora.pth

traj_count=$(find alfworld_data/alfworld/json_2.1.1 -name traj_data.json -type f | wc -l)
game_count=$(find alfworld_data/alfworld/json_2.1.1 -name game.tw-pddl -type f | wc -l)
echo "[info] ALFWorld traj_data.json: ${traj_count} (expected 7080)"
echo "[info] ALFWorld game.tw-pddl:  ${game_count} (expected 4027)"
if [ "$traj_count" -ne 7080 ] || [ "$game_count" -ne 4027 ]; then
    failed=1
fi

if [ -s models/Qwen3-8B/model.safetensors.index.json ] || \
   find models/Qwen3-8B -maxdepth 1 -name '*.safetensors' -type f -size +1M | grep -q .; then
    echo "[ok] Qwen3-8B weights found"
else
    echo "[missing] Qwen3-8B weights are not present" >&2
    failed=1
fi

if [ -x .shared-runtime/venv/bin/python ]; then
    eval_python=.shared-runtime/venv/bin/python
    echo "[ok] using shared uv environment: ${eval_python}"
elif [ -x .venv/bin/python ]; then
    eval_python=.venv/bin/python
    echo "[ok] using local uv environment: ${eval_python}"
else
    echo "[missing] no executable uv environment found" >&2
    echo "          run: bash scripts/setup_shared_env.sh" >&2
    failed=1
    eval_python=
fi

if [ -n "$eval_python" ]; then
    "$eval_python" - <<'PY'
import torch

print(f"[info] torch={torch.__version__} cuda_runtime={torch.version.cuda}")
print(f"[info] cuda_available={torch.cuda.is_available()} devices={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
x = torch.randn(64, 64, device="cuda")
_ = x @ x
torch.cuda.synchronize()
print(f"[ok] CUDA smoke test passed on {torch.cuda.get_device_name(0)}")
PY
fi

if [ "$failed" -ne 0 ]; then
    echo "Migration verification failed." >&2
    exit 1
fi

echo "Migration verification passed."
