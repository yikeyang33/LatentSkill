#!/usr/bin/env bash
# Rebuild the isolated vLLM runtime entirely under the project's media path.
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUNTIME_ROOT="$PROJECT_ROOT/runtime/vllm"
UV_BIN=${UV_BIN:-/media/yangyike/uv/bin/uv}
PYTHON_VERSION=${PYTHON_VERSION:-3.12.11}

if [ ! -x "$UV_BIN" ]; then
    UV_BIN=$(command -v uv || true)
fi
if [ -z "$UV_BIN" ] || [ ! -x "$UV_BIN" ]; then
    echo "uv executable not found; set UV_BIN=/path/to/uv" >&2
    exit 1
fi

export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$RUNTIME_ROOT/python}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$RUNTIME_ROOT/cache/uv}
export UV_LINK_MODE=${UV_LINK_MODE:-copy}
mkdir -p "$UV_PYTHON_INSTALL_DIR" "$UV_CACHE_DIR"

"$UV_BIN" python install "$PYTHON_VERSION"
PYTHON_BIN=$(
    "$UV_BIN" python find --no-project --managed-python --system "$PYTHON_VERSION"
)
"$UV_BIN" sync \
    --project "$RUNTIME_ROOT" \
    --locked \
    --python "$PYTHON_BIN" \
    --find-links /media/public/packages

"$RUNTIME_ROOT/.venv/bin/python" - <<'PY'
import torch
import vllm

print("vLLM:", vllm.__version__)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
PY
