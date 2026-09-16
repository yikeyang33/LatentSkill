#!/usr/bin/env bash
# Run uv with both its managed Python and project environment on shared storage.

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ -z "${UV_BIN:-}" ]; then
    if command -v uv >/dev/null 2>&1; then
        UV_BIN=$(command -v uv)
    elif [ -x "${HOME}/.local/bin/uv" ]; then
        UV_BIN="${HOME}/.local/bin/uv"
    else
        echo "uv was not found in PATH or ${HOME}/.local/bin" >&2
        exit 1
    fi
fi

export UV_PYTHON_INSTALL_DIR="${PROJECT_ROOT}/.shared-runtime/python"
export UV_PROJECT_ENVIRONMENT="${PROJECT_ROOT}/.shared-runtime/venv"
export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/latentskill-uv-cache}
export UV_LINK_MODE=${UV_LINK_MODE:-copy}

exec "$UV_BIN" "$@"
