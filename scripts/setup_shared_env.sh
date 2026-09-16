#!/usr/bin/env bash
# Create/update the shared Python 3.10 runtime and locked uv environment.

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

bash scripts/uv_shared.sh python install 3.10
bash scripts/uv_shared.sh sync --locked --python 3.10

echo "Shared environment ready at ${PROJECT_ROOT}/.shared-runtime/venv"
