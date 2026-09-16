#!/usr/bin/env bash
# Recreate the Git checkout on a target server, transfer large assets, and
# rebuild the locked uv environment.
#
# Usage:
#   bash scripts/migrate_to_server.sh user@host /absolute/path/LatentSkill
#
# Optional environment variables:
#   REPO_URL=https://github.com/yikeyang33/LatentSkill.git
#   SKIP_ENV=1        Skip `uv sync --locked` on the target.
#   SKIP_VERIFY=1     Skip the target-side verification step.
#   DRY_RUN=1         Only show which assets rsync would transfer.

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: bash scripts/migrate_to_server.sh <user@host> <absolute-target-path>" >&2
    exit 2
fi

REMOTE=$1
TARGET_PATH=$2
REPO_URL=${REPO_URL:-https://github.com/yikeyang33/LatentSkill.git}
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if [[ "$TARGET_PATH" != /* ]]; then
    echo "Target path must be absolute: ${TARGET_PATH}" >&2
    exit 2
fi

printf -v quoted_target '%q' "$TARGET_PATH"
printf -v quoted_repo '%q' "$REPO_URL"

echo "[1/4] Preparing Git checkout on ${REMOTE}:${TARGET_PATH}"
ssh "$REMOTE" \
    "if [ -d ${quoted_target}/.git ]; then git -C ${quoted_target} pull --ff-only; else mkdir -p \$(dirname ${quoted_target}) && git clone ${quoted_repo} ${quoted_target}; fi"

echo "[2/4] Synchronizing large assets with resumable rsync"
DRY_RUN=${DRY_RUN:-0} \
    bash "${PROJECT_ROOT}/scripts/sync_assets.sh" "${REMOTE}:${TARGET_PATH}"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "Dry run complete; target environment was not changed."
    exit 0
fi

if [ "${SKIP_ENV:-0}" != "1" ]; then
    echo "[3/4] Rebuilding locked uv environment on target"
    ssh "$REMOTE" \
        "cd ${quoted_target} && UV_LINK_MODE=copy uv sync --locked"
else
    echo "[3/4] Skipping target uv environment rebuild"
fi

if [ "${SKIP_VERIFY:-0}" != "1" ]; then
    echo "[4/4] Verifying target assets and CUDA runtime"
    ssh "$REMOTE" \
        "cd ${quoted_target} && bash scripts/verify_migration.sh"
else
    echo "[4/4] Skipping target verification"
fi
