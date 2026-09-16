#!/usr/bin/env bash
# Synchronize large, non-Git experiment assets to another machine or directory.
#
# Usage:
#   bash scripts/sync_assets.sh user@host:/absolute/path/LatentSkill
#   DRY_RUN=1 bash scripts/sync_assets.sh user@host:/absolute/path/LatentSkill

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: bash scripts/sync_assets.sh <destination-project-root>" >&2
    exit 2
fi

DESTINATION=${1%/}
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

ASSET_DIRS=(
    checkpoints
    data
    alfworld_data
    models
    wiki_index
)

SOURCES=()
for asset_dir in "${ASSET_DIRS[@]}"; do
    source_path="${PROJECT_ROOT}/${asset_dir}"
    if [ -e "$source_path" ]; then
        SOURCES+=("$source_path")
    fi
done

if [ "${#SOURCES[@]}" -eq 0 ]; then
    echo "No asset directories found under ${PROJECT_ROOT}" >&2
    exit 1
fi

RSYNC_ARGS=(
    --archive
    --human-readable
    --copy-links
    --partial
    --partial-dir=.rsync-partial
    --info=progress2,stats2
    --itemize-changes
    --protect-args
    --mkpath
)

if [ "${DRY_RUN:-0}" = "1" ]; then
    RSYNC_ARGS+=(--dry-run)
fi

echo "Project root : ${PROJECT_ROOT}"
echo "Destination  : ${DESTINATION}/"
echo "Asset dirs   : ${ASSET_DIRS[*]}"
echo "Delete mode  : disabled"

rsync "${RSYNC_ARGS[@]}" "${SOURCES[@]}" "${DESTINATION}/"
