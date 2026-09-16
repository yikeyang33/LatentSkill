# Cross-server migration

> For the current machine-specific pull workflow, including the public
> Qwen3-8B path and the media-backed uv environment, use
> [SYNC_README.md](SYNC_README.md). This file documents the older generic
> migration helpers.

The source code and reproducible Python environment are stored in Git. Large
experiment assets are copied separately with resumable `rsync`:

- `checkpoints/`
- `data/`
- `alfworld_data/`
- `models/`
- `wiki_index/`

The local `.venv/` is intentionally not transferred. The destination rebuilds
it from `uv.lock`.

## Shared-storage clusters

When multiple GPU servers mount this repository at the same absolute `/media`
path, no asset transfer is necessary. Create one shared runtime instead:

```bash
bash scripts/setup_shared_env.sh
```

This installs both the managed Python interpreter and the environment below
`.shared-runtime/`, so the virtual environment does not point at a machine-local
Python under `$HOME`. Run commands through the shared environment with:

```bash
bash scripts/uv_shared.sh run python -m evals.alfworld.evaluate --help
```

The environment may be read concurrently from multiple machines, but only one
process should run `setup_shared_env.sh` or `uv sync` at a time.

## One-command migration

The destination must have Git, uv, rsync, SSH access, and a compatible NVIDIA
driver. From the source server, run:

```bash
bash scripts/migrate_to_server.sh \
  user@destination-host \
  /absolute/path/LatentSkill
```

For a non-default SSH port or a specific private key:

```bash
SSH_PORT=32073 \
SSH_IDENTITY_FILE="$HOME/.ssh/id_ed25519" \
bash scripts/migrate_to_server.sh \
  root@117.186.102.101 \
  /absolute/path/LatentSkill
```

The script performs four operations:

1. Clone or fast-forward `https://github.com/yikeyang33/LatentSkill.git`.
2. Copy large assets with resumable rsync without deleting destination files.
3. Run `uv sync --locked` on the destination.
4. Verify checkpoints, ALFWorld files, Qwen3 weights, and a real CUDA matrix multiplication.

Preview the asset transfer without modifying the destination:

```bash
DRY_RUN=1 bash scripts/migrate_to_server.sh \
  user@destination-host \
  /absolute/path/LatentSkill
```

If the destination is a login node without GPU access, migrate and build the
environment first, then run verification inside an allocated GPU job:

```bash
SKIP_VERIFY=1 bash scripts/migrate_to_server.sh \
  user@destination-host \
  /absolute/path/LatentSkill

ssh user@destination-host \
  'cd /absolute/path/LatentSkill && bash scripts/verify_migration.sh'
```

Interrupted transfers can be resumed by running the same command again. The
script does not enable rsync deletion, so files that exist only on the target
are preserved.

## Current source inventory

The authoritative Qwen3-8B copy is now the shared local model at
`/media/public/models/huggingface/Qwen/Qwen3-8B`. The project-local
`models/Qwen3-8B/` directory is not the authoritative model source. Follow
`SYNC_README.md` to reuse or rsync the public model explicitly.
