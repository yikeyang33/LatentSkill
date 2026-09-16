# Cross-server migration

The source code and reproducible Python environment are stored in Git. Large
experiment assets are copied separately with resumable `rsync`:

- `checkpoints/`
- `data/`
- `alfworld_data/`
- `models/`
- `wiki_index/`

The local `.venv/` is intentionally not transferred. The destination rebuilds
it from `uv.lock`.

## One-command migration

The destination must have Git, uv, rsync, SSH access, and a compatible NVIDIA
driver. From the source server, run:

```bash
bash scripts/migrate_to_server.sh \
  user@destination-host \
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

At the time this migration setup was added, the source contained approximately
128 GiB of checkpoints and 2.3 GiB of ALFWorld data. Qwen3-8B model weights
were not yet present in `models/Qwen3-8B/`; download or synchronize them before
expecting `verify_migration.sh` to pass.
