# LatentSkill 跨开发机同步说明

本文说明另一台能够 SSH 连接本机的开发机，如何同步本项目的代码、模型、
checkpoint、ALFWorld 数据和 uv 运行环境。

当前源端信息：

| 内容 | 源端位置 |
|---|---|
| 项目代码 | `/media/yangyike/research/RSI/repos/LatentSkill` |
| Git 仓库 | `https://github.com/yikeyang33/LatentSkill.git` |
| Qwen3-8B | `/media/public/models/huggingface/Qwen/Qwen3-8B` |
| LatentSkill checkpoint | 项目下的 `checkpoints/` |
| ALFWorld 数据 | 项目下的 `alfworld_data/alfworld/` |
| Python 版本和依赖锁 | `.python-version`、`pyproject.toml`、`uv.lock` |

这里使用的是 **`Qwen/Qwen3-8B` 后训练模型**，不是单独的
`Qwen/Qwen3-8B-Base`。不要从 Hugging Face 在线路径直接启动评测；先使用
上表中的本地目录，或者把该目录同步到目标机后再加载。

## 1. 在目标机配置到源端的 SSH

以下命令都在目标开发机执行。推荐先在目标机的 `~/.ssh/config` 中配置别名：

```sshconfig
Host changliu
  HostName 117.186.102.101
  User root
  Port 32073
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
```

验证连接：

```bash
ssh changliu 'hostname; test -d /media/yangyike/research/RSI/repos/LatentSkill'
```

不要把密码或私钥写入本仓库。首次配置密钥登录时，可在目标机执行：

```bash
ssh-copy-id -p 32073 root@117.186.102.101
```

## 2. 同步代码：使用 Git，不要 rsync 工作树

首次获取：

```bash
mkdir -p /media/"$USER"/research/RSI/repos
cd /media/"$USER"/research/RSI/repos
git clone --recurse-submodules \
  https://github.com/yikeyang33/LatentSkill.git
cd LatentSkill
```

后续更新：

```bash
cd /media/"$USER"/research/RSI/repos/LatentSkill
git status --short
git pull --ff-only origin main
git submodule update --init --recursive
```

源端尚未 commit/push 的改动不会出现在目标机。应先在源端提交并推送，再在
目标机 `git pull --ff-only`。不要用 rsync 覆盖 `.git/`、源码或正在运行的工作树。

## 3. 模型：优先复用公共目录，否则用 rsync 拉取

### 情况 A：两台机器挂载同一个 `/media/public`

不需要复制模型。确认目标机能看到完整文件：

```bash
MODEL_PATH=/media/public/models/huggingface/Qwen/Qwen3-8B
test -s "$MODEL_PATH/model.safetensors.index.json"
find "$MODEL_PATH" -maxdepth 1 -name 'model-*.safetensors' -type f | sort
```

运行时显式指定：

```bash
export MODEL_PATH=/media/public/models/huggingface/Qwen/Qwen3-8B
```

### 情况 B：目标机没有共享公共盘

在目标机选择 media 盘上的本地目录，并从源端断点续传：

```bash
MODEL_PATH=/media/"$USER"/models/huggingface/Qwen/Qwen3-8B
mkdir -p "$MODEL_PATH"

rsync -avh \
  --partial \
  --partial-dir=.rsync-partial \
  --info=progress2,stats2 \
  --protect-args \
  changliu:/media/public/models/huggingface/Qwen/Qwen3-8B/ \
  "$MODEL_PATH/"
```

中断后重新执行同一命令即可继续。命令没有 `--delete`，不会删除目标端已有文件。

同步后在两端比较清单和关键文件哈希：

```bash
ssh changliu \
  'cd /media/public/models/huggingface/Qwen/Qwen3-8B && sha256sum config.json model.safetensors.index.json model-00001-of-00005.safetensors model-00005-of-00005.safetensors'

cd "$MODEL_PATH"
sha256sum config.json model.safetensors.index.json \
  model-00001-of-00005.safetensors model-00005-of-00005.safetensors
```

两边四行哈希必须逐项一致。

## 4. 同步 checkpoint 和 ALFWorld 数据

在目标项目根目录执行：

```bash
cd /media/"$USER"/research/RSI/repos/LatentSkill

mkdir -p checkpoints alfworld_data/alfworld

rsync -avh \
  --partial \
  --partial-dir=.rsync-partial \
  --info=progress2,stats2 \
  --protect-args \
  changliu:/media/yangyike/research/RSI/repos/LatentSkill/checkpoints/ \
  checkpoints/

rsync -avh \
  --partial \
  --partial-dir=.rsync-partial \
  --info=progress2,stats2 \
  --protect-args \
  changliu:/media/yangyike/research/RSI/repos/LatentSkill/alfworld_data/alfworld/ \
  alfworld_data/alfworld/
```

如需 SearchQA，再同步：

```bash
rsync -avh --partial --info=progress2,stats2 --protect-args \
  changliu:/media/yangyike/research/RSI/repos/LatentSkill/data/ \
  data/

rsync -avh --partial --info=progress2,stats2 --protect-args \
  changliu:/media/yangyike/research/RSI/repos/LatentSkill/wiki_index/ \
  wiki_index/
```

不要同步正在写入的 `evals/**/results*`、日志或训练 checkpoint。等待任务完成，
或者先在源端生成只读快照后再同步。

## 5. uv 环境：同步锁文件，通常不要复制虚拟环境

### 独立磁盘或不同项目路径

不要复制 `.venv/` 或 `.shared-runtime/venv/`。虚拟环境可能包含绝对路径、机器
相关解释器和二进制扩展。代码通过 Git 获取后，在目标机根据 `uv.lock` 重建：

```bash
cd /media/"$USER"/research/RSI/repos/LatentSkill

# 目标机没有 uv 时，可在同为 Linux x86_64 的机器间只复制 uv 可执行文件。
if ! command -v uv >/dev/null 2>&1; then
  mkdir -p "$HOME/.local/bin"
  scp changliu:/home/yangyike/.local/bin/uv "$HOME/.local/bin/uv"
  chmod +x "$HOME/.local/bin/uv"
  export PATH="$HOME/.local/bin:$PATH"
fi

uv --version

# 可选：把下载缓存也放到目标机的 media 盘。
export UV_CACHE_DIR=/media/"$USER"/.cache/uv

bash scripts/setup_shared_env.sh
```

该脚本会把 uv 管理的 Python 3.10 和项目环境放在：

```text
<PROJECT_ROOT>/.shared-runtime/python/
<PROJECT_ROOT>/.shared-runtime/venv/
```

依赖严格来自已提交的 `uv.lock`，PyTorch 为项目锁定的 CUDA 12.4 构建。后续统一
通过以下入口运行 uv：

```bash
bash scripts/uv_shared.sh run python --version
bash scripts/uv_shared.sh run python -c \
  'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

若目标机不是 Linux x86_64，请不要使用上面的 `scp` 方法，改用 uv 官方安装方式；
无论哪种情况，都不要复制整个虚拟环境。

### 两台机器共享同一项目目录

如果两台机器看到的是**同一绝对项目路径**，并且 Linux、CPU 架构和 glibc 兼容，
可以共同读取同一个 `.shared-runtime/`，无需重复创建。注意：

1. 同一时间只能有一个进程执行 `setup_shared_env.sh` 或 `uv sync`。
2. 普通训练和评测进程可以并发读取该环境。
3. 宿主机 NVIDIA 驱动仍须兼容锁定的 PyTorch CUDA 12.4 runtime。
4. 若任一机器出现动态库或 ABI 错误，应在各自的 media 路径中单独重建环境。

## 6. 验证同步结果

先检查关键资源：

```bash
cd /media/"$USER"/research/RSI/repos/LatentSkill

test -s "$MODEL_PATH/model.safetensors.index.json"
test -s checkpoints/latentskill_sft_qwen3_8b/checkpoint-epoch-10/metanetwork.pth
test -s checkpoints/latentskill_sft_qwen3_8b/checkpoint-epoch-10/metalora.pth
test -s alfworld_data/alfworld/logic/alfred.pddl
test -x .shared-runtime/venv/bin/python
```

执行真实 CUDA 运算：

```bash
CUDA_VISIBLE_DEVICES=0 .shared-runtime/venv/bin/python - <<'PY'
import torch

print("torch:", torch.__version__)
print("runtime CUDA:", torch.version.cuda)
assert torch.cuda.is_available()
print("GPU:", torch.cuda.get_device_name(0))
x = torch.randn(256, 256, device="cuda")
_ = x @ x
torch.cuda.synchronize()
print("CUDA smoke test passed")
PY
```

最后做一次轻量导入检查：

```bash
MODEL_PATH="$MODEL_PATH" \
PYTHONPATH="$PWD" \
.shared-runtime/venv/bin/python -m evals.alfworld.evaluate --help
```

## 7. 在远端提交长任务

所有远端长任务必须在命名 tmux 会话中运行，并把输出写入日志：

```bash
cd /media/"$USER"/research/RSI/repos/LatentSkill
PROJECT_ROOT="$PWD"
export MODEL_PATH=/实际的本地模型路径/Qwen3-8B

tmux new-session -d -s latentskill-sft-seen \
  "cd '$PROJECT_ROOT' && \
   MODEL_PATH='$MODEL_PATH' bash scripts/eval_alfworld_sft.sh seen all 0"

tmux list-sessions
tmux attach -t latentskill-sft-seen
```

默认日志位于 `evals/alfworld/logs/`，结果位于 `evals/alfworld/results/`。

## 8. 最短操作清单

目标机没有共享盘时，按以下顺序操作：

1. 配置并验证 `ssh changliu`。
2. 从 GitHub clone/pull 代码。
3. 从源端 rsync `Qwen3-8B`、`checkpoints/` 和 `alfworld_data/`。
4. 设置本地 `MODEL_PATH`。
5. 运行 `bash scripts/setup_shared_env.sh`，不要复制 `.venv/`。
6. 做文件检查和 CUDA smoke test。
7. 在 tmux 中提交正式任务。
