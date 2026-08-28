# ConPath Pro 6000 迁移与训练说明

这个目录可以直接复制到远程训练机。不要复制本机的 `.venv/`、`.uv-cache/`、`__pycache__/` 或已有 checkpoint；这些目录与本机 Python、操作系统和 CUDA 绑定，通常不能跨机器复用。

## 1. 先做只读检查

进入 `research/conpath`（当前 transfer 目录仍可保留旧路径）后运行：

```powershell
python scripts/check_environment.py
python scripts/check_environment.py --json > environment.json
```

Linux/macOS 使用同样的 Python 命令即可。脚本只读取 Python、`nvidia-smi` 和 `torch.cuda` 状态，不会安装软件、不改变驱动，也不会申请大块显存。

必须确认：

```text
torch.cuda.is_available() = True
torch.version.cuda       非 None
GPU 名称、显存和 compute capability 与目标机器一致
```

“Pro 6000”可能指不同代际：RTX PRO 6000 Blackwell、RTX 6000 Ada，或者旧的 Quadro P6000。不要仅凭名称复制 CUDA 命令；以 `nvidia-smi` 输出和 PyTorch 官方安装矩阵为准。Blackwell 机器通常需要支持 Blackwell 的 CUDA 12.8 或更新 wheel；旧 Quadro P6000 则可能需要 legacy PyTorch/CUDA 组合。

参考：

- [NVIDIA GPU compute capability 表](https://developer.nvidia.com/cuda/gpus)
- [PyTorch 官方安装页](https://pytorch.org/get-started/locally/)
- [PyTorch previous versions 与 CUDA wheel](https://pytorch.org/get-started/previous-versions/)
- [NVIDIA Blackwell compatibility guide](https://docs.nvidia.com/cuda/archive/13.0.0/blackwell-compatibility-guide/index.html)

## 2. 建立远程环境

项目要求 Python 3.10–3.12。建议在远程机新建环境，不要上传本机 `.venv`：

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
```

Linux：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
```

根据 `check_environment.py` 与官方安装页选择 CUDA wheel。Blackwell 的常见示例是 CUDA 12.8：

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy
```

如果目标驱动明确支持 CUDA 13.0，也可以按官方页面改用 `cu130`。`torchvision` 不是当前合成原型的必需依赖，但为后续 RGB/LiDAR backbone 预留。若目标机是 Ada、旧数据中心卡或 Quadro P6000，使用官方矩阵中与驱动匹配的 wheel，不要强行使用上面的 Blackwell 示例。

安装后再次运行：

```bash
python scripts/check_environment.py
```

## 3. 先跑最小 GPU smoke

当前 `train_synthetic.py` 支持显式设备参数，但尚未实现 AMP、DDP 或自动 resume。先用小步数确认 CUDA 路径：

PowerShell：

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe scripts\train_synthetic.py `
  --config configs\synthetic.json `
  --device cuda:0 `
  --steps 1 `
  --validation-size 2 `
  --validation-samples 4 `
  --checkpoint checkpoints\gpu_smoke.pt
```

Linux：

```bash
export PYTHONPATH=src
.venv/bin/python scripts/train_synthetic.py \
  --config configs/synthetic.json \
  --device cuda:0 \
  --steps 1 \
  --validation-size 2 \
  --validation-samples 4 \
  --checkpoint checkpoints/gpu_smoke.pt
```

看到 `paper_result: false` 是正常的：这一步只验证 forward、backward、optimizer 和 checkpoint，不代表论文结果。

## 4. 推荐的第一轮配置

当前默认合成配置是 `H=W=24, B=8, K=8, Q=2, R=3`，最大传播步数为 `24*24=576`。RTX PRO 6000 Blackwell 的大显存足以容纳这个原型，但速度主要受连通层影响：每次 query 都展开 `[B,K,Q,H,W]`，并且 Python 循环最多执行 `H*W` 次。

因此建议按下面顺序增加负载：

```text
GPU smoke：B=4, K=4, 1 step
合成调试：B=8, K=8, 120 steps
更大地图前：先固定 B=2, K=4, Q<=8，再逐项 profile
```

不要在第一轮同时提高地图分辨率、query 数、样本数和传播迭代数。当前原型是单 GPU 代码，多 GPU 训练和 query/sample chunking 仍是后续工程任务。

## 5. 当前已知限制

- 没有 AMP：连通事件和离线标签仍建议保持 FP32；不要自行把整个 forward 包进 fp16 后当作论文结果。
- 没有 DDP、多卡同步、自动恢复和 best-checkpoint 选择。
- 连通层是研究原型，复杂度随 `B*K*Q*H*W*max_steps` 增长；大规模公开数据训练前必须做可扩展实现。
- 当前输入仍是 `[B,3,H,W]` 栅格 BEV 观测，不是 RGB/LiDAR loader；真实数据接入通过 `PathRelNet.forward_features()`。

## 6. 传输后的文件清单

应传输：

```text
README.md
ROADMAP_ZH.md
ALGORITHM.md
GPU_SETUP.md
pyproject.toml
configs/
data/README.md
results/README.md
scripts/
src/
tests/
```

可不传输：

```text
.venv/
.uv-cache/
checkpoints/*.pt
__pycache__/
data/raw/
data/processed/
results/*（保留 results/README.md）
```
