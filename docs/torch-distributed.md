# PyTorch 分布式训练后端

状态：后端探测、`torchrun` rank/device 规划和 collective smoke 已落地；PyTorch 模型与正式 pretraining loop 尚未接入。本页不涉及 MLX 训练。

## 1. “NPU + GPU 分布式”的边界

这里必须区分两类完全不同的 NPU：

- Apple ANE：M4 芯片上的 Neural Engine。它不是 PyTorch device，也没有 c10 collective backend。仓库里的 `native/ane/` 能把部分算子放到 ANE，并把 LayerNorm/SiLU 放到 Metal GPU；这是单进程、算子级异构，不是 DDP。
- 昇腾 NPU：由 `torch_npu` 暴露为 `npu:N`，多卡 collective 用 HCCL。它与 Apple ANE 没有关系。

同一个 DDP replica group 不能混用 CUDA GPU 和昇腾 NPU：CUDA tensor 用 NCCL，昇腾 tensor 用 HCCL，Apple ANE 则没有 PyTorch collective backend。统一内存也不会把 ANE 和 Metal 变成一个可 all-reduce 的设备类型。因此当前支持的拓扑是：

| 设备 | PyTorch device | 多进程 backend | 当前用途 |
|---|---|---|---|
| V100/A800/3060 | `cuda:N` | NCCL | 正式训练目标 |
| 昇腾 | `npu:N` | HCCL | 仅运行时适配骨架，项目没有已确认硬件 |
| Apple GPU | `mps` | 无 | 单进程 PyTorch；本项目本机路径已由 MLX 收口 |
| Apple ANE | 无 | 无 | `native/ane/` 算子级 ANE+Metal |
| CPU | `cpu` | Gloo | 分布式入口 smoke，不用于正式训练 |

代码会拒绝 `cuda,ascend-npu`、多进程 MPS、Apple ANE DDP，以及含糊的 `npu` 名称；必须明确写 `apple-ane` 或 `ascend-npu`。

## 2. 只读探测

本机没有安装 `torch`/`torch_npu` 也可以运行探测，因为导入是 lazy 的：

```bash
PYTHONPATH=src python -m jishui.train_torch --probe
```

只生成某个 `torchrun` rank 的规划、不导入 PyTorch：

```bash
RANK=5 LOCAL_RANK=1 WORLD_SIZE=8 \
PYTHONPATH=src python -m jishui.train_torch \
  --plan-only --accelerator cuda
```

输出应包含 `device=cuda:1` 和 `process_group_backend=nccl`。环境变量必须同时包含 `RANK`、`LOCAL_RANK`、`WORLD_SIZE`，避免残缺环境默默退回 rank 0。

## 3. 机器上的 runtime gate

安装与驱动匹配的 PyTorch 后，可以先检查单进程设备：

```bash
PYTHONPATH=src python -m jishui.train_torch \
  --check-runtime --accelerator cuda
```

然后在目标机器做不训练、不写 checkpoint 的 collective smoke：

```bash
PYTHONPATH=src torchrun --standalone --nproc_per_node=8 \
  -m jishui.train_torch --collective-smoke --accelerator cuda
```

昇腾机器把 `cuda` 换成 `ascend-npu`，并安装与 PyTorch/CANN 精确匹配的 `torch_npu`。CUDA 和 `torch_npu` 的 wheel 与驱动版本强绑定，因此不放进通用 `requirements.txt`，应在各自训练节点按厂商矩阵安装。

## 4. 接入正式 trainer 时的约束

- 8×V100 使用 homogeneous CUDA DDP、NCCL、fp16 autocast + GradScaler；不要开 bf16。
- 每个 rank 用 `LOCAL_RANK` 绑定一张卡，sampler seed 需要纳入 rank，DDP accumulation 的非最后 microstep 使用 `no_sync()`。
- checkpoint 只由 rank 0 原子写入，同时保存 optimizer、scaler、sampler/RNG 和 global step；恢复后再由所有 rank barrier。
- 正式 trainer 复用 `jishui.distributed.initialize_runtime()`，不要自行猜测 `npu` 是 Apple ANE 还是昇腾。
