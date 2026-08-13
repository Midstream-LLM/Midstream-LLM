# NPU 优化教程

## 先理解四种内存

模型权重和激活是普通 fp32 host 数组；ANE 输入/输出是 IOSurface 上的 fp16；Metal 可以把页面对齐的 host/IOSurface 地址包装成 shared buffer；CPU/Accelerate 负责尚未迁移的操作。统一内存意味着物理内存可能共享，但不代表不同 API 自动共享同一个 tensor view，仍需明确 layout、锁和生命周期。

## MIL 图是什么

MIL 是 ANE 编译器读取的静态计算图文本。`func main<ios18>` 的输入输出 shape 在编译时固定，`matmul/softmax/concat` 等节点会被编译成 ANE 可执行程序。运行时只更新预留 IOSurface 中的激活和权重，不重复编译整张图。

## 为什么固定 shape 有用

固定 `SEQ=2048`、`DIM=704`、`HEADS=11` 让编译器预先规划 tile、临时内存和 DMA。代价是换序列长度必须换 model header 并重新编译；这与 MLX 对固定 batch/sequence 的 `mx.compile` 类似。

## 一次 forward/backward 的边界

1. host/Metal 生成 LayerNorm 激活并写入 IOSurface。
2. ANE 执行投影和 attention，输出 fp16 cache。
3. CPU 或 Metal 解包成 backward 所需 fp32 激活。
4. ANE 执行 dX，CPU/Accelerate 异步累计 dW。
5. 所有 dW 完成后执行 fp32 AdamW，再把更新后的权重重写回 IOSurface。

任何异步任务仍可能读取激活，所以 group wait 是内存所有权 fence，不是“每个节点都必须同步”。

## 内存复用怎么判断

`logits` 在 loss 后已经不再需要原始值，因此交叉熵直接原地生成 dlogits，省掉第二个完整词表矩阵。forward 的每层输出写到下一层 `layer_in`，而每层自己的输入保留到反向；最后一层单独写 `final_residual`。反向中 `dx_kv` 复用 `dffn`，因为异步 dW 已经捕获了 dffn 副本。

SDPA backward 的 Q/K/V/da 以及 fused QKV 的输入段共享各自的 IOSurface
映射。host 端先校验所有段的范围，再一次性 lock、转换 fp16、unlock。fused
QKV 在 MIL 内先按 fp16 合并 `dx_k + dx_v`，只回传 `dx_q` 和 `dx_kv` 两段，
避免第三个输出平面，同时保持 split `kvBwd` 的舍入边界。

Adam 更新后的 Q/K/V/O、W1/W3 会被 ANE 以转置布局读取。现在 Adam 循环直接
把更新后的元素写入对应转置 staging buffer，随后只做 fp16 staging，不再另跑
六遍完整矩阵转置；fp32 master 权重和 optimizer state 的更新顺序保持不变。

这些不是“统一内存自动安全”的例子。覆盖前必须确认没有消费者，异步 dW 必须经过 `dispatch_group_wait`，并且 cleanup 只能 free owner 指针一次。

## 本轮实际启用的优化

生产 Jishui-200M 默认启用 dead-state 清理、MHA 固定 shape 分支、RoPE backward 表、tied embedding 复用、`W2t` scratch 删除，以及 Adam/转置 staging 合并。它们不需要额外命令行开关。`--metal-silu` 时还会省掉 CPU-only 的 SiLU 临时数组。

QKV backward 是唯一默认关闭的实验开关。先用：

```bash
./train --scratch --index /private/tmp/jishui_ane_train.index \
  --data-dir /Volumes/PS2000/LLM/Jishui/dataset/processed \
  --optimizer-steps 3 --accum 1 --no-checkpoint \
  --metal-norm --metal-silu --metal-io \
  --npu-fuse-qkv-bwd-shadow \
  --parity-report /private/tmp/ane.qkv-shadow.parity.jsonl
```

融合只对普通 MHA 生效。shadow 会逐层重新运行 split qBwd/kvBwd；差异超阈值时使用 finite split reference 并关闭融合。split request 失败或 split 输出非有限时，gate 直接 fail-closed，不会继续训练或复制坏结果，并以非零状态退出。通过后再去掉 `--npu-fuse-qkv-bwd-shadow` 测吞吐。

2026-08-11 的两输出 runtime gate 中，30 层 shadow 都是逐位一致，严格 checker 为 relative L2 `3.44296%`、cosine `0.999408`。但 `split A -> fused -> split B` 的稳态均值为 `3.7618 -> 3.7731 -> 3.7156 s/step`；fused 比两侧 split 均值慢约 `0.9%`，所以开关仍默认关闭。精度通过不等于性能通过。

## 编译检查

MLX 正式训练运行时只执行：

```bash
make -C native/ane/training_dynamic \
  MODEL=jishui_200m_2048 BUILD_DIR=/tmp/jishui-npu-static-check -B
git diff --check
```

MLX 空闲后才做 runtime gate。当前固定记录 gate 已完成，但 QKV fusion 因无性能收益仍不进入正式配置。短 gate 使用 `--no-checkpoint`，避免产生约 2.4GB 的 native checkpoint。

先比较 finite loss 和 parity，再记录 tok/s；不要把一次冷机结果外推成长期吞吐。第三方 ANE binary 在本机触发过系统级 panic，不纳入教程实验。
