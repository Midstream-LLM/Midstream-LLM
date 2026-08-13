# NPU 优化总览

本轮目标是把 MLX 已经验证过的思路，逐项映射到 Jishui 的 ANE 路径：固定 shape、减少中间张量、减少 host/ANE 边界搬运、把可复用常量常驻，以及让 CPU dW 工作与 ANE 调度尽量重叠。

## 当前状态

| 优化 | 状态 | 默认行为 | 主要收益 |
| --- | --- | --- | --- |
| 删除 SDPA 未使用输出 | 已实现 | 默认启用 | 每次 SDPA 输出少一个 `DIM x SEQ` fp16 平面 |
| 激活内存清理 | 已实现 | 默认启用 | 删除 `ffn_out`/`o_out` 和一次 `xnorm` memcpy |
| MHA 固定 shape 特化 | 已实现 | `GQA_RATIO=1` 自动启用 | 跳过 K/V tile、reduce、额外 full-dQ 缓存 |
| 删除 forward 冗余 dW barrier | 已实现 | 删除每层 forward 开头的全局等待 | 上一个 microstep 末尾已经等待全部 dW，避免重复 CPU fence |
| RoPE 反向表 | 已实现 | 默认启用 | 每个 shape 只算一次三角函数 |
| 删除 W2 转置 scratch | 已实现 | 默认启用 | 不再常驻 `W2t_buf`，也不做无效转置写入 |
| Adam 与转置 staging 融合 | 已实现 | 默认启用 | 更新矩阵时直接填充 ANE 转置副本，省掉 6 次完整 transpose pass/update |
| tied embedding 直接复用 | 已实现 | 恒等词表映射自动启用 | 消除完整 vocab 的 `cembed` 复制和更新拷贝 |
| classifier logits 原地转 dlogits | 已实现 | 默认启用 | 复用 `[SEQ,CV]` fp32 buffer；Jishui-200M/2048 省约 256 MiB |
| IOSurface 段批量映射 | 已实现 | 默认启用 | SDPA/QKV 同一 surface 多段读写只做一次 lock/unlock |
| forward residual 直接链式写入 | 已实现 | 默认启用 | 下一层直接接收上一层输出；删除额外 residual buffer 和逐层 memcpy |
| backward scratch 生命周期复用 | 已实现 | 默认启用 | `dx_kv` 复用 `dffn`；Jishui-200M/2048 再省约 5.5 MiB |
| Metal SiLU 临时缓冲清理 | 已实现 | `--metal-silu` 时启用 | SiLU backward 直接写目标缓冲 |
| qkv backward 单图融合 | 已通过精度 gate，性能未过 | `--npu-fuse-qkv-bwd` 默认关闭 | 两输出图减少一次 ANE request，但当前配对测速约慢 0.9% |
| ANE scratch pool | 暂缓 | 不启用 | 需要先测 allocator/copy 是否为实际瓶颈 |

## 与 MLX 的对应关系

MLX 的 `mx.compile` 会在固定输入 shape 下规划惰性计算图、复用临时内存并融合一部分算子。ANE 私有 API 没有公开的等价 Python 编译器，因此这里采用显式方式达到相同目标：MIL 图在启动时按固定 shape 编译一次；host 端只保留 backward 必需的激活；固定常量以 MIL blob 形式提供；MHA/GQA 分支在编译期由模型宏裁剪。

这不是模型复制式分布式训练，也没有把 optimizer 或梯度自动搬到 ANE。当前职责仍是：ANE 做矩阵密集的 forward/backward-dx，Metal 做已验证的 LayerNorm/SiLU/I/O，CPU/Accelerate 做 classifier、softmax、dW 和 fp32 AdamW。

`--npu-fuse-qkv-bwd` 只在 `GQA_RATIO == 1` 时尝试。融合图在 MIL 内完成 fp16 `dx_k + dx_v`，只输出 `dx_q` 和 `dx_kv` 两段；这与既有 split `kvBwd` 的舍入边界一致，同时少写一个 `DIM x SEQ` 输出平面。`--npu-fuse-qkv-bwd-shadow` 会额外运行 split 图作为 oracle。当前实现仍保留 split surface/request，因此它减少的是运行时 request 数，不承诺减少启动时常驻内存。shadow 中 split request 失败或 split 输出出现 NaN/Inf 时会 fail-closed，结束本次 gate，不会把坏结果复制回训练状态。

## 显式内存规划

这些别名是基于消费者和生命周期证明的 host 端规划，不是 ANE runtime 的隐式 alias：

- `logits` 在交叉熵之后已经没有原始 logits 消费者，`cross_entropy_loss_inplace()` 直接把每行改写成平均后的 dlogits。classifier backward 和异步 dEmbed 完成前，下一 microstep 不会覆盖它。
- forward 中 `acts[L+1].layer_in` 直接作为第 `L` 层 FFN/residual 输出；每层自己的 `acts[L].layer_in` 留给 reverse pass。最后一层写独立的 `final_residual`，避免 final LayerNorm 输入输出别名。
- reverse pass 中 `dffn` 写入 ANE 前已被消费并复制给异步 dW；随后 `dx_kv` 才复用这块 scratch。cleanup 只释放一次 `dffn`。

任何异步 dW 读者都由 `dw_grp` fence 保护；没有这条 fence，以上规划不成立。

## Runtime gate（2026-08-11）

在正式 MLX 训练由用户暂停后，本仓库 binary 已完成 production shape gate：

- step-7200 checkpoint、固定 sampler 记录 0、`--metal-shadow --metal-silu --npu-fuse-qkv-bwd-shadow` 下，30 层 `dx_q`/`dx_kv` 对 split oracle 的 `max_dx_diff` 全部为 `0`；
- 逐层手写 native mixed-precision VJP 对完整 200,224,640 个导出梯度得到 relative L2 `3.44296%`、cosine `0.999408`，与该记录既有 split 结果一致；
- 五 microstep、丢弃各段 step 0 的 `split A -> fused -> split B` 配对为 `3.7618 -> 3.7731 -> 3.7156 s/step`。两侧 split 均值 `3.7387 s/step`，fused 慢约 `0.9%`。

因此两输出 fusion 的精度 gate 已过，但性能 gate 未过，继续默认关闭。它相对旧三输出版本约 `1.8%` 的负收益有所改善，但不足以成为生产默认值。

## 验证边界

MLX 正式训练运行时只应执行静态检查：

```bash
make -C native/ane/training_dynamic \
  MODEL=jishui_200m_2048 \
  BUILD_DIR=/tmp/jishui-npu-static-check -B
```

这不会加载 ANE 或提交 Metal command buffer。后续真实 parity/吞吐复测仍要在 MLX 停止或明确空闲后执行。本机此前出现过第三方 ANE binary 触发系统 panic，因此只运行本仓库构建的 ANE 可执行文件。短 gate 必须带 `--no-checkpoint`，并先过 shadow parity，再比较不带 shadow 的吞吐。
