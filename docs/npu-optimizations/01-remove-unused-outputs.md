# 删除未使用的 SDPA 输出

## 设计

旧的 `sdpaFwd` MIL 图输出 `(attn_out, Q_rope, K_rope, V, xnorm)`。训练循环在 ANE evaluation 前已经把 `xnorm` 写入每层 `LayerActs.xnorm`，backward 的 dW 也从这个 host buffer 读取，因此最后一个输出只是重复数据。

现在图只输出 `(attn_out, Q_rope, K_rope, V)`，同时缩小 `compile_kern_mil_w` 的 output IOSurface。前四段的 channel offset 没有变化，所以 CPU 和 Metal 解包代码不需要改布局。

## 为什么对应 MLX

计算图里一个不被消费者读取的值仍可能占用 output surface、边界带宽和生命周期。MLX 的图优化通常会消除这类 dead value；在 ANE 路径中必须手工从 MIL 输出 tuple 和 host surface 大小两处一起删除。

## 风险与验收

风险是 output shape 声明与实际 consumer 不一致。验收顺序：

1. `git diff --check`；
2. 生产模型 clang 编译/链接；
3. ANE 空闲后运行一条固定记录，确认 MIL compile、loss、gradient probe 和 CPU shadow parity；
4. 再做多步吞吐比较。

本轮静态编译已通过；runtime parity 尚未在 MLX 占用期间执行。

