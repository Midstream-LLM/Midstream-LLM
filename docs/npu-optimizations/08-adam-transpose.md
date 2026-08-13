# Adam 与 ANE 权重转置合并

## 动机

ANE 的动态 matmul 输入把每个投影权重按转置后的 channel-first 布局暂存；CPU/Accelerate 的 dW 和 fp32 master 权重仍使用普通 row-major 布局。旧路径在每次 optimizer update 后，先更新 master 矩阵，再对 `Wq/Wk/Wv/Wo/W1/W3` 各做一次完整转置，最后把转置结果转成 fp16 写入 IOSurface。

## 修改

`adam_update_with_transpose()` 在同一个元素循环里完成 AdamW 更新，并把更新后的值直接写到 ANE 转置缓存。`W2` 没有独立的转置缓存，因此继续走普通 `adam_update()`。这只改变 host 端写入顺序，不改变 Adam moments、weight decay、fp32 master 或 ANE 的 fp16 staging 边界。

## 生命周期与验证

optimizer 更新前已有 `dw_grp` fence，异步 dW 不会再读取正在更新的 master 权重。转置缓存在 restage 前已完整写完，因此每层 surface 仍然接收同一组更新后的值。

仅静态构建即可验证布局和链接：

```bash
make -C native/ane/training_dynamic \
  MODEL=jishui_200m_2048 \
  BUILD_DIR=/tmp/jishui-npu-fused-adam-transpose -B
```

真实数值 parity 和吞吐比较仍需等 MLX 空闲后按 baseline gate 执行。
