# QKV backward 融合：实验路径

当前 qBwd 与 kvBwd 是两个 ANE evaluation。对 MHA，可选的 `qkvBwd` MIL 图用一次 request 同时计算 `dq @ Wq`、`dk @ Wk` 和 `dv @ Wv`。第二版在 MIL 内执行 fp16 `dx_k + dx_v`，只输出 `dx_q` 和 `dx_kv` 两段；相比第一版少一个 `DIM x SEQ` 输出平面，同时保持既有 split `kvBwd` 的舍入边界。

融合图输入布局为每个 channel 依次放置 `dq[SEQ]`、`dk[SEQ]`、`dv[SEQ]`、`Wq[DIM]`、`Wk[DIM]`、`Wv[DIM]`。输出按 channel-major 连续放置两份 `DIM x SEQ`：

```text
dx_q  = fp16(dq @ Wq)
dx_kv = fp16(fp16(dk @ Wk) + fp16(dv @ Wv))
```

host 分别除去共同的 `qkv_bwd_scale`，再用 fp32 完成 `dx_q + dx_kv`。这不是把 dW 融合进图，也不是把 optimizer 搬到 ANE；dW 仍由异步 Accelerate 任务累计，Adam 仍由 fp32 master state 执行。

## 失败策略

`--npu-fuse-qkv-bwd-shadow` 逐层重新运行 split qBwd/kvBwd。split 任一 request 失败或输出非有限值时，本次 gate fail-closed：跳过后续 embedding backward 和 optimizer update，统一 cleanup 后以非零状态退出。若 fused 有限但差异超过 `2e-2`，则复制有限的 split 结果继续诊断，并在当前进程内永久禁用融合。普通 fused 模式的 evaluation 或 finite 检查失败也会自动回退 split。

split surface/request 仍常驻，供 fallback 和 shadow oracle 使用。因此当前实现减少一次运行时 request，但不减少启动时的 split kernel 编译与常驻资源。

## 实测结论（2026-08-11）

在 Jishui-200M、seq 2048、step-7200 fp16 权重和固定 sampler 记录 0 上：

- 带 Metal 全 shadow 的 30 层 `dx_q`/`dx_kv` 对照全部 `max_dx_diff=0`；
- 完整导出梯度对逐层 native mixed-precision VJP 为 relative L2 `3.44296%`、cosine `0.999408`，与该记录的 split 指标一致；
- 五 microstep 配对测速丢弃每段 step 0 后，split A `3.7618 s`、fused `3.7731 s`、split B `3.7156 s`；按两侧 split 均值，fused 慢约 `0.9%`；
- backward ANE bucket 只减少约 `2.4 ms`，backward I/O bucket 反而增加约 `27 ms`，少一次 request 尚未抵消 fused 图调度和大 surface 边界成本。

结论：精度成立，性能不成立。`--npu-fuse-qkv-bwd` 继续作为默认关闭的研究开关，不进入正式训练配置。
