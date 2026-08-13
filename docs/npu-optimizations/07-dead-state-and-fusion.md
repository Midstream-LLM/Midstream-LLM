# Dead state、常量与融合的关系

本轮迁移 MLX 优化时，最重要的判断不是“能不能把更多算子塞进 ANE”，而是先确认每个中间值的消费者和生命周期。一个值若没有 backward 或下一层消费者，就应该从图输出和 host allocation 同时删除；一个值若会被异步 dW 读取，就必须保留到 dispatch group 完成。

## 已删除的状态

- `sdpaFwd` 不再输出已经在 host 保存的 `xnorm`。
- `LayerActs.ffn_out`、`o_out` 不参与 backward；Wo 输出改用已有的瞬时缓冲。
- Jishui 的 MHA 不需要 GQA tile/reduce scratch。
- `W2t_buf` 只用于一次无效转置，已删除；FFN forward 直接使用原始 `W2` staging。
- 恒等 vocab 映射下，classifier 直接读取 tied `embed`，dEmbed 直接累加到 `gembed`。
- classifier logits 不再复制成第二份 dlogits；稳定 softmax、target 减一和 `1/SEQ` 缩放都在原 `[SEQ,V]` buffer 内完成。
- forward residual 不再经过独立的 `x_cur`：第 `L` 层把输出写进第 `L+1` 层的 `layer_in`。最后层使用独立 `final_residual`，保证 final norm 的输入仍可供 backward 读取。
- `dx_kv` 是 `dffn` 的生命周期复用别名。它只在 dffn 的最后一个消费者完成后才被覆盖，且异步 dW 捕获的是独立副本。

这些修改改变的是 buffer ownership 和数据流，不改变模型方程。它们的共同验收标准是：静态 shape 检查、固定记录的 loss/gradient/Adam probes、有限状态，以及中断后的资源清理。

### Alias 通过条件

每个别名都必须满足三个条件：旧值没有后续消费者；所有异步读者已经复制或等待；释放路径只拥有一份 free。静态审计目前覆盖正常训练和 microstep fence；早期错误路径仍可能直接 `return 1`，这属于既有 cleanup 风险，不应被误认为 alias 已经自动安全。

## 仍然保留的边界

ANE 输入输出依旧是 fp16 IOSurface，host master weights、gradients 和 Adam moments 依旧是 fp32。统一内存不等于可以无锁地让 MLX 和 ANE 同时写同一份数组；必须有明确的 producer/consumer fence。当前 hybrid 路径已由 `src/jishui/train_hybrid.py` 接通：MLX 拥有 optimizer/checkpoint，native worker 按 control generation 从共享 fp16 权重 mmap 读取，并把平均后的 fp16 梯度写回独立 mmap。权重和梯度仍不是同一块可并发写 tensor；control page 的 READY/RUNNING/GRAD_READY 状态就是所有权 fence。该路径已通过 bounded gate，但仍是研究原型，不是 overnight 生产入口。

## QKV 融合为何单独放在实验区

删除 dead state 是局部且可证明的；QKV 融合会改变 ANE request 的输入/输出 surface 和调度，属于运行时行为变化。第二版在 MIL 内保留 split `kvBwd` 的 fp16 `dx_k + dx_v` 舍入边界，并只输出 `dx_q`/`dx_kv`。30 层 shadow 已逐位一致，但配对测速仍慢约 `0.9%`，所以它继续默认关闭。该结论只适用于 Jishui MHA 的 dX request，不说明 GQA、dW 或 Adam 可以直接合并。split oracle request 失败或输出非有限时，shadow 会 fail-closed 结束 gate；只有 finite 的 split 结果才允许作为 fallback。
