# 激活内存与复制清理

## 修改内容

`LayerActs.ffn_out` 和 `LayerActs.o_out` 从未在 backward 使用，却各自为每层分配 `DIM x SEQ` fp32 缓冲；现已删除它们的字段、分配和释放。Wo 投影输出改写到已有的 `x_final` 临时缓冲，立即用于 residual，最终 norm 前再覆盖。

原先 LayerNorm 先写入全局 `xnorm_buf`，再 `memcpy` 到每层的 `ac->xnorm`。现在 CPU/Metal LayerNorm 直接写入 `ac->xnorm`，并把同一指针传给 SDPA 输入和后续 dW 捕获。

forward 的 residual stream 现在使用 `acts[L+1].layer_in` 直接接收第 `L` 层输出，删除了单独的 `x_cur` allocation 和每层 `memcpy(ac->layer_in, x_cur, ...)`。最后一层写到独立 `final_residual`，因此 final LayerNorm 仍然是非别名输入/输出。

反向的 `dx_kv` 与 `dffn` 生命周期不重叠：W2 backward 输入已经提交、异步 dW 也使用 `capt_dffn` 副本后，`dx_kv` 可以直接指向 `dffn`，删除另一份 `DIM x SEQ` scratch。

对 `D=704, SEQ=2048, L=30`，删除约 330 MiB 的无用 per-layer allocation、约 5.5 MiB 的 residual scratch、约 5.5 MiB 的 `dx_kv` scratch，并省掉每个 microstep 约 165 MiB 的 xnorm 复制写流量。实际收益取决于内存带宽和热状态，不能仅由字节数推断 tok/s。

## 生命周期约束

异步 dW 任务会读取 `ac->xnorm`、`ac->x2norm`、`ac->attn_out` 和 `ac->silu_out`。因此不能在这些任务完成前覆盖激活。当前 microstep 在 embedding backward 前等待总 `dw_grp`，optimizer、checkpoint 和最终清理前也会等待；`dw_slots` 限制尚未完成的捕获缓冲数量。forward 开头的重复全局 wait 已删除。

classifier 的 `[SEQ,CV]` logits 也采用原地规划：交叉熵把它改写为 dlogits，随后 classifier backward 和 dEmbed 都只读这一个 buffer。下一步开始前的 `dispatch_group_wait(dw_grp)` 是覆盖它的必要条件。

## 初学者理解

激活是“这一次输入经过某层后留下的中间结果”，反向传播还要用它。删除一个没有消费者的副本安全；把两个本来相同的缓冲合为一个，只有在确认旧值没有异步读者时才安全。
