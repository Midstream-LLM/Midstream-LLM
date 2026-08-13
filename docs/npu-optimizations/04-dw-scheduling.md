# dW 调度与生命周期 fence

## 目标与结论

MLX 用惰性执行把可独立的算子排到设备队列中。native 路径中的 dW GEMM 在 CPU/Accelerate 异步执行，原实现曾在每层 forward 开头等待全局 `dw_grp`。

审计了激活生命周期后确认，这一处等待是冗余的：前一个 microstep 在进入 embedding backward 前已经等待了全部 dW，因此当前 microstep 开始时上一轮异步任务都已完成。现已删除该 forward 等待，避免在每层 ANE evaluation 前重复阻塞。

其余等待仍然是所有权 fence，不能删：

- 每个 microstep 的 reverse backward 完成后，embedding backward 前等待 `dw_grp`；
- Adam 更新、checkpoint、最终清理前再次等待 `dw_grp`；
- `dw_slots` 继续限制异步捕获缓冲的数量。

这不是“按层独立 group”实现，也不声称已经形成跨层 dW/ANE 的真正流水线。若要进一步重叠，必须重构每层 activation capture 生命周期，并通过 runtime race/parity gate。
