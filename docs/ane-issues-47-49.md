# ANE issues #47/#49 对 Jishui 混合训练的影响

状态：源码调查完成；第三方 ANE runtime 实测因系统 panic 停止。本文记录哪些结论可采用，以及哪些结论不能外推到 Jishui-200M。

## Issue #47：runtime weights、dW 与 Adam

有帮助，但核心发现并非全新。Jishui native trainer 已经把权重放进运行时 `IOSurface`，kernel 启动时编译一次，optimizer update 后重写 surface，不会每步重编译。

该 issue 仍提供了三类有价值的约束：

- runtime matmul 的 contraction dimension 必须为 32 的倍数；Jishui 的 `DIM=704`、`HIDDEN=1856`、`HD=64`、`SEQ=2048` 均满足。
- 输入 surface 需要按大小升序、输出按大小降序；普通 transformer projection 在 `C_out <= SEQ` 时较友好。Jishui 的最大 FFN 输出 1856 小于 2048，但 32768 词表 classifier 不满足，不能据此整体搬上 ANE。
- dW 和 Adam 可以表达成额外 ANE program，但这不等于端到端更快。当前 CPU/Accelerate dW 已与 ANE backward 异步重叠，实测 `cblas_wait=0`；把 dW 放回单 in-flight 的 ANE 会占用原本执行 forward/backward 的同一条计算通道。

不采用 issue #47 的 fp16 ANE Adam 作为正式 optimizer。它把权重和 moments 保存在 fp16，`eps` 实际需要约 `1e-4`，而 Jishui 当前使用 fp32 master/moments 与 `eps=1e-5`。小模型数百步收敛不能证明 200M LLM 长训练的更新精度，尤其在 warmup、小梯度和梯度累积下。

## Issue #49：Espresso、ANEForge 与队列

Issue 正文本身主要讨论推理生态合作。对训练真正有用的是链接项目和评论中的以下模式：

- Espresso 证明可以预绑定 `IOSurface` 为 Metal shared buffer，减少 ANE/CPU/GPU 边界的 lock、memcpy 和重复 buffer 创建。这与当前最高价值方向一致：减少 native trainer 的 fp16 surface 与 fp32 host array 往返。
- ANEForge 展示了 resident optimizer state、output-to-input buffer alias 和按层复用 program。它证明“状态常驻”可行，但公开结果主要是小型 MLP/CNN/char-LM，不足以证明 Jishui-200M 的吞吐和数值稳定性。
- 127 是 loaded/in-flight 请求上限，不是 127 路 ANE 并行。公开实测表明单 die dispatch 是 single-in-flight，`execute_sync` 仍按提交顺序串行。它可降低 host submission 开销或帮助调度，但不是 ANE data parallel/DDP。
- Espresso 的 lane-packed attention、三层 recurrent fusion 和 519 tok/s 是 decode/inference 结果，不能外推到 seq=2048 的 full-sequence pretraining。

因此目标仍是单模型的算子级异构和流水重叠，而不是在 24GB 统一内存中复制一份 ANE 模型和一份 GPU 模型。

## 本机兼容性结论

2026-08-09 在 M4、macOS 15.7.1 上对 `imperatormk/ane-train` 的
`modules/test_bwd` 做了源码/符号审计；该二进制之后列为永久禁用，不能再运行。
精确的用户态崩溃点是 `modules/test_bwd.m:163`：

- 第一次 Adam 测试忽略了 `IOSurfaceLock` 的返回值；随后
  `IOSurfaceGetBaseAddress(opt->w_surf)` 返回 `NULL`。
- 优化后的 `memcpy` 在 `NULL + 0x40` 写入，报告中的 `x0=0`、`FAR=0x40`
  和 `EXC_BAD_ACCESS` 与此完全吻合。
- 但两次重启前的统一日志还记录了相同的 ANE/AMCC 非法 DMA 事务
  （`ADDR 0x10106db3fc0`, `AID 0x1c343`, `CMD/SIZE 0x5/0x3f`，仅 TID 不同）。
  普通用户态空指针写不会制造 AMCC 硬件事务，因此更可信的因果顺序是：面向
  macOS 26 的 backward/Adam ANE 程序在 macOS 15.7.1 上先触发非法 DMA，
  导致 IOSurface 映射失效；随后被忽略的 lock/base 检查把这个失效暴露成
  `SIGSEGV`。由于跨重启日志没有保留完整的 ANE 时间线，不能声称每个硬件事件
  的先后已被完全证明，但“单纯 memcpy 导致 panic”与证据不符。
- 这解释了紧接着出现的系统 panic：`SOCD report detected: (iBoot async abort)`，
  底层为 `AMCC error`。当天 15:50 与 16:00 两次 panic 的 AMCC 地址相同。

结论是版本/私有运行时不兼容触发了硬件侧非法事务，而不是 Jishui 自研 trainer
的 Metal-only 路径。后续只借鉴该项目的源码约束；不执行其 binary，也不通过反复
试错探测私有 ANE 行为。

该外部项目标注 macOS 26+，不能在本机直接执行。后续只借鉴其 MIL/IOSurface 设计；不再运行其 binary，也不通过反复试错探测私有 ANE 行为。

## 后续顺序

1. 用 Metal-only probe 验证预绑定 IOSurface、fp16 输入输出和同步语义，不调用 ANE。
2. 在源码中统计并减少 `io_read_dyn` / `io_write_dyn` 的转换与拷贝，保留现有 CPU/Metal shadow reference。
3. 只在已验证的 Jishui kernel 集上接入零拷贝边界；任何 ANE 执行都需与现有 gate 使用完全相同的 MIL 形状和 I/O 约束。
4. dW/Adam 上 ANE 仅保留为研究候选；除非端到端 A/B 显示收益且长期 parity 通过，否则继续使用异步 CPU dW 与 fp32 Adam。
