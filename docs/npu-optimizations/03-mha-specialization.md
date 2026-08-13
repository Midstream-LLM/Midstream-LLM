# MHA 固定形状特化

## 背景

通用 MIL 图支持 GQA：K/V 先从 `KV_HEADS` 复制到 `HEADS`，backward 再把 dK/dV 按共享 KV head 求和。Jishui-200M 的 `HEADS=KV_HEADS=11`，即普通 MHA，复制和求和都是恒等操作。

## 修改

当编译宏 `GQA_RATIO == 1` 时：

- MIL attention 直接使用 `k_rope` 和 `v`，不生成 GQA `concat` tile 节点；
- host backward 直接把 ANE 输出写入 `dq/dk_buf/dv`，不做 `memcpy(dq_full, ...)`、tile 或 reduce；
- 不分配 GQA-only `k_tiled/v_tiled/dk_full/dv_full` scratch。

当 `GQA_RATIO != 1` 时，旧的通用路径完整保留。这样同一套源文件仍可作为未来 GQA 模型的参考实现。

## 风险

MHA 分支只在头数相等时成立。任何修改 `HEADS`、`KV_HEADS` 或 layout 的模型都必须重新编译，并通过 CPU/ANE backward parity；不要通过运行时字符串判断替代编译期宏。

