# RoPE 反向表预计算

## 修改

旧实现每层、每个 microstep、每个 head pair 重复执行 `powf`、`cosf`、`sinf`。现在按固定 `(SEQ, HD)` 在 trainer 初始化时创建 fp32 cos/sin 表，backward 只做查表和乘加；结束时释放表。

RoPE 正向仍由固定 MIL blob 执行，反向表的索引布局与 CPU split-half/adjacent-pair 分支一致。这个优化不改变浮点精度（表仍为 fp32），只移除重复的 transcendental 计算。

## 风险

如果改变 `rope_theta`、`ROPE_TRADITIONAL`、`SEQ` 或 `HD`，必须重新初始化表。当前 native 模型宏固定这些值；未来动态配置不能复用旧表。

