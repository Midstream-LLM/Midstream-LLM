# 模型配置说明（对齐 Qwen3-1.7B 尺度）

状态：设计稿，未落地。实现时以本文档数值为准，数值变动需同步更新本文档与 AGENTS.md。

## 1. 参考：Qwen3-1.7B 配置

本项目结构参考 Qwen3-1.7B：标准 MHA + RoPE + GQA + SwiGLU + RMSNorm + 共享 embedding（tie_word_embeddings）。

| 参数 | Qwen3-1.7B |
|---|---|
| hidden_size | 2048 |
| num_hidden_layers | 28 |
| num_attention_heads | 16 |
| num_key_value_heads | 8 |
| head_dim | 128 |
| intermediate_size | 8192（≈4×d_model） |
| hidden_activation | silu（SwiGLU） |
| rms_norm_eps | 1e-6 |
| rope_theta | 1_000_000 |
| max_position_embeddings | 32768 |
| tie_word_embeddings | true |
| vocab_size | 151936（本项不照抄，用自训词表） |
| attention_dropout / hidden_dropout | 0 |

## 2. 三阶段配置

词表固定用 `ccc-bbpe-32k`（32768，见 docs/tokenizer.md）。参数量按「embedding + 每层(attn + mlp)」手工核算：

每层参数（GQA，head_dim=128，kv_heads=h/2，intermediate=4d）：
- attn：q= d·d，k = v = d·(d/2)，o = d·d → 3.5d²
- mlp：gate+up = 2·d·4d = 8d²，down = 4d·d = 4d² → 12d²
- 每层合计 15.5d²

### Stage 0：Jishui-200M-Base（本机 1×3060 试验，模型结构已定）

```yaml
name: Jishui-200M-Base
vocab_size: 32768             # ccc-bbpe-32k
hidden_size: 704
num_hidden_layers: 30
attention_type: MHA
num_attention_heads: 11
num_key_value_heads: 11
head_dim: 64
ffn_type: SwiGLU
intermediate_size: 1856
norm_type: LayerNorm
norm_position: pre_norm
norm_eps: 1.0e-5
norm_bias: true
position_embedding: RoPE
rope_theta: 10000
max_position_embeddings: 2048
tie_word_embeddings: true
attention_bias: false
mlp_bias: false
dropout: 0.0
qk_norm: false
final_norm: true
```

核算（无 bias）：
- embedding：32768 × 704 ≈ 23.07M
- 每层：attn 4×704² = 1.98M + mlp 3×704×1856 = 3.92M + LayerNorm 2×2×704 ≈ 2.8k → ≈ 5.91M
- 30 层 ≈ 177.2M
- **合计 ≈ 200.2M**

用途：管线冒烟（tokenize/shard/训练脚本全链路验证）+ 超参/配比试验。3060 12G 上 fp16 可跑 seq 2048。

注意：本配置为试验取向，与 Stage 1/2 的 Qwen3 家族有偏差（LayerNorm+bias、head_dim 64、rope_theta 1e4、seq 2048）——Stage 0 用于快速试验，**不承诺与 600M/1.5B 结构完全同族**；Stage 1/2 以 Qwen3 结构为准。

### Stage 1：~600M（首选）

| 参数 | 值 |
|---|---|
| hidden_size | 1536 |
| num_hidden_layers | 16 |
| num_attention_heads | 12 |
| num_key_value_heads | 6 |
| head_dim | 128 |
| intermediate_size | 6144 |
| rms_norm_eps | 1e-6 |
| rope_theta | 1_000_000 |
| max_position_embeddings | 4096 |
| tie_word_embeddings | true |
| vocab_size | 32768 |

核算：
- embedding：32768 × 1536 ≈ 50.3M
- 每层：15.5 × 1536² ≈ 36.6M（细算 attn 7.08M + mlp 28.31M = 35.39M）
- 16 层 ≈ 566M
- **合计 ≈ 617M**

### Stage 2：~1.5B

| 参数 | 值 |
|---|---|
| hidden_size | 2048 |
| num_hidden_layers | 24（Qwen3-1.7B 少 4 层） |
| num_attention_heads | 16 |
| num_key_value_heads | 8 |
| head_dim | 128 |
| intermediate_size | 8192 |
| rms_norm_eps | 1e-6 |
| rope_theta | 1_000_000 |
| max_position_embeddings | 4096 |
| tie_word_embeddings | true |
| vocab_size | 32768 |

核算：
- embedding：32768 × 2048 ≈ 67.1M
- 每层：attn 12.58M + mlp 50.33M = 62.91M
- 24 层 ≈ 1.51B
- **合计 ≈ 1.58B**

备选：直接照搬 Qwen3-1.7B 的 28 层 → ≈1.83B（超出 1.5B 目标，暂不采用）。

## 3. 决策记录与开放问题

- **seq_len = 4096**：Qwen3 支持 32k，但古汉语单文档长（殆知阁整书 >10k tokens），pre-training 阶段 4096 已够，且 V100 显存友好。若后续要长文本能力，用 rope_theta 固定 + 延长训练 seq 的方式升级。
- **vocab 用 32k**：24k/32k 见 tokenizer 文档；选 32k 降低字节碎片、embedding 成本增量（8M）可忽略。
- **tie embeddings**：与 Qwen3 一致，1.5B 以下 + 共享词表收益明显。
- **rope_theta=1e6**：照 Qwen3；若 4096 seq 上 loss 异常可退回 1e4 实验。
- **待验证**：600M 用 d=1536 + 12 heads（head_dim=128）而非 16 heads（head_dim=96，非对齐）；如实现时发现 kernel 对齐问题，可改为 16 heads + head_dim 96 并更新本文档。

## 4. 训练相关（硬件约束）

- **V100 无 bf16 tensor core**：用 fp16 + `torch.cuda.amp` + `GradScaler`；bf16 只能在 A800/3060 上开。
- 6×V100 32G：600M/1.5B 用 **DDP + grad checkpoint** 即可，无需 FSDP。1.5B + Adam fp32 ≈ 21GB/卡（含激活），32G 可行。
- 启动：`torchrun --nproc_per_node=N`。
