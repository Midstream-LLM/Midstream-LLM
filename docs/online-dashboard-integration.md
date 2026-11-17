# 在线训练看板接入指南

看板：<https://www.fgprc.org.cn/training>

推送地址：`https://www.fgprc.org.cn/api/training/ingest`

读取地址：<https://www.fgprc.org.cn/api/training>

每次实验使用一个固定且唯一的 `runId`；同一实验的所有进度、验证和结束消息都用同一个 ID。页面默认展示开始时间最新的实验，可在顶部切换历史实验。训练重启续跑同一实验时沿用 ID；全新实验换新 ID。若无法确定真实开始时间，看板会用该实验首次收到的记录时间作为近似值。

## 准备

1. 从看板管理员通过安全渠道取得写入令牌，并在训练机器的进程环境中设置：

   ```bash
   export TRAINING_DASHBOARD_INGEST_URL='https://www.fgprc.org.cn/api/training/ingest'
   export TRAINING_INGEST_TOKEN='<从管理员获取>'
   ```

   不要把令牌提交到代码、写入日志或分享给无关人员。当前看板使用**共用写入令牌**，持有者可以更新所有实验，不提供按实验隔离的权限；只向可信同事发放。数据将公开展示，只推送可以公开的内容。
2. 为本次运行确定一个不会与其他人冲突的 ID，例如 `teamname-model-20261029-01`，记下实际开始时间，使用含时区的 ISO 8601 格式，如 `2026-10-29T10:00:00+08:00`。不要在断点恢复时改变该时间。
3. 多卡训练只让主进程（通常 global rank 0）推送，避免重复记录；建议每 30–60 秒发送一次，验证结果出来时以及结束时再发一次。持续超过约 3 分钟未更新的 `live` 实验会显示为等待更新。

管理员可从服务器 `serverbeijing` 的 `/home/sidi762/apps/jishui-dashboard/runtime.env` 中读取 `TRAINING_INGEST_TOKEN`，并通过安全渠道单独提供给同事；该文件不应作为附件或文档公开分享。

## Python 最小接入

只依赖标准库。将下面函数放进训练脚本，在主进程调用；示例中的指标应替换为训练中的实际值。每次调用都带 `runId` 和原始 `startedAt`：

```python
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone

RUN_ID = "teamname-model-20261029-01"  # 每个新实验唯一；续跑沿用
STARTED_AT = "2026-10-29T10:00:00+08:00"  # 实际启动时间；续跑不变


def push_dashboard(step, train_loss=None, val_loss=None, lr=None,
                   tokens=None, target_tokens=None, throughput=None, status="live"):
    payload = {
        "runId": RUN_ID,
        "runName": "我的模型训练",
        "startedAt": STARTED_AT,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "progress": {"step": step, "tokens": tokens, "targetTokens": target_tokens},
        "metrics": {"trainLoss": train_loss, "valLoss": val_loss,
                    "learningRate": lr, "throughput": throughput},
        "config": {"model": "我的模型", "dataset": "公开语料名称",
                   "batchSize": 128, "sequenceLength": 2048},
    }
    request = urllib.request.Request(
        os.environ["TRAINING_DASHBOARD_INGEST_URL"],
        data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + os.environ["TRAINING_INGEST_TOKEN"]},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.load(response)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        # 推送失败不应中断训练；可记录错误并在下次周期重试。
        print(f"看板推送失败: {type(exc).__name__}", flush=True)
        return None

# 在训练循环中由 rank 0 定期调用：
# push_dashboard(step, train_loss=loss, lr=learning_rate, tokens=tokens_seen,
#                target_tokens=target_tokens, throughput=tokens_per_second)
# 验证完成时也可带 val_loss；训练正常结束时发送 status="finished"。
# 明确暂停或失败时分别发送 status="paused" 或 status="error"。
```

配置字段和目标值请按实际情况修改，不知道的字段可以省略或设为 `None`。建议将网络推送放到训练主进程的后台线程或队列中，以免网络超时拖慢训练；上例同步调用最多等待 5 秒。只发送有限数值，不要发送 NaN/Infinity。异常退出可能来不及发送 `error`，此时页面会在超时后显示等待更新；恢复后可继续用同一 ID 更新状态。

## 字段与行为

| 字段 | 用途 |
|---|---|
| `runId` | 唯一且稳定的实验标识，所有多实验更新必须携带 |
| `runName` | 选择器中显示的名称 |
| `title` | 页面大标题；不传则保留已有标题 |
| `subtitle` | 大标题下的说明；新实验默认“汉语原生语言模型从零预训练”，不传则保留，传空字符串 `""` 可隐藏 |
| `startedAt` | 实际开始时间，决定实验排序；缺失时以首次记录时间近似 |
| `timestamp` | 本次数据时间；省略时服务器使用接收时间 |
| `status` | `live`、`finished`、`paused`、`error`；省略时沿用该实验上次状态 |
| `progress.step` | 当前训练步数；曲线横轴 |
| `progress.tokens`、`targetTokens` | 已训练及目标 token 数；用于计算进度 |
| `progress.epoch`、`totalEpochs`、`percent` | 可选进度；`percent` 可传 0–1 或 0–100 |
| `metrics.trainLoss`、`valLoss`、`learningRate`、`throughput` | 当前指标；吞吐量单位为 token/秒 |
| `config.model`、`dataset`、`batchSize`、`sequenceLength` | 公开配置 |

各字段均属于同一个实验，不传的数值通常沿用该实验上次值。每次发送带 `step` 的指标会加入该实验的曲线，同一步的新值会替换旧值；曲线只保留服务器配置的最近点数。保持同一实验消息按时间顺序发送，避免迟到的旧消息覆盖较新的状态与指标。不要为了试接入创建虚假生产实验，直接使用真实实验 ID 的真实数据。

## 推送实验成本（可选）

在同一实验的推送 JSON 中加入完整的 `cost` 对象。例如：

```json
{
  "runId": "teamname-model-20261029-01",
  "startedAt": "2026-10-29T10:00:00+08:00",
  "status": "live",
  "cost": {
    "gpu": "8x Tesla V100-SXM2-32G",
    "gpuCount": 8,
    "watts": 2400,
    "currency": "CNY",
    "schemes": [
      {"key":"rent", "label":"云租用口径", "basis":"V100 32G 按 1.6 元/卡时 × 8 卡", "ratePerHour":12.8},
      {"key":"industrial", "label":"工业电价口径", "basis":"2400W × 0.85 元/kWh", "ratePerHour":2.04},
      {"key":"residential", "label":"居民电价口径", "basis":"2400W × 0.4883 元/kWh", "ratePerHour":1.17}
    ]
  }
}
```

- `gpu`、`watts`、`currency`、`schemes` 必填；目前币种支持 `CNY`。`watts` 和单价必须是非负有限数字。
- `gpuCount` 是 GPU 数量（正整数），用于计算累计卡时。建议显式填写；未填写时，`gpu` 以 `8x ` 或 `8× ` 这类数字前缀开头会推断数量，否则默认 1。
- `schemes` 是非空数组；每项需要 `key`、`label`、`basis`、`ratePerHour`。`key` 使用 `rent`、`industrial`、`residential`，不能重复；`ratePerHour` 是**整组机器每小时**的金额，不能再乘卡数。可只发送需要的口径。
- `cost.startedAt` 可选，设定成本计时起点；省略或设为 `null` 时使用实验的 `startedAt`，都没有时使用首次记录时间。`cost.priorHours` 可选，表示本轮之前的运行小时数，默认为 0。
- 首次发送后，后续不带 `cost` 的更新会保留配置；再次发送时须提供完整对象，并整体替换。`cost: null` 和不完整或非法配置返回 HTTP 400，整次更新不写入。读取 `GET /api/training?runId=<ID>` 的 `cost` 字段可验证结果。
- 同一实验的 API 成本配置优先于管理员既有配置；未通过 API 设置成本的实验继续使用管理员配置。所有展示数据公开；共享写入令牌的持有者均可更新成本。

## 检查与排错

推送成功会返回 `{"ok":true, ...}`。打开看板，选择该实验，确认名称、状态、步数和指标；公开读取接口可用 `GET /api/training?runId=<URL编码后的ID>` 检查具体实验。HTTP 401 表示令牌不匹配；503 表示服务器未配置令牌；400 可表示 JSON 格式有误或多实验时缺少 `runId`；404 表示读取的 ID 不存在。网络故障时保留原训练过程，并在后续周期重试。若需要补录过去的曲线，可先与看板管理员确认。
