"""Score one SFT checkpoint with the 2026-11-25 2K-vs-32K protocol and compare.

Protocol (runs/evaluations/20261125-3060-sft-2k, reproduced here):
- Full simplified test: all 6,965 records, supervised targets only (assistant
  text + <|im_end|>). Batches from sft_torch.batch_plan(test, 4096, seed 0,
  epoch 0), sft_torch.collate, fp16 autocast on fp32 weights; per-record NLL,
  token-weighted per task. Per-task target counts must equal the official ones.
- Greedy fp32 generation with a KV cache: all 500 material-extraction items
  (limit 128, exact match after strip) and all 100 remainder variants
  (limit 192, value of the first answer sentence vs. the unique n in 1..104).

Usage (from the repo root, one GPU):
  PYTHONPATH=src python scripts/sft/eval_sft_compare.py \\
      --config configs/sft-v1-simplified-2k-muon.json \\
      --checkpoint runs/jishui-200m-sft-v1-simplified-2k-muon/checkpoints/step_00001434 \\
      --out runs/evaluations/sft-2k-muon
"""
from __future__ import annotations

import argparse
import json
import math
import re
import socket
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from jishui.config import ModelConfig
from jishui.model_torch import JishuiForCausalLM
from jishui.sft_torch import CompactSFT, autocast, batch_plan, collate, load_strict_config, sha256_file

EOD, IM_END = 2, 4
OFFICIAL_TARGETS = {"1": 5849, "3": 193910, "4": 131236, "5": 5919, "math": 23302,
                    "translation": 11931, "chat": 10208}
TASK_NAMES = {"1": "说文问答", "3": "断句", "4": "诗文举例与续写", "5": "据材料摘录",
              "math": "数学", "translation": "古今互译", "chat": "对白"}
# runs/evaluations/20261125-3060-sft-2k/summary.json
BASELINES = {
    "2k_old_sft_20261124": {
        "label": "旧 2K 基座 SFT（11-24，step 1434）",
        "ce": {"all": 1.7364060714631682, "1": 2.425364223584166, "3": 0.23682545039586278,
               "4": 3.5499823201581595, "5": 0.0207248027493866, "math": 3.169696242335978,
               "translation": 2.641299416559079, "chat": 3.177214926593356},
        "extraction_exact": 495, "remainder_correct": 1},
    "32k_release3_sft_20261124": {
        "label": "Release3 32K 基座 SFT（11-24，step 936）",
        "ce": {"all": 1.8115735262955301, "1": 2.5970398954668807, "3": 0.3889483588579532,
               "4": 3.580927216518713, "5": 0.04968320747922705, "math": 2.6189423139059906,
               "translation": 3.0600709687753254, "chat": 3.357777567866454},
        "extraction_exact": 480, "remainder_correct": 0},
}
HAN = "零一二三四五六七八九"


def put(path: Path, obj) -> None:
    tmp = path.with_suffix(".writing")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def numeral(n: int) -> str:
    if n < 10:
        return HAN[n]
    if n < 100:
        return ("" if n < 20 else HAN[n // 10]) + "十" + (HAN[n % 10] if n % 10 else "")
    return "一百" + ("零" + HAN[n - 100] if n > 100 else "")


def parse_numeral(text: str):
    """First Chinese numeral (0..999) in text, e.g. 物数二十六 -> 26, 一百三十五 -> 135."""
    m = re.search(r"[零一二三四五六七八九十百]+", text)
    if not m:
        return None
    s, total, current = m.group(), 0, 0
    for ch in s:
        if ch in HAN:
            current = HAN.index(ch)
        elif ch == "十":
            total += (current or 1) * 10
            current = 0
        elif ch == "百":
            total += (current or 1) * 100
            current = 0
    return total + current


@torch.no_grad()
def generate(model, prompt: list[int], limit: int) -> dict:
    """Greedy decode with a KV cache, identical math to the 11-25 fixed_helper."""
    cfg = model.config
    d, heads = cfg.head_dim, cfg.num_attention_heads
    device = next(model.parameters()).device
    maxlen = len(prompt) + limit
    if maxlen > cfg.max_position_embeddings or cfg.num_key_value_heads != heads:
        raise ValueError("prompt too long for generation limit")
    keys = [torch.empty((1, heads, maxlen, d), device=device) for _ in model.layers]
    values = [torch.empty((1, heads, maxlen, d), device=device) for _ in model.layers]
    inv = cfg.rope_theta ** (-torch.arange(0, d, 2, device=device, dtype=torch.float32) / d)
    angles = torch.arange(maxlen, device=device, dtype=torch.float32)[:, None] * inv
    cs, sn = angles.cos(), angles.sin()
    ids = torch.tensor([prompt], device=device)
    tokens, offset, stop = [], 0, "length"
    for step in range(limit):
        n = ids.shape[1]
        end = offset + n
        x = model.embed_tokens(ids)
        c, s = cs[offset:end], sn[offset:end]

        def shape(y):
            return y.view(1, n, heads, d).transpose(1, 2)

        def rotate(y):
            a, b = y.chunk(2, -1)
            return torch.cat((a * c - b * s, b * c + a * s), -1)
        for i, layer in enumerate(model.layers):
            att = layer.attention
            v = layer.input_norm(x)
            q = rotate(shape(att.q_proj(v)))
            keys[i][:, :, offset:end].copy_(rotate(shape(att.k_proj(v))))
            values[i][:, :, offset:end].copy_(shape(att.v_proj(v)))
            y = F.scaled_dot_product_attention(q, keys[i][:, :, :end], values[i][:, :, :end], is_causal=n > 1)
            x = x + att.o_proj(y.transpose(1, 2).reshape(1, n, -1))
            x = x + layer.mlp(layer.post_attention_norm(x))
        logits = F.linear(model.norm(x[:, -1]), model.embed_tokens.weight)[0]
        t = int(logits.argmax())
        if step in (0, 2) and len(tokens) == step:
            native = model(torch.tensor([prompt + tokens], device=device))[0, -1]
            if (native - logits).abs().max().item() >= 4e-3 or int(native.argmax()) != t:
                raise AssertionError("KV-cache decode disagrees with a full forward")
        if t in (EOD, IM_END):
            stop = "im_end" if t == IM_END else "eod"
            break
        tokens.append(t)
        offset = end
        ids = torch.tensor([[t]], device=device)
    return {"tokens": tokens, "stop": stop}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True, help="SFT checkpoint directory")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-generation", action="store_true", help="CE only (debugging)")
    args = ap.parse_args(argv)
    strict = load_strict_config(args.config)
    sft, expect = strict["sft"], strict.get("expect", {})
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)

    data = CompactSFT(Path(sft["data_dir"]))
    tokenizer_path = Path(sft["tokenizer"])
    checks = {"data_fingerprint": data.fingerprint(), "tokenizer_sha256": sha256_file(tokenizer_path),
              "test_samples": int(len(data.splits["test"]))}
    for key, actual in checks.items():
        if key in expect and expect[key] != actual:
            raise SystemExit(f"STOP: {key} mismatch: expected {expect[key]}, got {actual}")
    test = data.splits["test"]
    if len(set(test.tolist())) != 6965 or np.intersect1d(test, data.splits["train"]).size \
            or np.intersect1d(test, data.splits["val"]).size:
        raise SystemExit("STOP: test split is not the 6,965-record held-out set")

    state = json.loads((args.checkpoint / "train_state.json").read_text())
    if not state.get("sft"):
        raise SystemExit("STOP: checkpoint is not an SFT checkpoint")
    cfg = ModelConfig.from_dict(state["model"])
    with torch.device("meta"):
        model = JishuiForCausalLM(cfg)
    bundle = torch.load(args.checkpoint / "training.pt", map_location="cpu", mmap=True, weights_only=True)
    model.load_state_dict(bundle["model"], strict=True, assign=True)
    del bundle
    model.to(device=device, dtype=torch.float32).eval()
    info = {"host": socket.gethostname(), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "torch": torch.__version__, "checkpoint": str(args.checkpoint.resolve()), "step": state["step"],
            "weights_sha256": sha256_file(args.checkpoint / "training.pt"), "config_sha256": sha256_file(args.config),
            "model": cfg.to_dict(), "sft_parent": state["sft"].get("parent_checkpoint"),
            "sft_parent_step": state["sft"].get("parent_step"), **checks}
    put(args.out / "info.json", info)
    print("loaded", info["checkpoint"], info["gpu"], flush=True)

    records = []
    with torch.inference_mode():
        for bi, batch in enumerate(batch_plan(data.lengths, test, 4096, 0, 0)):
            x, y = collate(data, batch, device)
            with autocast(device):
                logits = model(x)
            flat = F.cross_entropy(logits.float().transpose(1, 2), y, ignore_index=-100, reduction="none")
            for i, loss, count in zip(batch, flat.sum(1).tolist(), (y != -100).sum(1).tolist()):
                records.append({"index": int(i), "task": data.tasks[i], "n": int(count), "nll": float(loss)})
            if (bi + 1) % 100 == 0:
                print("test batches", bi + 1, flush=True)
    counts = Counter()
    for r in records:
        counts[r["task"]] += r["n"]
    if len(records) != 6965 or dict(counts) != OFFICIAL_TARGETS:
        raise SystemExit(f"STOP: scored targets differ from the official protocol: {dict(counts)}")
    put(args.out / "test-records.json", records)
    ce = {}
    for key in ["all", *OFFICIAL_TARGETS]:
        rows = records if key == "all" else [r for r in records if r["task"] == key]
        n = sum(r["n"] for r in rows)
        ce[key] = sum(r["nll"] for r in rows) / n
    print("full test CE", json.dumps(ce), flush=True)

    objective = None
    if not args.skip_generation:
        import pyarrow.parquet as pq
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(tokenizer_path))
        meta = pq.read_table(data.root / "records.parquet", filters=[("provisional_split", "=", "test")],
                             columns=["sample_index", "task", "subtask", "messages", "loss_message_indices"]).to_pylist()
        test_set = set(test.tolist())
        rows = []
        for r in meta:
            if r["task"] != "5" and r["subtask"] != "ancient_program_variant":
                continue
            i = r["sample_index"]
            assert i in test_set
            ids, mask = data.sample(i)
            m = np.asarray(mask, dtype=np.int32)
            start = int(np.flatnonzero(np.diff(np.r_[0, m]) == 1)[-1])
            last = r["loss_message_indices"][-1]
            rows.append({"index": int(i), "task": r["task"], "prefix": np.asarray(ids[:start]).tolist(),
                         "prompt": r["messages"][last - 1]["content"], "reference": r["messages"][last]["content"]})
        if Counter(r["task"] for r in rows) != {"math": 100, "5": 500}:
            raise SystemExit("STOP: expected 500 extraction and 100 remainder items")
        out = []
        with torch.inference_mode():
            for k, limit in (("5", 128), ("math", 192)):
                for j, row in enumerate(sorted((r for r in rows if r["task"] == k), key=lambda v: len(v["prefix"]))):
                    g = generate(model, row["prefix"], limit)
                    text = tok.decode(g["tokens"], skip_special_tokens=False)
                    out.append({**{a: b for a, b in row.items() if a != "prefix"}, "text": text, "stop": g["stop"]})
                    if (j + 1) % 100 == 0:
                        print("generated", k, j + 1, flush=True)
        extraction = [r for r in out if r["task"] == "5"]
        remainder = []
        for r in (r for r in out if r["task"] == "math"):
            rem = re.findall(r"(?:三三|五五|七七)数之(适尽|剩[一二三四五六])", r["prompt"])
            residues = [0 if x == "适尽" else HAN.index(x[1]) for x in rem]
            solutions = [n for n in range(1, 105) if all(n % m == v for m, v in zip((3, 5, 7), residues))]
            if len(rem) != 3 or len(solutions) != 1 or not r["reference"].startswith("答曰：" + numeral(solutions[0]) + "。"):
                raise SystemExit(f"STOP: remainder item {r['index']} does not match the audited family")
            lead = re.split(r"[。\n！？；]", r["text"].strip(), maxsplit=1)[0].removeprefix("答曰：")
            value = parse_numeral(lead)
            remainder.append({"index": r["index"], "expected": solutions[0], "first_answer": lead,
                              "value": value, "correct": value == solutions[0]})
        put(args.out / "generations.json", out)
        put(args.out / "remainder-audit.json", remainder)
        values = Counter(r["value"] for r in remainder)
        objective = {"extraction_exact": sum(r["text"].strip() == r["reference"].strip() for r in extraction),
                     "extraction_total": len(extraction), "remainder_correct": sum(r["correct"] for r in remainder),
                     "remainder_total": len(remainder), "most_common_remainder_value": values.most_common(1)[0]}

    summary = {"info": info, "ce": ce, "ppl": {k: math.exp(v) for k, v in ce.items()},
               "objective": objective, "baselines": BASELINES}
    put(args.out / "summary.json", summary)
    lines = ["# SFT 对比：2K Muon 基座", "",
             f"checkpoint `{info['checkpoint']}`（step {info['step']}），GPU {info['gpu']}。", "",
             "| 任务 | 计分 tokens | 本次 CE | 旧 2K SFT CE | 32K SFT CE |", "|---|---:|---:|---:|---:|"]
    for key in ["all", *OFFICIAL_TARGETS]:
        name = "全测试集" if key == "all" else TASK_NAMES[key]
        n = sum(OFFICIAL_TARGETS.values()) if key == "all" else OFFICIAL_TARGETS[key]
        old, long = BASELINES["2k_old_sft_20261124"]["ce"][key], BASELINES["32k_release3_sft_20261124"]["ce"][key]
        lines.append(f"| {name} | {n:,} | {ce[key]:.4f} | {old:.4f} | {long:.4f} |")
    lines.append("")
    lines.append(f"全测试集 PPL：本次 {math.exp(ce['all']):.3f}，旧 2K {math.exp(BASELINES['2k_old_sft_20261124']['ce']['all']):.3f}，"
                 f"32K {math.exp(BASELINES['32k_release3_sft_20261124']['ce']['all']):.3f}。")
    if objective:
        lines += ["", "| 生成检查 | 本次 | 旧 2K SFT | 32K SFT |", "|---|---:|---:|---:|",
                  f"| 据材料逐字摘录 | {objective['extraction_exact']}/500 | 495/500 | 480/500 |",
                  f"| 余数变式首个作答值正确 | {objective['remainder_correct']}/100 | 1/100 | 0/100 |",
                  "", f"余数题最常见作答值：{objective['most_common_remainder_value']}。"]
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    sys.exit(main())
