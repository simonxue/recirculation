#!/usr/bin/env python3
"""
严肃验证：recirculation 复现实验（多文档 × 多位置窗口 × 多配置）
================================================================

【与之前实验的差异（针对三个弱点）】
  1. 多位置窗口：每个文档不只取开头，而是取 3 个随机位置窗口
     （论文图 9 表明收益随位置/滞后增长，只测开头会低估效果）
  2. 多文档：默认 6 个文档 × 3 位置 = 18 个窗口，统计更稳
  3. 逐位置诊断：输出每 128-token 桶的 log-likelihood 增益，
     验证"收益随 lag 增长"这一论文核心机制（而非整体平均掩盖噪声）

【用法】
  python3 eval_strict.py --n_docs 6 --positions 3 \
      --alphas 0.04 0.07 0.10 --layer_pairs 11-4 10-4 12-5 \
      --out results_strict.json

【输出】
  - 每个配置的加权平均 ppl 与相对变化
  - 每文档 per-doc 明细
  - 最优配置的逐位置增益桶（对照论文图 9a）
  - 结果 JSON
"""
import argparse
import json
import math
import random
import time

import torch
import torch.nn.functional as F

from recirculation import (load_model, eval_baseline_ppl, recirc_logits,
                           perplexity_from_logits, RecircParams,
                           get_env_fingerprint)


def get_pg19_docs(n_docs: int, min_len: int = 4000):
    """
    从 emozilla/pg19（parquet 镜像）streaming 取 n_docs 个足够长的文档。

    注意：新版 datasets（5.x）不再支持官方 deepmind/pg19 的旧式脚本
    数据集，所以用 parquet 镜像仓库（可直接流式读取，无需全量下载）。
    """
    import datasets
    ds = datasets.load_dataset("emozilla/pg19", split="test", streaming=True)
    docs = []
    for ex in ds:
        text = ex["text"]
        if len(text) >= min_len:
            docs.append(text)
            if len(docs) >= n_docs:
                break
    return docs


def sample_windows(tokenizer, docs, positions: int, window: int, seed: int,
                   device):
    """
    从每个文档取 positions 个随机位置窗口，返回 token id 列表（已在目标设备）。

    策略：文档开头必取一个（论文实验的一致性），其余在文档中后部
    随机取。窗口起点用随机种子控制，保证可复现。

    为什么窗口要够"深"？
      论文图 9 显示 recirculation 的收益随滞后（lag）增长——窗口开头
      的 token 没有历史状态可传播，收益最弱。只有深入文档内部才能
      测出真实效果。
    """
    rng = random.Random(seed)
    windows = []
    for text in docs:
        # 先 token 化整个文档开头部分（最多 8 倍窗口长度，控制内存）
        ids = tokenizer(
            text, truncation=True, max_length=window * 8,
            return_tensors="pt").input_ids[0]
        max_start = ids.numel() - window
        if max_start < 1:
            continue
        # 位置 0（开头）必取
        starts = [0]
        # 其余位置在 (0, max_start] 之间随机
        for _ in range(positions - 1):
            starts.append(rng.randint(1, max_start))
        for s in starts:
            windows.append(ids[s:s + window].to(device))
    return windows


def weighted_ppl(ppls_and_counts):
    """按 token 数加权的平均困惑度：exp(Σ NLL·n / Σ n)。"""
    nll_sum = sum(math.log(p) * n for p, n in ppls_and_counts)
    cnt_sum = sum(n for _, n in ppls_and_counts)
    return math.exp(nll_sum / cnt_sum)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_docs", type=int, default=6, help="用几个 PG-19 文档")
    ap.add_argument("--positions", type=int, default=3, help="每文档取几个位置窗口")
    ap.add_argument("--window", type=int, default=1024, help="窗口长度（token）")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.04, 0.07, 0.10])
    ap.add_argument("--layer_pairs", type=str, nargs="+",
                    default=["11-4", "10-4", "12-5"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results_strict.json")
    args = ap.parse_args()

    bundle = load_model("google/gemma-3-1b-pt")

    print(f"\n=== 获取 {args.n_docs} 个 PG-19 文档，每文档取 {args.positions} 个位置窗口 ===")
    docs = get_pg19_docs(args.n_docs)
    windows = sample_windows(bundle.tokenizer, docs, args.positions,
                             args.window, args.seed, bundle.device)
    print(f"共 {len(windows)} 个窗口，每窗口 {args.window} token")

    # ---------- baseline（并行 prefill，快）----------
    t0 = time.time()
    base_items = []   # (ppl, token数)
    for i, ids in enumerate(windows):
        p = eval_baseline_ppl(bundle, ids)
        base_items.append((p, ids.numel() - 1))
    ppl_b = weighted_ppl(base_items)
    print(f"\nbaseline ppl = {ppl_b:.4f}  ({len(windows)} 窗口, {time.time()-t0:.0f}s)")

    # ---------- recirculation 扫描 ----------
    results = []
    best = None   # 记录最优配置（供诊断用）
    for alpha in args.alphas:
        for pair in args.layer_pairs:
            src, dst = map(int, pair.split("-"))
            t0 = time.time()
            r_items = []
            per_win = []
            for ids in windows:
                logits = recirc_logits(
                    bundle, ids,
                    RecircParams(source=src, dest=dst, alpha=alpha, ramp=10))
                p = perplexity_from_logits(logits[:-1], ids[1:])
                r_items.append((p, ids.numel() - 1))
                per_win.append(p)
            ppl_r = weighted_ppl(r_items)
            pct = (ppl_r / ppl_b - 1) * 100
            # 逐窗口对比 baseline：recirc ppl 更低的窗口数
            wins = sum(1 for i, p in enumerate(per_win)
                       if p < base_items[i][0])
            print(f"α={alpha:.2f} {pair}: ppl={ppl_r:.4f}  {pct:+.2f}%  "
                  f"(窗口胜出 {wins}/{len(windows)})  [{time.time()-t0:.0f}s]")
            entry = {"alpha": alpha, "src": src, "dst": dst,
                     "ppl": ppl_r, "pct_change": pct, "per_window": per_win,
                     "wins": wins}
            results.append(entry)
            if best is None or pct < best["pct_change"]:
                best = entry

    # ---------- 逐位置诊断（最优配置）----------
    print(f"\n=== 逐位置诊断（最优配置 α={best['alpha']} {best['src']}-{best['dst']}）===")
    # 重新跑 baseline 和最优配置，逐位置比较 log-likelihood
    # 只在第一个窗口上做（省时间，趋势已足够）
    ids = windows[0]
    with torch.no_grad():
        logits_b = bundle.model(ids.unsqueeze(0)).logits[0]
    logits_r = recirc_logits(
        bundle, ids,
        RecircParams(source=best["src"], dest=best["dst"],
                     alpha=best["alpha"], ramp=10))
    lp_b = F.log_softmax(logits_b.float(), -1).gather(
        1, ids[1:].unsqueeze(1)).squeeze(1)
    lp_r = F.log_softmax(logits_r.float(), -1).gather(
        1, ids[1:].unsqueeze(1)).squeeze(1)
    gain = lp_r - lp_b
    print("位置桶平均增益（正 = recirc 更好；应随位置增长）:")
    for b in range(0, ids.numel() - 1, 128):
        seg = gain[b:min(b + 128, ids.numel() - 1)]
        print(f"  pos {b:4d}-{min(b+127, ids.numel()-2):4d}: {seg.mean().item():+.4f}")

    with open(args.out, "w") as f:
        json.dump({"baseline_ppl": ppl_b, "n_windows": len(windows),
                   "window": args.window, "results": results,
                   "env": get_env_fingerprint()}, f, indent=2)
    print(f"\n结果已保存: {args.out}")


if __name__ == "__main__":
    main()
