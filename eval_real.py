#!/usr/bin/env python3
"""
真实长文本评估：PG-19 test 文档开头 1024 token
================================================

【这个脚本做什么？】
    用 PG-19（Project Gutenberg 的 19 世纪英文书籍）的真实长文档，
    对比 baseline 与 recirculation 的困惑度，并扫描多组超参数
    （α 与 源/目标层组合）。

【为什么用 PG-19？】
    论文的核心主张是：recirculation 通过"跨 token 的状态传播"帮助模型
    跟踪长文本中的状态（人物、事件、指代）。只有真实的长文档里才有
    这种状态——用重复的短文本测不出来（我们实测：内置重复文本上
    recirculation 反而有害，换到 PG-19 后才出现收益）。

【用法】
    python3 eval_real.py --n_docs 2 --alphas 0.07 0.10 --layer_pairs 11-4 10-4 12-5
"""
import argparse   # 解析命令行参数
import json       # 结果存成 JSON 文件
import math       # math.log / math.exp（累加困惑度用）
import time       # 计时

import torch      # PyTorch

from recirculation import (load_model, eval_baseline_ppl, recirc_logits,
                           perplexity_from_logits, RecircParams)


def get_pg19_docs(n_docs: int, min_len: int = 4000):
    """
    从 PG-19 test 集取 n_docs 个足够长的文档（返回原始文本列表）。

    用 emozilla/pg19 这个 HuggingFace 仓库（parquet 格式）而不是官方
    deepmind/pg19，原因是：新版 datasets 库（5.x）不再支持旧式"数据集
    脚本"（pg19.py），而 parquet 格式可以直接流式读取。

    streaming=True：边下载边读，只取前几个文档，不用下载整个 11GB 数据集。

    参数：
      n_docs  : 需要的文档数
      min_len : 过滤条件——文本至少这么长（字符数）才要
                （太短的文档 token 化后凑不满窗口）
    """
    import datasets
    # 新版 datasets 不支持脚本数据集，用 parquet 镜像
    ds = datasets.load_dataset("emozilla/pg19", split="test", streaming=True)
    docs = []
    for ex in ds:
        text = ex["text"]                    # 每个样本的原始文本
        if len(text) >= min_len:             # 只收长文档
            docs.append(text)
            if len(docs) >= n_docs:          # 够了就停
                break
    return docs


def main():
    # ---------- 命令行参数 ----------
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_docs", type=int, default=3)          # 用几个文档
    ap.add_argument("--window", type=int, default=1024)       # 每文档取开头多少 token
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.07, 0.10])  # 要扫的 α
    ap.add_argument("--layer_pairs", type=str, nargs="+",
                    default=["11-4", "10-4", "12-5", "8-3"])  # 要扫的 源-目标 层对
    ap.add_argument("--out", default="results_real.json")     # 结果输出文件
    args = ap.parse_args()

    # ---------- 加载模型 ----------
    bundle = load_model("google/gemma-3-1b-pt")
    print(f"\n=== 获取 PG-19 test 文档（前 {args.n_docs} 个长文档）===")
    docs = get_pg19_docs(args.n_docs)
    print(f"获取 {len(docs)} 个文档，取每文档开头 {args.window} token")

    # ---------- 每个文档取开头 window 个 token，构成测试窗口 ----------
    windows = []
    for i, text in enumerate(docs):
        # tokenizer 把文本切成 token id；truncation=True + max_length 截断到 window
        ids = bundle.tokenizer(
            text, truncation=True, max_length=args.window,
            return_tensors="pt").input_ids[0].to(bundle.device)
        windows.append(ids)
        print(f"  doc {i}: {ids.numel()} tokens")

    # ---------- baseline：标准并行 prefill ----------
    # 累加"负对数似然 × token 数"再统一取 exp，等价于按 token 数加权的平均困惑度
    # （直接平均每个文档的 ppl 会被短文档偏置）
    t0 = time.time()
    nll_b, cnt_b = 0.0, 0
    per_doc_b = []
    for ids in windows:
        p = eval_baseline_ppl(bundle, ids)         # 单文档 baseline 困惑度
        nll_b += math.log(p) * (ids.numel() - 1)   # 累加 NLL（×token 数加权）
        cnt_b += ids.numel() - 1                   # 累加 token 数
        per_doc_b.append(p)                        # 记录每文档值（便于观察）
    ppl_b = math.exp(nll_b / cnt_b)                # 总体 baseline 困惑度
    print(f"\nbaseline ppl = {ppl_b:.4f}  (per-doc: "
          f"{[f'{p:.3f}' for p in per_doc_b]})  [{time.time()-t0:.0f}s]")

    # ---------- recirculation：扫描 α × 层对 ----------
    results = []
    for alpha in args.alphas:
        for pair in args.layer_pairs:
            src, dst = map(int, pair.split("-"))   # "11-4" -> src=11, dst=4
            t0 = time.time()
            nll_r, cnt_r = 0.0, 0
            per_doc_r = []
            for ids in windows:
                # recirc_logits：顺序 prefill + 两遍前向（核心，见 recirculation.py）
                logits = recirc_logits(
                    bundle, ids,
                    RecircParams(source=src, dest=dst, alpha=alpha, ramp=10))
                p = perplexity_from_logits(logits[:-1], ids[1:])
                nll_r += math.log(p) * (ids.numel() - 1)
                cnt_r += ids.numel() - 1
                per_doc_r.append(p)
            ppl_r = math.exp(nll_r / cnt_r)
            pct = (ppl_r / ppl_b - 1) * 100         # 相对 baseline 的变化百分比
            print(f"α={alpha:.2f} src={src} dst={dst}: ppl={ppl_r:.4f}  "
                  f"{pct:+.2f}%  (per-doc: {[f'{p:.3f}' for p in per_doc_r]})  "
                  f"[{time.time()-t0:.0f}s]")
            results.append({"alpha": alpha, "src": src, "dst": dst,
                            "ppl": ppl_r, "pct_change": pct,
                            "per_doc": per_doc_r})

    # ---------- 保存结果 ----------
    with open(args.out, "w") as f:
        json.dump({"baseline_ppl": ppl_b, "n_docs": len(windows),
                   "results": results}, f, indent=2)
    print(f"\n结果已保存: {args.out}")


if __name__ == "__main__":
    main()
