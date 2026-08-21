#!/usr/bin/env python3
"""
真实长文本评估：PG-19 test 文档开头 1024 token
baseline vs recirculation（多 α × 多层对扫描）

用法:
  python3 eval_real.py --n_docs 3 --alphas 0.07 0.10 --layer_pairs 11-4 10-4 12-5
"""
import argparse
import json
import math
import time

import torch

from recirculation import (load_model, eval_baseline_ppl, recirc_logits,
                           perplexity_from_logits, RecircParams)


def get_pg19_docs(n_docs: int, min_len: int = 4000):
    """从 emozilla/pg19（parquet 镜像，免脚本）streaming 取 n_docs 个长文档"""
    import datasets
    # 新版 datasets 不支持脚本数据集，用 parquet 镜像
    ds = datasets.load_dataset("emozilla/pg19", split="test", streaming=True)
    docs = []
    for ex in ds:
        text = ex["text"]
        if len(text) >= min_len:
            docs.append(text)
            if len(docs) >= n_docs:
                break
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_docs", type=int, default=3)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.07, 0.10])
    ap.add_argument("--layer_pairs", type=str, nargs="+",
                    default=["11-4", "10-4", "12-5", "8-3"])
    ap.add_argument("--out", default="results_real.json")
    args = ap.parse_args()

    bundle = load_model("google/gemma-3-1b-pt")
    print(f"\n=== 获取 PG-19 test 文档（前 {args.n_docs} 个长文档）===")
    docs = get_pg19_docs(args.n_docs)
    print(f"获取 {len(docs)} 个文档，取每文档开头 {args.window} token")
    windows = []
    for i, text in enumerate(docs):
        ids = bundle.tokenizer(
            text, truncation=True, max_length=args.window,
            return_tensors="pt").input_ids[0].to(bundle.device)
        windows.append(ids)
        print(f"  doc {i}: {ids.numel()} tokens")

    # ---- baseline（并行 prefill，每文档一次）----
    t0 = time.time()
    nll_b, cnt_b = 0.0, 0
    per_doc_b = []
    for ids in windows:
        p = eval_baseline_ppl(bundle, ids)
        nll_b += math.log(p) * (ids.numel() - 1)
        cnt_b += ids.numel() - 1
        per_doc_b.append(p)
    ppl_b = math.exp(nll_b / cnt_b)
    print(f"\nbaseline ppl = {ppl_b:.4f}  (per-doc: "
          f"{[f'{p:.3f}' for p in per_doc_b]})  [{time.time()-t0:.0f}s]")

    # ---- recirculation 扫描 ----
    results = []
    for alpha in args.alphas:
        for pair in args.layer_pairs:
            src, dst = map(int, pair.split("-"))
            t0 = time.time()
            nll_r, cnt_r = 0.0, 0
            per_doc_r = []
            for ids in windows:
                logits = recirc_logits(
                    bundle, ids,
                    RecircParams(source=src, dest=dst, alpha=alpha, ramp=10))
                p = perplexity_from_logits(logits[:-1], ids[1:])
                nll_r += math.log(p) * (ids.numel() - 1)
                cnt_r += ids.numel() - 1
                per_doc_r.append(p)
            ppl_r = math.exp(nll_r / cnt_r)
            pct = (ppl_r / ppl_b - 1) * 100
            print(f"α={alpha:.2f} src={src} dst={dst}: ppl={ppl_r:.4f}  "
                  f"{pct:+.2f}%  (per-doc: {[f'{p:.3f}' for p in per_doc_r]})  "
                  f"[{time.time()-t0:.0f}s]")
            results.append({"alpha": alpha, "src": src, "dst": dst,
                            "ppl": ppl_r, "pct_change": pct,
                            "per_doc": per_doc_r})

    with open(args.out, "w") as f:
        json.dump({"baseline_ppl": ppl_b, "n_docs": len(windows),
                   "results": results}, f, indent=2)
    print(f"\n结果已保存: {args.out}")


if __name__ == "__main__":
    main()
