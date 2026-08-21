#!/usr/bin/env python3
"""
困惑度评估：baseline vs recirculation（论文 Table 1 的轻量复现）

用法示例：
  python3 eval_ppl.py --datasets pg19 --n_windows 50 --window 512
  python3 eval_ppl.py --datasets pg19 arxiv c4 --n_windows 200 --window 1024 --out results_p0.json
"""
import argparse
import json
import math
import time

import torch

from recirculation import (ModelBundle, RecircParams, eval_baseline_ppl,
                           load_model, perplexity_from_logits, recirc_logits)

DATASET_CONFIG = {
    # name: (hf dataset id, text column, split)
    "pg19": ("pg19", "text", "test"),
    "arxiv": ("monology/arxiv", "text", "train"),
    "c4": ("c4", "text", "validation"),
}

# 内置长文本（无需下载数据集，用于快速验证；~300 token）
BUILTIN_TEXT = (
    "The Roman Empire began in 27 BC when Augustus became the first emperor. "
    "It controlled the Mediterranean world for centuries. The empire built roads, aqueducts, and cities. "
    "Trade flourished across its provinces. Latin was the language of law and administration. "
    "The army protected the borders from invaders. Emperors ruled with absolute power. "
    "Eventually the western empire fell in 476 AD. The eastern empire continued for another thousand years. "
    "Its capital Constantinople was a center of learning and commerce. "
    "Historians study its rise and fall to understand how civilizations change over time. "
    "The economy depended on agriculture, taxation, and trade networks that spanned three continents. "
    "Merchants carried silk, spices, and glass across the Mediterranean Sea. "
    "Scholars preserved Greek philosophy and science in libraries and monasteries. "
    "The legal system developed principles that still influence modern law. "
    "Engineering achievements included aqueducts, roads, and monumental public buildings. "
    "Citizens enjoyed public baths, theaters, and chariot races in the capital. "
    "The army's discipline and organization made it the most effective fighting force of its age. "
    "Emperors often came from military backgrounds, reflecting the army's political power. "
    "Religious diversity marked the early empire, with temples to many gods across the provinces. "
    "The spread of Christianity eventually transformed the empire's culture and institutions. "
    "Historians debate the causes of the empire's decline, citing economic strain and political instability. "
    "Its legacy endures in language, law, architecture, and the idea of a unified Mediterranean world. "
)


def get_windows(dataset_name: str, n_windows: int, window: int, seed: int = 0):
    """从数据集取 n_windows 个窗口；dataset_name='builtin' 时用内置文本。"""
    if dataset_name == "builtin":
        texts = [BUILTIN_TEXT * 4] * n_windows  # 每窗口重复 4 次凑够长度
        print(f"[data] builtin: {len(texts)} windows")
        return texts
    ds_id, col, split = DATASET_CONFIG[dataset_name]
    import datasets
    ds = datasets.load_dataset(ds_id, split=split, streaming=True)
    rng = __import__("random").Random(seed)
    windows, skipped = [], 0
    for ex in ds:
        text = ex[col]
        if not text or len(text) < window + 50:
            skipped += 1
            continue
        # 随机起点，保证窗口内无换行截断问题（粗粒度）
        start = rng.randint(0, len(text) - window - 50)
        windows.append(text[start:start + window])
        if len(windows) >= n_windows:
            break
    print(f"[data] {dataset_name}: {len(windows)} windows (skipped {skipped})")
    return windows


def tokenize_windows(bundle: ModelBundle, texts, window: int, device):
    """把文本 token 化，截断到 window。返回 list[1D LongTensor]。"""
    out = []
    for txt in texts:
        ids = bundle.tokenizer(txt, truncation=True, max_length=window,
                               return_tensors="pt").input_ids[0]
        out.append(ids.to(device))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-1b-pt")
    ap.add_argument("--datasets", nargs="+", default=["builtin"],
                    help="builtin(免下载) | pg19 | arxiv | c4")
    ap.add_argument("--n_windows", type=int, default=10)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--source", type=int, default=11)
    ap.add_argument("--dest", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--ramp", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="结果 JSON 输出路径")
    ap.add_argument("--no-baseline", action="store_true")
    args = ap.parse_args()

    bundle = load_model(args.model)
    device = bundle.device

    params = RecircParams(source=args.source, dest=args.dest,
                          alpha=args.alpha, ramp=args.ramp)

    results = {"model": args.model, "params": params.__dict__,
               "windows": args.n_windows, "window_len": args.window,
               "datasets": {}}

    for ds in args.datasets:
        print(f"\n===== dataset: {ds} =====")
        texts = get_windows(ds, args.n_windows, args.window, args.seed)
        ids_list = tokenize_windows(bundle, texts, args.window, device)

        # baseline 用并行 prefill
        ppl_b, ppl_r = None, None
        t0 = time.time()
        nll_b, nll_r = 0.0, 0.0
        cnt_b, cnt_r = 0, 0
        for i, ids in enumerate(ids_list):
            if not args.no_baseline:
                p = eval_baseline_ppl(bundle, ids)
                nll_b += math.log(p) * (ids.numel() - 1)
                cnt_b += ids.numel() - 1
            t0r = time.time()
            logits = recirc_logits(bundle, ids, params)
            p_r = perplexity_from_logits(logits[:-1], ids[1:])
            nll_r += math.log(p_r) * (ids.numel() - 1)
            cnt_r += ids.numel() - 1
            if (i + 1) % 10 == 0:
                print(f"  window {i+1}/{len(ids_list)}  "
                      f"recirc ppl={p_r:.3f}  ({time.time()-t0r:.1f}s/win)")
        if not args.no_baseline and cnt_b > 0:
            ppl_b = math.exp(nll_b / cnt_b)
        if cnt_r > 0:
            ppl_r = math.exp(nll_r / cnt_r)
        red = (ppl_r / ppl_b - 1) * 100 if ppl_b else None
        print(f"[结果] {ds}: baseline={ppl_b:.3f}  recirc={ppl_r:.3f}  "
              f"相对变化 {red:+.2f}%  (总耗时 {time.time()-t0:.1f}s)")
        results["datasets"][ds] = {
            "baseline_ppl": ppl_b, "recirc_ppl": ppl_r, "pct_change": red,
            "tokens_eval": cnt_r,
        }

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {args.out}")


if __name__ == "__main__":
    main()
