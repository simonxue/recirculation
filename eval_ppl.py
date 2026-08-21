#!/usr/bin/env python3
"""
困惑度评估：baseline vs recirculation（论文 Table 1 的轻量复现）
================================================================

【这个脚本做什么？】
    从数据集（或内置文本）取若干"窗口"（一段固定长度的 token 序列），
    对每个窗口分别计算：
      1. baseline 困惑度（标准并行 prefill）
      2. recirculation 困惑度（顺序两遍前向）
    然后汇总对比，输出每个数据集的相对变化百分比。

【数据集的两种来源】
    - builtin（默认）：用内置的一段罗马史英文文本（无需下载任何数据，
      适合快速验证管线是否跑通）
    - pg19 / arxiv / c4：从 HuggingFace 数据集流式取窗口
      （注意：这些需要联网下载，且 pg19 官方仓库是旧式脚本，已被新版
      datasets 拒绝——需要真实长文评估时请用 eval_real.py 的 emozilla 镜像）

【用法示例】
    python3 eval_ppl.py --datasets builtin --n_windows 10 --window 512
    python3 eval_ppl.py --datasets pg19 --n_windows 50 --window 512
"""
import argparse   # 命令行参数
import json       # 结果存 JSON
import math       # math.log / math.exp
import time       # 计时

import torch      # PyTorch

from recirculation import (ModelBundle, RecircParams, eval_baseline_ppl,
                           load_model, perplexity_from_logits, recirc_logits)

# 数据集配置表：名字 -> (HuggingFace 仓库 id, 文本字段名, 划分)
DATASET_CONFIG = {
    # name: (hf dataset id, text column, split)
    "pg19": ("pg19", "text", "test"),
    "arxiv": ("monology/arxiv", "text", "train"),
    "c4": ("c4", "text", "validation"),
}

# 内置长文本（无需下载数据集，用于快速验证；~300 token）
# 注意：这段文本是"重复拼接"的，模型对它几乎确定（ppl 很低），
# 没有跨句状态可跟踪，所以 recirculation 在这种测试集上测不出收益。
# 真正验证效果请用 eval_real.py 配 PG-19 真实书籍。
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
    """
    从数据集取 n_windows 个窗口（每个窗口是一段文本）。

    窗口 = 一段足够长的连续文本，稍后会被 token 化并截断到 window 长度。

    参数：
      dataset_name : "builtin"（内置文本）或 DATASET_CONFIG 里的名字
      n_windows    : 要取几个窗口
      window       : 窗口目标长度（token 数）
      seed         : 随机种子（决定从文档哪个位置切窗口，保证可复现）

    对真实数据集：
      每个文档只取至多一个窗口；从文档中随机位置切一段（前后留 50 字符
      余量，避免切到半截句子/换行），跳过太短的文档。
    """
    if dataset_name == "builtin":
        # 内置文本：重复拼接 4 遍凑够长度（每窗口都用同一段文本）
        texts = [BUILTIN_TEXT * 4] * n_windows  # 每窗口重复 4 次凑够长度
        print(f"[data] builtin: {len(texts)} windows")
        return texts
    ds_id, col, split = DATASET_CONFIG[dataset_name]
    import datasets
    # streaming=True：边下载边迭代，不用等整个数据集下载完
    ds = datasets.load_dataset(ds_id, split=split, streaming=True)
    rng = __import__("random").Random(seed)   # 独立随机数生成器（不影响全局种子）
    windows, skipped = [], 0
    for ex in ds:
        text = ex[col]
        if not text or len(text) < window + 50:   # 太短：切不出完整窗口
            skipped += 1
            continue
        # 随机起点，保证窗口内无换行截断问题（粗粒度）
        start = rng.randint(0, len(text) - window - 50)
        windows.append(text[start:start + window])
        if len(windows) >= n_windows:   # 够了就停（streaming 的好处）
            break
    print(f"[data] {dataset_name}: {len(windows)} windows (skipped {skipped})")
    return windows


def tokenize_windows(bundle: ModelBundle, texts, window: int, device):
    """
    把文本列表 token 化，每段截断到 window 长度。

    返回 list[1D LongTensor]：每个元素是一个窗口的 token id 序列
    （已移到目标设备）。
    """
    out = []
    for txt in texts:
        # truncation=True + max_length=window：超长截断，短则保留全部
        ids = bundle.tokenizer(txt, truncation=True, max_length=window,
                               return_tensors="pt").input_ids[0]
        out.append(ids.to(device))
    return out


def main():
    # ---------- 命令行参数 ----------
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-1b-pt")
    ap.add_argument("--datasets", nargs="+", default=["builtin"],
                    help="builtin(免下载) | pg19 | arxiv | c4")
    ap.add_argument("--n_windows", type=int, default=10)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--source", type=int, default=11)   # 源层（论文 1B 最优 11）
    ap.add_argument("--dest", type=int, default=4)      # 目标层（论文 1B 最优 4）
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--ramp", type=int, default=10)     # 预热步数
    ap.add_argument("--seed", type=int, default=0)      # 窗口采样种子
    ap.add_argument("--out", default=None, help="结果 JSON 输出路径")
    ap.add_argument("--no-baseline", action="store_true")  # 只跑 recirc（省一半时间）
    args = ap.parse_args()

    # ---------- 加载模型，构造参数 ----------
    bundle = load_model(args.model)
    device = bundle.device

    params = RecircParams(source=args.source, dest=args.dest,
                          alpha=args.alpha, ramp=args.ramp)

    results = {"model": args.model, "params": params.__dict__,
               "windows": args.n_windows, "window_len": args.window,
               "datasets": {}}

    # ---------- 对每个数据集评估 ----------
    for ds in args.datasets:
        print(f"\n===== dataset: {ds} =====")
        texts = get_windows(ds, args.n_windows, args.window, args.seed)
        ids_list = tokenize_windows(bundle, texts, args.window, device)

        # baseline 用并行 prefill（快）；recirc 用顺序前向（慢）
        # 汇总方式：累加 NLL×token 数，最后统一 exp（按 token 加权平均）
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
            if (i + 1) % 10 == 0:   # 每 10 个窗口打印一次进度
                print(f"  window {i+1}/{len(ids_list)}  "
                      f"recirc ppl={p_r:.3f}  ({time.time()-t0r:.1f}s/win)")
        # 汇总：NLL 总量 / token 总量 -> exp
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

    # ---------- 保存结果 ----------
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {args.out}")


if __name__ == "__main__":
    main()
