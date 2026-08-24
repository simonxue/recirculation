#!/usr/bin/env python3
"""
评估：baseline vs recirculation（统一入口，覆盖论文 Table 1 / 图 9 的复现）
============================================================================

【这个脚本做什么？】
    从数据集（或内置文本）取若干"窗口"（一段固定长度的 token 序列），
    对每个窗口分别计算：
      1. baseline 困惑度（标准并行 prefill）
      2. recirculation 困惑度（顺序两遍前向，见 recirculation.py）
    然后汇总对比，输出每个数据集的相对变化百分比。

【它统一了旧 eval_ppl.py / eval_real.py / eval_strict.py 的全部能力】
    - 数据源：builtin（免下载，快速验证）/ pg19 / arxiv / c4
    - 窗口采样：文档开头（--mode doc-start）或 开头+随机位置（--mode multi-pos）
    - 参数扫描：多 α × 多 (源,目标) 层对（--alphas / --layer_pairs）
    - 逐位置诊断：最优配置下每 128-token 桶的 log-likelihood 增益
      （对照论文图 9：收益应随滞后/位置增长）

【用法示例】
    # 快速验证（内置文本，免下载，单配置）
    python3 eval.py --datasets builtin --n_windows 10 --window 512
    # 真实长文 + 参数扫描（论文核心实验）
    python3 eval.py --datasets pg19 --n_docs 6 --positions 3 \
        --alphas 0.04 0.07 0.10 --layer_pairs 11-4 10-4 12-5 --diagnose
    # 多数据集对比（Table 1 风格）
    python3 eval.py --datasets pg19 arxiv c4 --n_windows 50 --window 512

【为什么用 PG-19 而不用内置文本？】
    论文的核心主张是：recirculation 通过"跨 token 的状态传播"帮助模型
    跟踪长文本中的状态（人物、事件、指代）。只有真实的长文档里才有
    这种状态——用重复的短文本测不出来（实测：内置重复文本上 recirculation
    反而有害，换到 PG-19 后才出现收益）。
"""
import argparse   # 命令行参数
import json       # 结果存成 JSON 文件
import math       # math.log / math.exp（累加困惑度用）
import random     # 窗口位置采样
import time       # 计时

import torch      # PyTorch
import torch.nn.functional as F

from recirculation import (ModelBundle, RecircParams, eval_baseline_ppl,
                           get_env_fingerprint, load_model,
                           perplexity_from_logits, recirc_logits)

# 数据集配置表：名字 -> (HuggingFace 仓库 id, 文本字段名, 划分)
# 注意：pg19 用 emozilla 镜像（parquet）而非官方 deepmind/pg19——新版
# datasets（5.x）不再支持旧式"脚本数据集"，parquet 可以直接流式读取。
DATASET_CONFIG = {
    # name: (hf dataset id, text column, split)
    "pg19": ("emozilla/pg19", "text", "test"),
    "arxiv": ("monology/arxiv", "text", "train"),
    "c4": ("c4", "text", "validation"),
}

# 内置长文本（无需下载数据集，用于快速验证；~300 token）
# 注意：这段文本是"重复拼接"的，模型对它几乎确定（ppl 很低），
# 没有跨句状态可跟踪，所以 recirculation 在这种测试集上测不出收益。
# 真正验证效果请用 --datasets pg19 配真实书籍。
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
        texts = [BUILTIN_TEXT * 4] * n_windows
        print(f"[data] builtin: {len(texts)} windows")
        return texts
    ds_id, col, split = DATASET_CONFIG[dataset_name]
    import datasets
    # streaming=True：边下载边迭代，不用等整个数据集下载完
    ds = datasets.load_dataset(ds_id, split=split, streaming=True)
    rng = random.Random(seed)   # 独立随机数生成器（不影响全局种子）
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


def get_pg19_docs(n_docs: int, min_len: int = 4000):
    """
    从 emozilla/pg19（parquet 镜像）streaming 取 n_docs 个足够长的文档
    （返回原始文本列表）。

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
    从每个文档取 positions 个位置窗口，返回 token id 列表（已在目标设备）。

    策略：文档开头必取一个（与论文实验一致性），其余在文档中后部
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


def weighted_ppl(ppls_and_counts):
    """按 token 数加权的平均困惑度：exp(Σ NLL·n / Σ n)。"""
    nll_sum = sum(math.log(p) * n for p, n in ppls_and_counts)
    cnt_sum = sum(n for _, n in ppls_and_counts)
    return math.exp(nll_sum / cnt_sum)


def parse_pairs(pairs):
    """把 "11-4" 形式的字符串列表解析为 [(src, dst), ...]。"""
    return [tuple(map(int, p.split("-"))) for p in pairs]


def main():
    ap = argparse.ArgumentParser(
        description="baseline vs recirculation 困惑度评估（统一入口）")
    ap.add_argument("--model", default="google/gemma-3-1b-pt")
    ap.add_argument("--datasets", nargs="+", default=["builtin"],
                    help="builtin(免下载) | pg19 | arxiv | c4")
    ap.add_argument("--n_windows", type=int, default=10,
                    help="每个数据集取多少个窗口（随机位置模式）")
    ap.add_argument("--window", type=int, default=512, help="窗口长度（token）")
    # ---- 真实长文模式（PG-19 多文档 × 多位置窗口，论文核心实验）----
    # 注意：显式传入 --n_docs 才启用文档模式；不传则 pg19 也走
    # 通用窗口模式（--n_windows 个随机位置窗口，即旧 eval_ppl 的行为）。
    ap.add_argument("--n_docs", type=int, default=None,
                    help="PG-19 用几个文档（传入即启用文档模式）")
    ap.add_argument("--positions", type=int, default=3,
                    help="文档模式下每文档取几个位置窗口（0 开头 + N-1 随机）")
    # ---- 参数扫描 ----
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.07],
                    help="要扫描的 α 列表")
    ap.add_argument("--layer_pairs", type=str, nargs="+",
                    default=["11-4"], help="要扫描的 源-目标 层对，如 11-4 12-5")
    # ---- 单配置快捷参数（不提供时用 --alphas/--layer_pairs 扫描）----
    ap.add_argument("--source", type=int, default=None, help="源层（单配置模式）")
    ap.add_argument("--dest", type=int, default=None, help="目标层（单配置模式）")
    ap.add_argument("--alpha", type=float, default=None, help="α（单配置模式）")
    ap.add_argument("--ramp", type=int, default=10, help="预热步数")
    ap.add_argument("--diagnose", action="store_true",
                    help="对最优配置输出逐位置增益桶（对照论文图 9）")
    ap.add_argument("--seed", type=int, default=0, help="窗口采样种子")
    ap.add_argument("--no-baseline", action="store_true",
                    help="只跑 recirc（省一半时间）")
    ap.add_argument("--out", default=None, help="结果 JSON 输出路径")
    args = ap.parse_args()

    # ---------- 确定配置列表：单配置快捷参数 或 网格扫描 ----------
    if args.source is not None or args.dest is not None or args.alpha is not None:
        # 单配置模式：未显式给出的用默认值补齐
        pairs = [(args.source if args.source is not None else 11,
                  args.dest if args.dest is not None else 4)]
        alphas = [args.alpha if args.alpha is not None else 0.15]
    else:
        pairs = parse_pairs(args.layer_pairs)
        alphas = args.alphas
    ramp = args.ramp

    # ---------- 加载模型，构造窗口 ----------
    bundle = load_model(args.model)
    device = bundle.device

    # 文档模式（PG-19 多文档 × 多位置）与窗口模式（随机位置/内置文本）二选一：
    # 显式给了 --n_docs/--positions 且数据集为 pg19 时用文档模式，
    # 否则走通用窗口模式（builtin / arxiv / c4 / 单文档随机窗口）。
    doc_mode = (args.n_docs is not None) and "pg19" in args.datasets

    results = {"model": args.model, "params": {"alphas": alphas,
               "layer_pairs": [f"{s}-{d}" for s, d in pairs], "ramp": ramp},
               "window_len": args.window, "datasets": {},
               "env": get_env_fingerprint()}

    for ds in args.datasets:
        print(f"\n===== dataset: {ds} =====")
        if doc_mode:
            print(f"=== 获取 {args.n_docs} 个 PG-19 文档，每文档取 "
                  f"{args.positions} 个位置窗口 ===")
            docs = get_pg19_docs(args.n_docs)
            ids_list = sample_windows(bundle.tokenizer, docs, args.positions,
                                      args.window, args.seed, device)
            print(f"共 {len(ids_list)} 个窗口，每窗口 {args.window} token")
        else:
            texts = get_windows(ds, args.n_windows, args.window, args.seed)
            ids_list = tokenize_windows(bundle, texts, args.window, device)

        # ---------- baseline（并行 prefill，快）----------
        base_items = []   # (ppl, token 数)
        t0 = time.time()
        for i, ids in enumerate(ids_list):
            if not args.no_baseline:
                p = eval_baseline_ppl(bundle, ids)
                base_items.append((p, ids.numel() - 1))
        ppl_b = weighted_ppl(base_items) if base_items else None
        print(f"[baseline] ppl = {ppl_b:.4f}  ({len(ids_list)} 窗口, "
              f"{time.time()-t0:.0f}s)")

        # ---------- recirculation：扫描 α × 层对 ----------
        results["datasets"][ds] = {"baseline_ppl": ppl_b, "results": []}
        best = None   # 记录最优配置（供诊断用）
        for alpha in alphas:
            for src, dst in pairs:
                t0 = time.time()
                r_items = []
                per_win = []
                for ids in ids_list:
                    logits = recirc_logits(
                        bundle, ids,
                        RecircParams(source=src, dest=dst, alpha=alpha,
                                     ramp=ramp))
                    p = perplexity_from_logits(logits[:-1], ids[1:])
                    r_items.append((p, ids.numel() - 1))
                    per_win.append(p)
                ppl_r = weighted_ppl(r_items)
                pct = (ppl_r / ppl_b - 1) * 100 if ppl_b else None
                wins = (sum(1 for i, p in enumerate(per_win)
                            if p < base_items[i][0])
                        if base_items else None)
                print(f"α={alpha:.2f} {src}-{dst}: ppl={ppl_r:.4f}  "
                      f"{pct:+.2f}%" if pct is not None else
                      f"α={alpha:.2f} {src}-{dst}: ppl={ppl_r:.4f}  "
                      f"(--no-baseline 无相对变化)" +
                      (f"  (窗口胜出 {wins}/{len(ids_list)})"
                       if wins is not None else "") +
                      f"  [{time.time()-t0:.0f}s]")
                entry = {"alpha": alpha, "src": src, "dst": dst,
                         "ppl": ppl_r, "pct_change": pct,
                         "per_window": per_win, "wins": wins}
                results["datasets"][ds]["results"].append(entry)
                if best is None or (pct is not None and
                                    (best["pct_change"] is None or
                                     pct < best["pct_change"])):
                    best = entry

        # ---------- 逐位置诊断（最优配置，对照论文图 9）----------
        if args.diagnose and best is not None and not args.no_baseline:
            print(f"\n=== 逐位置诊断（最优配置 α={best['alpha']} "
                  f"{best['src']}-{best['dst']}）===")
            # 只在第一个窗口上做（省时间，趋势已足够）
            ids = ids_list[0]
            with torch.no_grad():
                logits_b = bundle.model(ids.unsqueeze(0)).logits[0]
            logits_r = recirc_logits(
                bundle, ids,
                RecircParams(source=best["src"], dest=best["dst"],
                             alpha=best["alpha"], ramp=ramp))
            lp_b = F.log_softmax(logits_b.float(), -1).gather(
                1, ids[1:].unsqueeze(1)).squeeze(1)
            lp_r = F.log_softmax(logits_r.float(), -1).gather(
                1, ids[1:].unsqueeze(1)).squeeze(1)
            gain = lp_r - lp_b
            print("位置桶平均增益（正 = recirc 更好；应随位置增长）:")
            for b in range(0, ids.numel() - 1, 128):
                seg = gain[b:min(b + 128, ids.numel() - 1)]
                print(f"  pos {b:4d}-{min(b+127, ids.numel()-2):4d}: "
                      f"{seg.mean().item():+.4f}")

    # ---------- 保存结果 ----------
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {args.out}")


if __name__ == "__main__":
    main()
