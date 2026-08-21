#!/usr/bin/env python3
"""
Recirculation 冒烟自检（无需 GPU / 无需 HF token）
====================================================
用 HuggingFace 上的 tiny-random Gemma3（2 层、随机权重、公开可下载）
在 CPU 上快速验证整条管线：
  1. 模型加载
  2. baseline 困惑度（标准并行 prefill）
  3. recirculation 顺序前向（核心逻辑）
  4. α=0 一致性：recirculation(α=0) 应与 baseline 完全一致（实现正确性）

用法:
  python3 smoke_test.py                # 跑全部检查
  python3 smoke_test.py --verbose      # 打印每一步耗时

适合的场景：
  - 安装完依赖后先跑一遍，确认环境没问题
  - CI（GitHub Actions）里作为冒烟测试，几秒内完成
  - 没有 GPU、也没有 Gemma3 访问权限时验证代码可运行

注意：
  tiny 模型是随机权重，困惑度数值没有意义（≈词表大小），
  这里只验证"管线能跑通 + α=0 一致"，不验证效果。
"""
import argparse
import time

from recirculation import (load_model, eval_baseline_ppl, recirc_logits,
                           perplexity_from_logits, RecircParams)

# tiny-random Gemma3（Gemma3ForCausalLM，2 层，hidden=8，Apache-2.0，公开）
# 选择理由：架构与 google/gemma-3-1b-pt 一致（含 sliding/full attention），
# 且自带 tokenizer、无需 HF token、权重仅几 MB。
TINY_MODEL = "optimum-intel-internal-testing/tiny-random-gemma3-text"

# 测试文本：够短（CPU 秒级跑完），且包含跨句内容
TEST_TEXT = ("The capital of France is Paris. "
             "The capital of Italy is Rome. "
             "The capital of Japan is Tokyo.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="打印每一步耗时")
    args = ap.parse_args()

    t_all = time.time()
    print(f"== Recirculation smoke test (model: {TINY_MODEL}) ==")

    # ---------- 1. 模型加载 ----------
    t0 = time.time()
    bundle = load_model(TINY_MODEL)
    if args.verbose:
        print(f"  [load] {time.time()-t0:.1f}s")

    # ---------- 2. token 化 ----------
    ids = bundle.tokenizer(TEST_TEXT, return_tensors="pt").input_ids[0]
    print(f"tokens: {ids.numel()}  |  layers: {bundle.n_layers}  "
          f"hidden: {bundle.hidden}  |  vocab: {bundle.config.vocab_size}")
    assert ids.numel() > 5, "测试文本太短，无法验证"

    # ---------- 3. baseline 困惑度 ----------
    t0 = time.time()
    ppl_b = eval_baseline_ppl(bundle, ids)
    if args.verbose:
        print(f"  [baseline] {time.time()-t0:.1f}s")
    print(f"baseline ppl = {ppl_b:.4f}")

    # ---------- 4. recirculation 顺序前向 ----------
    # tiny 模型只有 2 层，源/目标层用 {1, 0}
    t0 = time.time()
    logits = recirc_logits(
        bundle, ids, RecircParams(source=1, dest=0, alpha=0.1, ramp=2))
    if args.verbose:
        print(f"  [recirc] {time.time()-t0:.1f}s")
    ppl_r = perplexity_from_logits(logits[:-1], ids[1:])
    print(f"recirc  ppl = {ppl_r:.4f}")

    # 输出必须有限（无 NaN/Inf），形状必须正确
    assert logits.shape == (ids.numel(), bundle.config.vocab_size), \
        f"logits 形状错误: {tuple(logits.shape)}"
    assert torch_isfinite(logits), "logits 含 NaN/Inf"

    # ---------- 5. α=0 一致性（实现正确性核心校验） ----------
    logits0 = recirc_logits(
        bundle, ids, RecircParams(source=1, dest=0, alpha=0.0, ramp=0))
    ppl_0 = perplexity_from_logits(logits0[:-1], ids[1:])
    rel = abs(ppl_0 / ppl_b - 1)
    print(f"alpha=0 consistency: rel diff = {rel*100:.4f}%")
    assert rel < 1e-3, "α=0 一致性失败！recirculation 实现有 bug"

    print(f"\n✅ 全部通过（总耗时 {time.time()-t_all:.1f}s）")


def torch_isfinite(t):
    """logits 是否全部有限（不引入 torch 顶层 import，随用随取）。"""
    import torch
    return bool(torch.isfinite(t).all())


if __name__ == "__main__":
    main()
