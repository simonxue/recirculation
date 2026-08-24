#!/bin/bash
# =============================================================================
# Recirculation 复现 —— GPU 一键运行脚本（在 GPU 可用的 WSL 终端执行）
#
# 用法:
#   bash run_gpu.sh            # 快速验证（α=0 自检 + 200 token 单文本）
#   bash run_gpu.sh --full     # 完整实验（1024 窗口 × 3 个 α × 5 窗口 + 诊断）
#
# 前置条件:
#   - WSL 终端里 nvidia-smi 可见 RTX 2000 Ada（wsl --shutdown 后重开）
#   - 模型权重已缓存（.hf_cache/，已下载 1GB）
# =============================================================================
set -e
cd "$(dirname "$0")"

export HF_HOME="$PWD/.hf_cache"
# HF token 从环境变量读取（gated 模型必需）：export HF_TOKEN=... 或写入 ~/.netrc
export MPLCONFIGDIR="$PWD/.mpl"

echo "========== [1] 环境检查 =========="
python3 -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
print('显存:', torch.cuda.get_device_properties(0).total_memory/1024**3, 'GB') if torch.cuda.is_available() else None
assert torch.cuda.is_available(), 'CUDA 不可用！请在 GPU 可用的 WSL 终端运行'
"

echo ""
echo "========== [2] 快速验证：α=0 一致性 + 单文本效果 =========="
python3 - <<'EOF'
import torch, time
from recirculation import load_model, eval_baseline_ppl, recirc_logits, perplexity_from_logits, RecircParams

bundle = load_model("google/gemma-3-1b-pt")  # GPU + bf16
text = ("The Roman Empire began in 27 BC when Augustus became the first emperor. "
        "It controlled the Mediterranean world for centuries. The empire built roads, aqueducts, and cities. "
        "Trade flourished across its provinces. Latin was the language of law and administration. "
        "The army protected the borders from invaders. Emperors ruled with absolute power. "
        "Eventually the western empire fell in 476 AD. The eastern empire continued for another thousand years. "
        "Its capital Constantinople was a center of learning and commerce. "
        "Historians study its rise and fall to understand how civilizations change over time. ") * 2
ids = bundle.tokenizer(text, truncation=True, max_length=200, return_tensors="pt").input_ids[0].to(bundle.device)
print("tokens:", ids.numel())

# 1) α=0 一致性（实现正确性校验）
ppl_b = eval_baseline_ppl(bundle, ids)
t0 = time.time()
logits0 = recirc_logits(bundle, ids, RecircParams(source=11, dest=4, alpha=0.0, ramp=0))
ppl_0 = perplexity_from_logits(logits0[:-1], ids[1:])
d = abs(ppl_0 / ppl_b - 1)
print(f"baseline ppl = {ppl_b:.6f}")
print(f"recirc(α=0)  = {ppl_0:.6f}  相对差 = {d*100:.6f}%  ({time.time()-t0:.1f}s)")
assert d < 1e-3, "α=0 一致性失败！实现有 bug"
print("✅ α=0 一致性通过")

# 2) α=0.15 效果（论文 1B 配置 {src=11, dst=4}）
t0 = time.time()
logits = recirc_logits(bundle, ids, RecircParams(source=11, dest=4, alpha=0.15, ramp=10))
ppl_r = perplexity_from_logits(logits[:-1], ids[1:])
print(f"recirc(α=0.15) = {ppl_r:.4f}  相对变化 = {(ppl_r/ppl_b-1)*100:+.2f}%  ({time.time()-t0:.1f}s)")
print("（注：短窗口收益可能为负；论文图 9 显示收益随 lag 增长，需长窗口）")
EOF

MODE="${1:---quick}"
if [ "$MODE" = "--full" ]; then
    echo ""
    echo "========== [3] 完整实验：1024 窗口 × α∈{0.07,0.10,0.15} × {src=11,dst=4} =========="
    python3 eval.py --datasets builtin --n_windows 5 --window 1024 \
        --source 11 --dest 4 --alpha 0.07 --out results_a007.json
    python3 eval.py --datasets builtin --n_windows 5 --window 1024 \
        --source 11 --dest 4 --alpha 0.10 --out results_a010.json
    python3 eval.py --datasets builtin --n_windows 5 --window 1024 \
        --source 11 --dest 4 --alpha 0.15 --out results_a015.json

    echo ""
    echo "========== [4] 诊断：逐位置收益分布（1024 窗口, α=0.10）=========="
    python3 - <<'EOF'
import torch, time
from recirculation import load_model, recirc_logits, RecircParams, eval_baseline_ppl
import torch.nn.functional as F
from eval import BUILTIN_TEXT

bundle = load_model("google/gemma-3-1b-pt")
text = BUILTIN_TEXT * 4
ids = bundle.tokenizer(text, truncation=True, max_length=1024, return_tensors="pt").input_ids[0].to(bundle.device)
print("tokens:", ids.numel())

t0 = time.time()
logits0 = recirc_logits(bundle, ids, RecircParams(source=11, dest=4, alpha=0.0, ramp=0))
print(f"α=0 完成 ({time.time()-t0:.0f}s)")
t0 = time.time()
logitsA = recirc_logits(bundle, ids, RecircParams(source=11, dest=4, alpha=0.10, ramp=10))
print(f"α=0.10 完成 ({time.time()-t0:.0f}s)")

def ppl(logits, ids):
    lp = F.log_softmax(logits.float(), -1)
    nll = -lp.gather(1, ids[1:].unsqueeze(1)).squeeze(1)
    return torch.exp(nll.mean()).item()

pb, pa = ppl(logits0, ids), ppl(logitsA, ids)
print(f"\nα=0 ppl = {pb:.4f}   α=0.10 ppl = {pa:.4f}   相对变化 = {(pa/pb-1)*100:+.2f}%")

# 逐位置：每个 token 的 log-likelihood 增益（α=0.10 vs α=0）
lp0 = F.log_softmax(logits0.float(), -1).gather(1, ids[1:].unsqueeze(1)).squeeze(1)
lpA = F.log_softmax(logitsA.float(), -1).gather(1, ids[1:].unsqueeze(1)).squeeze(1)
gain = (lpA - lp0)
print("\n位置桶平均 log-likelihood 增益（正=recirc 更好）：")
for b in range(0, 1023, 64):
    seg = gain[b:min(b+64, 1023)]
    print(f"  pos {b:4d}-{b+63:4d}: {seg.mean().item():+.4f}  (n={seg.numel()})")
EOF

    echo ""
    echo "========== 结果汇总 =========="
    echo "--- results_a007.json ---"; cat results_a007.json 2>/dev/null
    echo "--- results_a010.json ---"; cat results_a010.json 2>/dev/null
    echo "--- results_a015.json ---"; cat results_a015.json 2>/dev/null
else
    echo ""
    echo "（快速模式完成。完整实验: bash run_gpu.sh --full）"
fi
