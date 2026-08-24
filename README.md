# Recirculation (Paper Reproduction)

**零训练、纯推理时的 Transformer 架构增强** —— 把深层激活"回流"到浅层，给前馈 Transformer 注入跨时间步的循环状态跟踪能力，显著降低困惑度并提升下游性能。

**A training-free, inference-only architecture enhancement for Transformers** — recirculating a small fraction of deep-layer activations back into a shallow layer gives the feed-forward Transformer recurrent, cross-timestep state tracking, reducing perplexity and improving downstream performance.

> ⚠️ **非官方实现 / Unofficial implementation.** This is an independent re-implementation of the paper *Recirculation* (arXiv:2608.17981) by Michael C. Mozer et al. (Google DeepMind / UT Austin). The authors have not released official code; this repository is written from the paper's description and validated by local experiments. 论文作者未发布官方代码，本仓库系按论文描述独立实现，并经本地实验验证。
> 更多文档：论文理解与完整复现报告见 [`reading.md`](reading.md)；复现计划及其演变见 [`repro_plan.md`](repro_plan.md)。

---

## 目录 / Table of Contents

- [核心思想 / Core Idea](#核心思想--core-idea)
- [复现结果 / Reproduction Results](#复现结果--reproduction-results)
- [快速开始 / Quick Start](#快速开始--quick-start)
- [复现论文实验 / Reproducing the Paper's Experiments](#复现论文实验--reproducing-the-papers-experiments)
- [文件清单 / File Layout](#文件清单--file-layout)
- [与论文的对照 / Comparison with the Paper](#与论文的对照--comparison-with-the-paper)
- [许可与致谢 / License & Acknowledgments](#许可与致谢--license--acknowledgments)

---

## 核心思想 / Core Idea

普通 Transformer 是纯前馈的：信息只能从底层流到顶层，模型在最深层才完成的**语义消歧**（如确定 *bank* 是"河岸"还是"银行"），浅层看不到，后续 token 在浅层处理时只能用模棱两可的信息。

**Recirculation 的做法**：处理每个 token 时跑"两遍"——

1. **第一遍**：正常增量前向（复用 KV cache），记录每一层的残差流向量；
2. **第二遍**：把**深层某层**（source，如第 11 层）的输出按小比例 α 混入**浅层某层**（dest，如第 4 层），从 dest 层重算到顶层，用第二遍的结果预测下一个 token；同时把 dest..top 层的 KV 原位覆盖为 recirculated 版本，让后续 token 看到"已回流"的状态。

核心公式（`_mix` 函数）：

```
z_d' = α · f(z_s) + β · z_d          （公式 1：凸混合）
f(z_s) = z_s · ||z_d||₂ / ||z_s||₂  （公式 2：源向量缩放到目标层长度）
```

两个重要细节：

- **ramping 预热**：窗口开头前 10 个 token 把 α 从 0 线性升到目标值（`α_t = min(t/10, 1)·α`，论文附录 B.3）——开头还没有历史状态可传播，直接混合反而有害；
- **KV cache 覆盖**：第二遍重算的 K/V 必须"覆盖"第一遍的（自定义 `OverwriteCache`，包装 transformers `DynamicCache`），否则后续 token 看到的是未 recirculate 的状态。

零训练、零权重修改；代价是 prefill 阶段必须**串行**处理（无法并行），生成阶段几乎零额外延迟。

---

## 复现结果 / Reproduction Results

> 环境：NVIDIA RTX 2000 Ada Laptop 8GB（WSL2）/ torch 2.11+cu130 / transformers 4.57.6 / **$0 硬件成本**。
> 完整报告（含逐位置诊断与负面对照）见 [`reading.md`](reading.md)。

**结论：✅ 复现成功** —— 在 PG-19 真实长文档上，全部 6 组配置一致降低困惑度，窗口胜出率 72%–89%。

| 配置 | ppl 变化 | 窗口胜出率 |
|---|---|---|
| **α=0.07, 12-5** | **−7.93%** | 13/18 (72%) |
| α=0.07, 10-4 | −4.59% | 15/18 (83%) |
| α=0.07, 11-4 | −4.28% | 15/18 (83%) |
| α=0.04, 12-5 | −5.43% | 14/18 (78%) |
| α=0.04, 10-4 | −3.33% | 14/18 (78%) |
| α=0.04, 11-4 | −3.28% | 16/18 (89%) |

- 评估设置：PG-19 6 个文档 × 每文档 3 个位置窗口 = 18 窗口 × 1024 token（`results_strict_step1.json`）
- baseline 困惑度 = 27.30（多位置窗口含文档中后部更难位置）；文档开头窗口 baseline = 19.13（≈ 论文 arXiv 19.10，实现校准良好）
- 正确性校验：α=0 时 recirculation 与标准前向一致（相对差 0.0028%，纯浮点噪声）

---

## 快速开始 / Quick Start

### 环境要求 / Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.4（CUDA 版按 [pytorch.org](https://pytorch.org/get-started/locally/) 安装；CPU 也能跑，但 recirculation 是顺序前向，速度很慢）
- `transformers` **4.x**（本项目基于 4.x 的 Gemma3 层接口实现；5.x 重构了该 API，会给出明确报错。复现环境实测版本 4.57.6）
- `datasets`

```bash
pip install -r requirements.txt
# CUDA 版 torch 需单独按官网指引安装，例如：
# pip install torch --index-url https://download.pytorch.org/whl/cu130
```

### 冒烟自检（无需 GPU、无需 token，~10 秒）

用 tiny-random Gemma3（公开权重）在 CPU 上验证整条管线 + α=0 一致性：

```bash
python3 smoke_test.py
```

### 真实模型快速体验（需要 GPU + HF token）

Gemma3 是 **gated 模型**：需先在 [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) 申请访问权限并获取 token，然后：

```bash
export HF_TOKEN=hf_xxx   # 或写入 ~/.netrc
python3 recirculation.py --text "The capital of France is" --alpha 0.15
# 输出 baseline ppl 与 recirculation ppl 及相对变化
```

CPU 也能跑真实模型（自动降级 float32），但 1024 token 窗口会非常慢，建议仅用短文本验证。

---

## 复现论文实验 / Reproducing the Paper's Experiments

```bash
# [1] 环境自检（GPU/CUDA）
python3 check_gpu.py

# [2] 快速验证：α=0 一致性 + 短文本效果（GPU 终端）
bash run_gpu.sh

# [3] 完整实验：PG-19 多文档 × 多位置窗口 × 多配置 + 逐位置诊断（GPU，较慢）
python3 eval.py --datasets pg19 --n_docs 6 --positions 3 \
    --alphas 0.04 0.07 0.10 --layer_pairs 11-4 10-4 12-5 --diagnose \
    --out results_strict.json
```

> 💡 `eval.py` 还支持：多数据集对比（`--datasets builtin / arxiv / c4`）、文档开头扫描（`--n_docs 2 --positions 1`）、单配置快速评估（`--source 11 --dest 4 --alpha 0.07`）。PG-19 经 `emozilla/pg19` parquet 镜像流式读取（新版 `datasets` 不再支持官方脚本数据集）。

结果 JSON 均包含 `env` 字段（Python/torch/transformers 版本、GPU 名），便于对照复现。

---

## 文件清单 / File Layout

| 文件 | 说明 |
|---|---|
| `recirculation.py` | **核心实现**：模型加载、顺序 prefill、两遍前向、`OverwriteCache`、困惑度评估（教学式注释） |
| `eval.py` | 统一评估入口：多数据集（builtin/pg19/arxiv/c4）× 窗口采样（随机位置 / 多文档多位置）× α 与层对扫描 × 逐位置诊断 |
| `run_gpu.sh` / `check_gpu.py` | GPU 一键脚本 / 环境自检 |
| `smoke_test.py` | 无 GPU 冒烟自检（CI 用） |
| `results_real.json` / `results_strict_step1.json` | 核心实验原始结果（含环境指纹） |
| `reading.md` | 论文理解 + 完整复现报告（环境/实现/数据/对照/结论） |
| `repro_plan.md` | 复现计划及其演变（计划 vs 实际、方案转变） |

---

## 与论文的对照 / Comparison with the Paper

| 论文主张 | 论文数值 | 我们的复现 | 一致性 |
|---|---|---|---|
| 真实数据上 recirc 降 ppl | −14.4% (PG-19 全 test set) | −3.3% ~ −7.9%（18 窗口） | ✅ 方向一致，幅度偏小（样本少） |
| 最优层对 | {11, 4}（1B） | {12, 5}（相邻层对） | ✅ 相邻 |
| α 不宜过大 | 扫描最优 ~0.07–0.10 | 0.07 > 0.04 > 0.10（趋势） | ✅ |
| 收益随位置/lag 增长 | 图 9 幂律衰减 | 后段增益增强（+0.0196 峰值） | ✅ |
| 短序列无收益 | lambada 异常（短文） | 200–512 token 负收益 | ✅ |
| baseline 校准 | arXiv 19.10 | 19.13（文档开头） | ✅ |

> 幅度差异（−7.9% vs −14.4%）的原因分析见 [`reading.md`](reading.md) §5。

---

## 许可与致谢 / License & Acknowledgments

- **代码许可**：本仓库代码采用 [Apache License 2.0](LICENSE)。
- **论文**：*Recirculation* — Michael C. Mozer, Shoaib Ahmed Siddiqui, Danny Sawyer, Sunny Sanyal, Rosanne Liu (arXiv:2608.17981, 2026)。本实现未使用作者任何代码，仅基于论文公开描述。

```bibtex
@misc{recirculation2026,
  title={Recirculation},
  author={Mozer, Michael C. and Siddiqui, Shoaib Ahmed and Sawyer, Danny and Sanyal, Sunny and Liu, Rosanne},
  year={2026},
  eprint={2608.17981},
  archivePrefix={arXiv},
}
```

- **模型**：`google/gemma-3-1b-pt` 受 [Gemma Terms of Use](https://ai.google.dev/gemma/terms) 约束（gated，商用需另行确认）；评测数据集 PG-19 版权归原始出版社/作者所有，仅用于研究。本仓库不包含任何模型权重与数据集内容。
- **致谢**：`emozilla/pg19`（parquet 镜像）、`optimum-intel-internal-testing/tiny-random-gemma3-text`（CI 冒烟测试用 tiny 模型）。
- **生成说明**：本仓库由 DeepSeek-V4-Flash + DeepSeek Harness + dsh-TUI + 人工 Prompt 协作完成。
