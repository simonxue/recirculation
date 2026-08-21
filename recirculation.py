#!/usr/bin/env python3
"""
Recirculation 核心前向实现（arXiv:2608.17981）
================================================

【这篇代码在做什么？】
    这篇论文提出了一种"零训练、纯推理时"的技巧：不修改模型的任何权重，
    只在前向计算时做一点手脚，就能降低模型的困惑度（预测更准）。

【核心思想：把深层信息"漏"回浅层】
    普通的 Transformer 是"前馈"的：信息只能从底层流到顶层，不能回头。
    这导致一个问题：模型在最深层才完成的"语义消歧"（比如确定 bank 是
    "河岸"还是"银行"），浅层根本看不到，后续 token 在浅层处理时用的是
    模棱两可的信息。
    Recirculation 的做法：处理每个 token 时跑"两遍"——
      第一遍：正常前向，记录每一层输出的"残差流"向量；
      第二遍：把"深层某层"（source，如第 11 层）的输出，按一个小比例 α
              混入"浅层某层"（dest，如第 4 层）的输出，然后从 dest 层
              重新往顶层算一遍，用第二遍的结果预测下一个 token。
    这样深层已经消歧的信息就能"回到"浅层，帮助后续 token 更早看到它。

【论文公式（对应代码里的 _mix 函数）】
    z_d' = α · f(z_s) + β · z_d            （公式 1：混合）
    f(z_s) = z_s · ||z_d||₂ / ||z_s||₂    （公式 2：把源向量缩放到目标向量的长度）
    其中 z_s 是深层（source）残差，z_d 是浅层（dest）残差，α + β = 1（凸混合）。

【两个重要细节】
    1. ramping（预热）：窗口最开头的几个 token 还没有历史状态可传播，
       直接混入深层信息反而有害。所以前 10 个 token 把 α 从 0 线性升到
       目标值：α_t = min(t/10, 1) · α
    2. KV cache 覆盖：第二遍重算 dest..top 层时，这些层在这一 token 位置
       产生的 K/V（注意力缓存）应该用第二遍的结果"覆盖"第一遍的，
       因为后续 token 应该看到的是 recirculated 之后的状态。

【层索引约定】
    source / dest 使用 HuggingFace 的 0-based 层索引（第 0 层是第一个
    transformer 块）。论文表 B.1 给出的 Gemma3 1B 最优配置是
    {源层=11, 目标层=4}，与这里直接对应（论文脚注 4 说明 PyTorch/HF
    复现与论文的 JAX 实现结果一致）。
"""
import math          # 数学函数（这里只用 math.exp 计算困惑度）
import os            # 读取环境变量（HF_TOKEN）
import time          # 计时（打印每步耗时）
from dataclasses import dataclass, field   # 定义"参数包"数据类

import torch         # PyTorch 核心库
import torch.nn.functional as F            # PyTorch 函数式接口（log_softmax 等）
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# ===========================================================================
# 第一部分：模型加载
# ===========================================================================

@dataclass
class ModelBundle:
    """
    一个"打包"了模型及其相关信息的容器。

    为什么要打包？因为 recirc_logits 等函数需要用到模型的很多部件
    （层列表、词表头、配置、设备、精度……），与其每次传一堆参数，
    不如把常用的都装进一个对象里传递。

    字段说明：
      model     : 完整的因果语言模型（Gemma3ForCausalLM）
      tokenizer : 分词器（文本 <-> token id 互转）
      config    : 模型配置（层数、隐藏维度、sliding window 等）
      device    : 计算设备（cuda 或 cpu）
      dtype     : 张量精度（GPU 用 bfloat16，CPU 用 float32）
      n_layers  : transformer 块的数量（Gemma3 1B 是 26）
      hidden    : 隐藏层维度（Gemma3 1B 是 1152）
    """
    model: torch.nn.Module
    tokenizer: AutoTokenizer
    config: AutoConfig
    device: torch.device
    dtype: torch.dtype
    n_layers: int
    hidden: int

    @property
    def lm_head(self):
        """词表头（把最后的隐藏向量映射到词表大小的 logits）。

        用 @property 装饰后，可以直接写 bundle.lm_head 而不用
        bundle.lm_head()，读起来像访问属性而不是调用函数。
        """
        return self.model.lm_head


def load_model(model_name: str = "google/gemma-3-1b-pt", device=None, dtype=None,
               token=None) -> ModelBundle:
    """
    加载预训练模型并打包成 ModelBundle。

    参数：
      model_name : HuggingFace 上的模型名（默认 Gemma3 1B 预训练版）
      device     : 计算设备；None 表示自动选择（有 GPU 用 cuda，否则 cpu）
      dtype      : 张量精度；None 表示自动选择（GPU 用 bfloat16，CPU 用 float32）
      token      : HuggingFace 访问令牌；None 表示读环境变量 HF_TOKEN

    注意：google/gemma-3-1b-pt 是"受限"（gated）模型，下载权重必须带
    有效的 HF token（在 https://huggingface.co/settings/tokens 申请）。
    """
    if token is None:
        # 从环境变量读取 token；未设置则为 None（公开模型可以不要）
        token = os.environ.get("HF_TOKEN")
    if device is None:
        # torch.cuda.is_available() 为 True 说明 PyTorch 能访问 GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        # bfloat16 比 float32 省一半显存，GPU 上精度损失可接受
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    # AutoConfig：从模型仓库读取配置文件（层数、维度等超参）
    config = AutoConfig.from_pretrained(model_name, token=token)
    # AutoModelForCausalLM：按配置文件实例化因果语言模型并加载权重
    # attn_implementation="eager" 表示用朴素的注意力实现（不用 flash-attn），
    # 因为我们后面要逐层手动调用，朴素实现最可控。
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, config=config, attn_implementation="eager",
        token=token,
    )
    model = model.to(device).eval()   # 移到目标设备，并切到"评估模式"（关 dropout）
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    n_layers = config.num_hidden_layers   # transformer 块数量
    hidden = config.hidden_size           # 隐藏向量维度
    print(f"[load] {model_name}  layers={n_layers}  hidden={hidden}  "
          f"device={device}  dtype={dtype}")
    return ModelBundle(model=model, tokenizer=tokenizer, config=config,
                       device=device, dtype=dtype, n_layers=n_layers, hidden=hidden)


def get_env_fingerprint() -> dict:
    """
    采集"环境指纹"：Python / PyTorch / transformers 版本与 GPU 信息。

    作用：写进实验结果 JSON。开源复现项目里，别人拿到结果文件时能
    一眼看出"这是在什么环境下跑出来的"，方便对照复现；自己也免得
    几个月后忘了当初用哪个版本。

    返回一个字典，可直接 json.dump 序列化：
      python        : Python 版本号（如 "3.12.4"）
      torch         : PyTorch 版本（如 "2.11.0+cu130"）
      torch_cuda    : PyTorch 编译时的 CUDA 版本（如 "13.0"）
      transformers  : transformers 库版本
      cuda_available: 运行环境是否有可用 GPU（bool）
      gpu_name      : 第一块 GPU 的名字（无 GPU 时为 None）
    """
    import platform
    import sys

    fp = {
        "python": platform.python_version(),
        "torch": None,
        "torch_cuda": None,
        "transformers": None,
        "cuda_available": False,
        "gpu_name": None,
    }
    try:
        import torch
        fp["torch"] = torch.__version__
        fp["torch_cuda"] = torch.version.cuda
        fp["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            fp["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass  # torch 未安装时保留 None（不至于让指纹函数本身崩掉）
    try:
        import transformers
        fp["transformers"] = transformers.__version__
    except ImportError:
        pass
    return fp


# ===========================================================================
# 第二部分：Recirculation 顺序前向（核心）
# ===========================================================================

@dataclass
class RecircParams:
    """
    Recirculation 的超参数包。

    source : 源层索引。从这一层取残差向量"往浅层送"（论文 1B 最优 = 11）
    dest   : 目标层索引。这一层的残差向量会被"掺入"来自 source 的信息
             （论文 1B 最优 = 4）
    alpha  : 混合系数 α，控制深层信息占多大比例（论文常用 0.07 ~ 0.15）
    beta   : 混合系数 β；None 表示自动取 β = 1 - α（凸混合）
    ramp   : 预热步数。前 ramp 个 token 的 α 从 0 线性升到目标值
             （论文附录 B.3：ramp=10）；设 0 可关闭预热
    normalize : 是否把源向量缩放到目标向量的 L2 长度（论文公式 2）
    """
    source: int          # 源层索引（0-based）
    dest: int            # 目标层索引（0-based）
    alpha: float = 0.15  # 混合系数 α（论文 Table 1 配置）
    beta: float | None = None   # β，None 表示 β=1-α（1B 凸混合）
    ramp: int = 10       # 1B ramping：前 ramp 个 token 线性起调（附录 B.3）；0 表示不 ramping
    normalize: bool = True  # 源向量按 L2 范数归一化到目标层（公式 2）


def _mix(src: torch.Tensor, dst: torch.Tensor, alpha: float, beta: float,
         normalize: bool) -> torch.Tensor:
    """
    实现论文公式 (1)(2)：把深层向量 src 按比例混入浅层向量 dst。

    z_d' = α · f(z_s) + β · z_d

    为什么混之前要归一化（normalize=True 时）？
      深层向量的"长度"通常比浅层大（信息经过多层累积）。如果直接按
      α 混合，深层向量会"压过"浅层向量，把模型推离它熟悉的分布。
      公式 (2) 的做法：把 src 缩放到与 dst 相同的 L2 长度再混合——
      只借用 src 的"方向"（携带什么语义），不借用它的"大小"。

    参数形状约定：src / dst 都是 [..., hidden]（末尾是特征维），
    归一化沿着最后一维（dim=-1）做。
    """
    if normalize:
        # dst.norm(dim=-1)  : 每个向量的 L2 范数（长度），keepdim=True 保持维度便于广播
        # src.norm(dim=-1)  : 源向量的长度；+1e-12 防止除零
        # 整体效果：src × (dst长度 / src长度)，即把 src 缩放到 dst 的长度
        src = src * (dst.norm(dim=-1, keepdim=True) / (src.norm(dim=-1, keepdim=True) + 1e-12))
    # 加权混合：α 份缩放后的深层信息 + β 份原始浅层信息
    return alpha * src + beta * dst


def recirc_logits(bundle: ModelBundle, input_ids: torch.Tensor, params: RecircParams,
                  verbose: bool = False) -> torch.Tensor:
    """
    【核心函数】顺序 prefill + recirculation，返回每个位置的 logits。

    参数：
      bundle    : load_model 返回的模型包
      input_ids : 一维 token id 序列，形状 [seq]（只处理单个序列）
      params    : RecircParams 超参数包
      verbose   : True 时每 128 个 token 打印一次进度

    返回：
      logits 张量，形状 [seq, vocab]——第 t 行是"在位置 t 预测下一个
      token"的原始分数（未归一化的概率对数）。

    计算流程（对序列中每个 token t，顺序执行）：
      ┌─ 第一遍：增量前向 ──────────────────────────────┐
      │ 把 token t 喂进模型，但 attention 只计算与前面   │
      │ 已处理 token 的交互（靠 KV cache 加速）。逐层记  │
      │ 录每层输出的残差向量 first_pass_residuals[l]。   │
      └──────────────────────────────────────────────────┘
      ┌─ 第二遍：recirculation 混合 ─────────────────────┐
      │ 取深层残差 z_s（source 层）与浅层残差 z_d        │
      │ （dest 层），按公式 (1)(2) 混合得到 z_d'；        │
      │ 从 dest 层开始重算到顶层，用第二遍的结果预测。    │
      │ 同时把 dest..top 层在这一位置的 KV 覆盖为第二遍  │
      │ 的结果（后续 token 应看到 recirculated 状态）。  │
      └──────────────────────────────────────────────────┘
    """
    model, device = bundle.model, bundle.device
    n_layers, hidden = bundle.n_layers, bundle.hidden
    lm_head = bundle.lm_head
    config = bundle.config

    seq = input_ids.numel()   # 序列长度（token 个数）
    t0 = time.time()          # 起始时间（用于进度打印）

    # ------------------------------------------------------------------
    # KV cache 准备：自定义 OverwriteCache
    # ------------------------------------------------------------------
    # 背景：Transformer 自回归解码时，每个 token 的注意力要"看到"之前
    # 所有 token 的 Key/Value。把历史 K/V 存起来（KV cache），每个新
    # token 只需计算自己的 K/V 并拼接上去，避免重复计算。
    #
    # 为什么不能直接用 transformers 的 DynamicCache？
    #   recirculation 第二遍会"重算" dest..top 层。这些层在当前位置
    #   第一遍已写入过 K/V，第二遍的结果应该"覆盖"它（而不是追加）。
    #   DynamicCache 只支持追加，所以我们子类化它，加一个"覆盖模式"。
    from transformers.cache_utils import DynamicCache

    class OverwriteCache(DynamicCache):
        """支持"覆盖写入"的 KV cache。

        overwrite_from 属性：
          None              -> 普通追加模式（第一遍用）
          某个层索引 d      -> 层索引 >= d 的层改为"覆盖"模式（第二遍用）
        """

        def __init__(self, config):
            super().__init__(config=config)
            self.overwrite_from = None  # 从这个层索引起，update 改为覆盖

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            """
            重写 DynamicCache.update 以支持覆盖。

            参数（由模型层内部调用时传入）：
              key_states / value_states : 本层为当前 token 新算的 K/V
              layer_idx                 : 当前层索引
              cache_kwargs              : 附加信息（旋转位置编码等）

            覆盖的实现技巧：
              cache 里已有 [B, H, S, D]（S 个历史 token 的 K/V），
              当前位置的旧 K/V 在最后一位。把最后一位裁掉、拼上新算的
              K/V，就实现了"原位覆盖"。
            """
            if self.overwrite_from is not None and layer_idx >= self.overwrite_from:
                # ---- 覆盖模式：裁掉旧的最后一位，拼上新的 ----
                layer = self.layers[layer_idx]        # 这一层的缓存容器
                k, v = layer.keys, layer.values
                # k/v 形状 [B, H, S, D]（S 为已缓存 token 数）
                # k[..., :-1, :] 去掉最后一个 token 的 K；再与新的拼上
                layer.keys = torch.cat([k[..., :-1, :], key_states], dim=-2)
                layer.values = torch.cat([v[..., :-1, :], value_states], dim=-2)
                return layer.keys, layer.values
            # ---- 追加模式：直接调用父类的正常逻辑 ----
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

    kv_cache = OverwriteCache(config)   # 全局 KV cache（所有层共享这一个对象）
    kv_len = 0                          # 已处理的 token 数（即 cache 当前长度）

    # ------------------------------------------------------------------
    # 残差流记录容器
    # ------------------------------------------------------------------
    # first_pass_residuals[l] 保存"第一遍"第 l 层输出的残差向量，
    # 供第二遍混合使用。注意长度是 n_layers + 1：
    #   下标 0  -> embedding 输出（第 0 层之前）
    #   下标 l  -> 第 l 层 transformer 块的输出（l = 1..n_layers）
    first_pass_residuals = [None] * (n_layers + 1)

    last_hidden = None  # 第一遍最终隐藏向量（目前仅作占位，未使用）

    all_logits = []     # 收集每个 token 的 logits，最后拼成 [seq, vocab]

    with torch.no_grad():   # 全程不计算梯度（推理模式，省显存省时间）
        # --------------------------------------------------------------
        # 预计算：rotary 位置编码 与 attention mask 生成函数
        # --------------------------------------------------------------
        # Gemma3 的特殊性：它的旋转位置编码（RoPE）和因果掩码不是
        # 在层内部计算的，而是由模型顶层算好、逐层传下去。
        # 因此我们手动逐层调用时，也必须自己准备这两样东西。
        model_emb = model.model
        # 为序列中每个位置预计算 RoPE 的 cos/sin（形状 [1, seq, head_dim]）
        pos_ids = torch.arange(seq, device=device).unsqueeze(0)
        rotary_global = None
        rotary_local = None
        if hasattr(model_emb, "rotary_emb"):
            # 用一个全零的占位向量调 rotary_emb（它只依赖位置编号，
            # 不依赖内容），得到每个位置的 cos/sin 表
            probe = torch.zeros(1, seq, hidden, device=device, dtype=bundle.dtype)
            rotary_global = model_emb.rotary_emb(probe, pos_ids)       # 全局层用
            rotary_local = model_emb.rotary_emb_local(probe, pos_ids)  # 滑动窗口层用

        # Gemma3 的注意力分两种：full_attention（全局）与
        # sliding_attention（只关注最近 sliding_window=512 个 token）。
        # 两种层需要不同的掩码生成函数，按层类型分发。
        has_sliding = any(
            getattr(l, "attention_type", "") == "sliding_attention" for l in model_emb.layers
        )
        if has_sliding:
            from transformers.models.gemma3.modeling_gemma3 import (
                create_causal_mask,
                create_sliding_window_causal_mask,
            )
            # 这两个函数需要从"当时的 cache 状态"推断掩码尺寸，
            # 所以参数先留空，具体值在 make_masks 里填
            mask_kwargs = {
                "config": config,
                "input_embeds": None,
                "attention_mask": None,
                "cache_position": None,
                "past_key_values": None,
                "position_ids": None,
            }
            # 层类型 -> 掩码生成函数 的映射表
            mask_fns = {
                "full_attention": create_causal_mask,
                "sliding_attention": create_sliding_window_causal_mask,
            }
        else:
            mask_fns = None   # 没有滑动窗口的模型不需要

        def make_masks(step_x, step_cache_pos, step_pos_ids):
            """
            按"当前 KV cache 状态"生成这一 step 的 attention 掩码。

            为什么每步都要重新生成而不是一次算好？
              掩码的尺寸取决于"目前已缓存了多少 token"（kv_length），
              而这个值每处理一个 token 都会变，所以必须每步重新生成。
            """
            if mask_fns is None:
                return None
            mkw = {
                "config": config,
                "input_embeds": step_x,       # 只用来推断 batch/seq/dtype
                "attention_mask": None,       # 无 padding，不需要额外掩码
                "cache_position": step_cache_pos,
                # cache 非空时把 kv_cache 传进去，函数会据此算 kv_length
                "past_key_values": step_cache_pos.numel() > 0 and kv_cache or None,
                "position_ids": step_pos_ids,
            }
            # 对每种层类型各生成一个掩码，返回 {"full_attention": ..., "sliding_attention": ...}
            return {k: fn(**mkw) for k, fn in mask_fns.items()}

        # ==============================================================
        # 主循环：逐个 token 顺序处理
        # ==============================================================
        for t in range(seq):
            # ---- 取出当前 token，准备位置信息 ----
            # input_ids[t:t+1] 形状 [1]（一维）；unsqueeze(0) 变成 [1, 1]（二维）
            # 因为 embed_tokens 要求输入是 [batch, seq]，一维会得到 [1, hidden]
            # 而不是 [1, 1, hidden]，导致后续层维度错误（踩过的坑！）
            tok = input_ids[t:t + 1].unsqueeze(0).to(device)  # [1, 1]（embed 需要 2D）
            # pos：当前 token 在序列中的绝对位置（用于 RoPE 和位置编码）
            pos = torch.tensor([kv_len], device=device)
            # cache_pos：当前 token 写入 cache 的位置编号（与 pos 相同）
            cache_pos = torch.tensor([kv_len], device=device)

            # ---- 取当前 token 的 RoPE 切片 ----
            # 之前预计算了全序列的 cos/sin 表，这里按 kv_len 切出当前
            # 位置的那一行（形状 [1, 1, head_dim]）。RoPE 只与位置有关，
            # 与历史无关，所以可以直接从预计算表里切片。
            if rotary_global is not None:
                pe_global = (rotary_global[0][:, kv_len:kv_len + 1],
                             rotary_global[1][:, kv_len:kv_len + 1])
                pe_local = (rotary_local[0][:, kv_len:kv_len + 1],
                            rotary_local[1][:, kv_len:kv_len + 1])
            else:
                pe_global = pe_local = None

            # ---- 生成这一步的 attention 掩码 ----
            # x_probe 只是占位（掩码函数只用它的形状推断 batch/seq/dtype）
            x_probe = torch.zeros(1, 1, hidden, device=device, dtype=bundle.dtype)
            attn_mask = make_masks(x_probe, cache_pos, pos.unsqueeze(0))

            # ------------------------------------------------------------
            # 第一遍：正常前向（增量式，复用 KV cache），逐层记录残差
            # ------------------------------------------------------------
            kv_cache.overwrite_from = None  # 第一遍：追加模式
            x = model.model.embed_tokens(tok)   # token id -> 嵌入向量 [1, 1, hidden]
            first_pass_residuals[0] = x         # 记录 embedding 输出（第 0 层之前）
            for l in range(n_layers):
                layer = model.model.layers[l]   # 第 l 个 transformer 块
                # 手动调用该层（不经过模型的整体 forward），以便拿到中间层输出
                x = layer(
                    x,
                    position_embeddings_global=pe_global,   # RoPE cos/sin（全局层）
                    position_embeddings_local=pe_local,     # RoPE cos/sin（滑动层）
                    # 按该层类型取对应掩码；无掩码时传 None
                    attention_mask=attn_mask[layer.attention_type] if attn_mask is not None else None,
                    position_ids=pos,
                    past_key_values=kv_cache,   # 传入共享 KV cache（层内部自动更新）
                    use_cache=True,
                    cache_position=cache_pos,
                )
                # 层的返回是元组 (hidden_states, ...)，取第一个元素
                x = x[0] if isinstance(x, (tuple, list)) else x
                first_pass_residuals[l + 1] = x   # 记录第 l 层的残差输出
            last_hidden = x   # 第一遍的最终隐藏向量

            # 第一遍的 logits 仅作参考（baseline 用的不是它），这里注释掉
            # logits_1 = lm_head(last_hidden)

            # ------------------------------------------------------------
            # 第二遍：recirculation 混合 + 重算 dest..top 层
            # ------------------------------------------------------------
            # 取第一遍记录的深层/浅层残差
            z_s = first_pass_residuals[params.source]   # 深层（source 层）
            z_d = first_pass_residuals[params.dest]     # 浅层（dest 层）
            # ramping：前 ramp 个 token 把 α 从 0 线性升到目标值
            alpha_t = params.alpha
            if params.ramp and t < params.ramp:
                alpha_t = alpha_t * min(t / params.ramp, 1.0)
            # β 未指定时取 β = 1 - α（保证 α + β = 1，凸混合）
            beta = params.beta if params.beta is not None else 1.0 - alpha_t
            # 公式 (1)(2)：混合后的目标层输入
            z_d_new = _mix(z_s, z_d, alpha_t, beta, params.normalize)  # [1, 1, hidden]

            # 从 dest 层开始重算（第二遍）
            kv_cache.overwrite_from = params.dest  # 第二遍：覆盖模式（d..top 层）
            x = z_d_new                            # 第二遍的起点是混合后的向量
            for l in range(params.dest, n_layers):
                layer = model.model.layers[l]
                x = layer(
                    x,
                    position_embeddings_global=pe_global,
                    position_embeddings_local=pe_local,
                    attention_mask=attn_mask[layer.attention_type] if attn_mask is not None else None,
                    position_ids=pos,
                    past_key_values=kv_cache,   # 覆盖模式下，d..top 层的 KV 被原位替换
                    use_cache=True,
                    cache_position=cache_pos,
                )
                x = x[0] if isinstance(x, (tuple, list)) else x

            # ---- 由第二遍的最终隐藏向量计算 logits ----
            hidden_final = x  # [1, 1, hidden]
            # Gemma3 在顶层有一个 RMSNorm（不同模型叫法不同，这里兼容两种）
            if hasattr(model.model, "norm"):
                hidden_final = model.model.norm(hidden_final)
            elif hasattr(model.model, "final_layernorm"):
                hidden_final = model.model.final_layernorm(hidden_final)
            logits = lm_head(hidden_final)  # [1, 1, vocab]：每个词的"原始得分"
            # 部分模型（如 Gemma2/3 的某些变体）有 logit 软封顶：
            # 先把 logits 缩到 (-cap, cap) 再过 tanh，防止极端值。
            # Gemma3 1B 的 config 里 final_logit_softcapping 为 None，不执行。
            if config.final_logit_softcapping is not None:
                logits = logits / config.final_logit_softcapping
                logits = torch.tanh(logits)
                logits = logits * config.final_logit_softcapping
            all_logits.append(logits[0, 0])  # 去掉 batch/seq 两维，存 [vocab]

            kv_len += 1   # 已处理 token 数 +1

            # 【重要语义说明】为什么下一轮第一遍的 KV 是"对"的？
            #   本轮第二遍把 dest..top 层当前位置的 KV 覆盖成了
            #   recirculated 版本。下一轮 token 的第一遍在 attention 时
            #   就会看到这个覆盖后的 KV——这正是我们想要的：
            #   后续 token 应当基于"已经 recirculate 过"的历史状态继续。
            #   而 0..dest-1 层不受混合影响，其 KV 保持第一遍的即可。

            # 进度打印（verbose 模式，每 128 个 token 一次）
            if verbose and (t + 1) % 128 == 0:
                print(f"    [tok {t+1}/{seq}] kv={kv_len} elapsed={time.time()-t0:.1f}s")

    logits = torch.stack(all_logits)  # 把所有位置的 [vocab] 拼成 [seq, vocab]
    return logits


# ===========================================================================
# 第三部分：困惑度评估
# ===========================================================================

def perplexity_from_logits(logits: torch.Tensor, target_ids: torch.Tensor) -> float:
    """
    由 logits 和目标 token 计算困惑度（perplexity, ppl）。

    什么是困惑度？
      语言模型给每个位置预测"下一个词"的概率分布。困惑度 =
      exp(平均负对数似然)。直觉理解：
        - ppl = 1   完美预测（每个词都猜中，概率 1）
        - ppl = N   平均来说模型在每个位置有 N 个"候选词"拿不准
      所以困惑度越低越好。论文的 Gemma3 1B 在 PG-19 上 ppl ≈ 22。

    参数：
      logits    : 形状 [seq, vocab]，每个位置预测下一词的原始分数
      target_ids: 形状 [seq]，每个位置的"正确答案" token id
                  （注意：位置 t 的正确答案是 t+1 位置的 token）
    """
    logits = logits.float()          # 统一转 float32（bf16 上算 log_softmax 更稳）
    target = target_ids.long()       # 保证目标 id 是整数类型
    log_probs = F.log_softmax(logits, dim=-1)   # 分数 -> 对数概率（按词表维归一化）
    # gather(1, ...)：取出每个位置"正确答案"对应的对数概率
    #   log_probs: [seq, vocab]，target.unsqueeze(1): [seq, 1]
    #   结果: [seq, 1]，squeeze(1) 后是 [seq]
    nll = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
    # 平均负对数似然 -> exp -> 困惑度
    return math.exp(nll.mean().item())


def eval_baseline_ppl(bundle: ModelBundle, input_ids: torch.Tensor, verbose: bool = False) -> float:
    """
    计算"标准并行 prefill"的 baseline 困惑度。

    作用：作为对照基准（recirculation 的效果是相对它而言的），同时用于
    校准实现是否正确（如果 baseline ppl 与论文值差很远，说明实现有 bug）。

    与 recirc_logits 的区别：这里直接用模型的整体 forward（并行处理整个
    序列），不做任何 recirculation——这是论文里的"对照组"。
    """
    input_ids = input_ids.to(bundle.device)
    with torch.no_grad():
        # unsqueeze(0) 把 [seq] 变成 [1, seq]（batch 维 = 1）
        out = bundle.model(input_ids.unsqueeze(0))
    logits = out.logits[0]  # [seq, vocab]
    # 预测位置 t 的目标是 t+1 的 token；最后一位无目标，所以截掉
    # logits 的最后一行（logits[:-1]）与 input_ids 的第一个以后
    # （input_ids[1:]）对齐。
    return perplexity_from_logits(logits[:-1], input_ids[1:])


# ===========================================================================
# 命令行入口（方便快速试跑）
# ===========================================================================
# 用法示例：
#   python3 recirculation.py --text "The capital of France is" --alpha 0.15
#   python3 recirculation.py --source 11 --dest 4 --alpha 0.07
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-1b-pt")
    ap.add_argument("--source", type=int, default=11)
    ap.add_argument("--dest", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--text", default="The capital of France is")
    args = ap.parse_args()

    bundle = load_model(args.model)
    ids = bundle.tokenizer(args.text, return_tensors="pt").input_ids[0]
    print(f"tokens: {ids.numel()}  text: {args.text!r}")

    # baseline（标准前向）
    t0 = time.time()
    ppl_b = eval_baseline_ppl(bundle, ids)
    print(f"[baseline] ppl={ppl_b:.4f}  ({time.time()-t0:.2f}s)")

    # recirculation（顺序前向）
    params = RecircParams(source=args.source, dest=args.dest, alpha=args.alpha)
    t0 = time.time()
    logits = recirc_logits(bundle, ids, params, verbose=True)
    ppl_r = perplexity_from_logits(logits[:-1], ids[1:])
    print(f"[recirc ] ppl={ppl_r:.4f}  ({time.time()-t0:.2f}s)")
    print(f"[对比  ] 相对变化 {(ppl_r/ppl_b - 1)*100:+.2f}%")
