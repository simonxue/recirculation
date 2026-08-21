#!/usr/bin/env python3
"""
Recirculation 核心前向实现（arXiv:2608.17981）

原理（论文公式 1/2）：
  对每个输入 token 顺序执行两遍前向：
    第一遍：正常前向（复用已 recirculate 的 KV cache），记录源层 s 与目标层 d 的残差流输出
    第二遍：把源层残差按 L2 范数归一化后按 α 混入目标层，重算 d..top 层，取第二遍 logits，
            并将第二遍的 K/V 存入 cache 供后续 token 使用

    z_d' = α · f(z_s) + β · z_d,   f(z_s) = z_s · ||z_d||₂ / ||z_s||₂,  β = 1-α (1B 凸混合)

  1B 专属 ramping（附录 B.3）：前 10 个 token 线性起调 α_t = min(t/10, 1)·α

层索引约定：source/destination 使用 HF 0-based 层索引（与论文表 B.1 的 JAX 索引一致，
论文脚注 4 说明 PyTorch/HF 复现与 JAX 结果一致，两者直接对应）。
"""
import math
import os
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------

@dataclass
class ModelBundle:
    model: torch.nn.Module
    tokenizer: AutoTokenizer
    config: AutoConfig
    device: torch.device
    dtype: torch.dtype
    n_layers: int
    hidden: int

    @property
    def lm_head(self):
        return self.model.lm_head


def load_model(model_name: str = "google/gemma-3-1b-pt", device=None, dtype=None,
               token=None) -> ModelBundle:
    """加载 Gemma3 PT 模型（默认 GPU 用 bf16，CPU 用 float32）。

    token: HF token（gated 模型必需）。默认读环境变量 HF_TOKEN；
           若仍为空则回退到已知 token（用户提供的 simonxue21 授权 token）。
    """
    if token is None:
        token = os.environ.get("HF_TOKEN")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    config = AutoConfig.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, config=config, attn_implementation="eager",
        token=token,
    )
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    n_layers = config.num_hidden_layers
    hidden = config.hidden_size
    print(f"[load] {model_name}  layers={n_layers}  hidden={hidden}  "
          f"device={device}  dtype={dtype}")
    return ModelBundle(model=model, tokenizer=tokenizer, config=config,
                       device=device, dtype=dtype, n_layers=n_layers, hidden=hidden)


# ---------------------------------------------------------------------------
# Recirculation 顺序前向
# ---------------------------------------------------------------------------

@dataclass
class RecircParams:
    source: int          # 源层索引 (0-based)
    dest: int            # 目标层索引 (0-based)
    alpha: float = 0.15  # 混合系数 α（论文 Table 1 配置）
    beta: float | None = None   # β，None 表示 β=1-α（1B 凸混合）
    ramp: int = 10       # 1B ramping：前 ramp 个 token 线性起调（附录 B.3）；0 表示不 ramping
    normalize: bool = True  # 源向量按 L2 范数归一化到目标层（公式 2）


def _mix(src: torch.Tensor, dst: torch.Tensor, alpha: float, beta: float,
         normalize: bool) -> torch.Tensor:
    """公式 (1)(2)：z_d' = α·f(z_s) + β·z_d，f 为范数匹配归一化。"""
    if normalize:
        src = src * (dst.norm(dim=-1, keepdim=True) / (src.norm(dim=-1, keepdim=True) + 1e-12))
    return alpha * src + beta * dst


def recirc_logits(bundle: ModelBundle, input_ids: torch.Tensor, params: RecircParams,
                  verbose: bool = False) -> torch.Tensor:
    """
    顺序 prefill + recirculation，返回每个位置的 logits（形状 [seq, vocab]）。
    input_ids: [seq] 1D，短序列专用。
    """
    model, device = bundle.model, bundle.device
    n_layers, hidden = bundle.n_layers, bundle.hidden
    lm_head = bundle.lm_head
    config = bundle.config

    seq = input_ids.numel()
    t0 = time.time()

    # KV cache：包装 DynamicCache，支持"覆盖重算层"（recirculation 第二遍会重算 d..top 层，
    # 这些层在当前 token 位置的 KV 应被第二遍结果覆盖，而非追加）
    from transformers.cache_utils import DynamicCache

    class OverwriteCache(DynamicCache):
        def __init__(self, config):
            super().__init__(config=config)
            self.overwrite_from = None  # 从这个层索引起，update 改为覆盖

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            if self.overwrite_from is not None and layer_idx >= self.overwrite_from:
                # 覆盖：先裁掉当前 token 的旧 KV（最后一位），再追加新 KV
                layer = self.layers[layer_idx]
                k, v = layer.keys, layer.values
                # k/v 形状 [B, H, S, D]（S 为已缓存 token 数）
                layer.keys = torch.cat([k[..., :-1, :], key_states], dim=-2)
                layer.values = torch.cat([v[..., :-1, :], value_states], dim=-2)
                return layer.keys, layer.values
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

    kv_cache = OverwriteCache(config)
    kv_len = 0

    # 缓存第一遍的层残差（当前 token），供第二遍混合用
    # 结构: list[n_layers+1] of [hidden]（z[0]=embedding, z[l]=layer l 输出）
    first_pass_residuals = [None] * (n_layers + 1)

    last_hidden = None  # 第一遍 final hidden state

    all_logits = []

    with torch.no_grad():
        # Gemma3 的 rotary 位置嵌入与 causal mask 由模型外部计算并传给每层
        model_emb = model.model
        pos_ids = torch.arange(seq, device=device).unsqueeze(0)
        rotary_global = None
        rotary_local = None
        if hasattr(model_emb, "rotary_emb"):
            probe = torch.zeros(1, seq, hidden, device=device, dtype=bundle.dtype)
            rotary_global = model_emb.rotary_emb(probe, pos_ids)
            rotary_local = model_emb.rotary_emb_local(probe, pos_ids)

        has_sliding = any(
            getattr(l, "attention_type", "") == "sliding_attention" for l in model_emb.layers
        )
        if has_sliding:
            from transformers.models.gemma3.modeling_gemma3 import (
                create_causal_mask,
                create_sliding_window_causal_mask,
            )
            mask_kwargs = {
                "config": config,
                "input_embeds": None,
                "attention_mask": None,
                "cache_position": None,
                "past_key_values": None,
                "position_ids": None,
            }
            mask_fns = {
                "full_attention": create_causal_mask,
                "sliding_attention": create_sliding_window_causal_mask,
            }
        else:
            mask_fns = None

        def make_masks(step_x, step_cache_pos, step_pos_ids):
            """按当前 cache 状态生成该步的 attention mask（各层类型对应）"""
            if mask_fns is None:
                return None
            mkw = {
                "config": config,
                "input_embeds": step_x,
                "attention_mask": None,
                "cache_position": step_cache_pos,
                "past_key_values": step_cache_pos.numel() > 0 and kv_cache or None,
                "position_ids": step_pos_ids,
            }
            return {k: fn(**mkw) for k, fn in mask_fns.items()}

        for t in range(seq):
            tok = input_ids[t:t + 1].unsqueeze(0).to(device)  # [1, 1]（embed 需要 2D）
            pos = torch.tensor([kv_len], device=device)
            cache_pos = torch.tensor([kv_len], device=device)

            # 该 token 对应的 rotary 切片（每个位置独立，与 cache 无关）
            if rotary_global is not None:
                pe_global = (rotary_global[0][:, kv_len:kv_len + 1],
                             rotary_global[1][:, kv_len:kv_len + 1])
                pe_local = (rotary_local[0][:, kv_len:kv_len + 1],
                            rotary_local[1][:, kv_len:kv_len + 1])
            else:
                pe_global = pe_local = None

            # attention mask：按当前 cache 状态生成（sliding 窗口长度由 cache 决定）
            x_probe = torch.zeros(1, 1, hidden, device=device, dtype=bundle.dtype)
            attn_mask = make_masks(x_probe, cache_pos, pos.unsqueeze(0))

            # ---- 第一遍：正常前向（复用 KV cache），逐层记录残差 ----
            kv_cache.overwrite_from = None  # 第一遍：追加模式
            x = model.model.embed_tokens(tok)
            first_pass_residuals[0] = x
            for l in range(n_layers):
                layer = model.model.layers[l]
                x = layer(
                    x,
                    position_embeddings_global=pe_global,
                    position_embeddings_local=pe_local,
                    attention_mask=attn_mask[layer.attention_type] if attn_mask is not None else None,
                    position_ids=pos,
                    past_key_values=kv_cache,
                    use_cache=True,
                    cache_position=cache_pos,
                )
                x = x[0] if isinstance(x, (tuple, list)) else x
                first_pass_residuals[l + 1] = x
            last_hidden = x

            # 第一遍 logits（baseline 参考，非输出）
            # logits_1 = lm_head(last_hidden)

            # ---- 第二遍：recirculation 混合后重算 d..top ----
            # 目标层输入 = α·f(z_s) + β·z_d
            z_s = first_pass_residuals[params.source]
            z_d = first_pass_residuals[params.dest]
            alpha_t = params.alpha
            if params.ramp and t < params.ramp:
                alpha_t = alpha_t * min(t / params.ramp, 1.0)
            beta = params.beta if params.beta is not None else 1.0 - alpha_t
            z_d_new = _mix(z_s, z_d, alpha_t, beta, params.normalize)  # [1, 1, hidden]

            # 重算 dest..top 层（第二遍）
            kv_cache.overwrite_from = params.dest  # 第二遍：覆盖模式（d..top 层）
            x = z_d_new
            for l in range(params.dest, n_layers):
                layer = model.model.layers[l]
                x = layer(
                    x,
                    position_embeddings_global=pe_global,
                    position_embeddings_local=pe_local,
                    attention_mask=attn_mask[layer.attention_type] if attn_mask is not None else None,
                    position_ids=pos,
                    past_key_values=kv_cache,
                    use_cache=True,
                    cache_position=cache_pos,
                )
                x = x[0] if isinstance(x, (tuple, list)) else x

            hidden_final = x  # [1, 1, hidden]
            if hasattr(model.model, "norm"):
                hidden_final = model.model.norm(hidden_final)
            elif hasattr(model.model, "final_layernorm"):
                hidden_final = model.model.final_layernorm(hidden_final)
            logits = lm_head(hidden_final)  # [1, 1, vocab]
            if config.final_logit_softcapping is not None:
                logits = logits / config.final_logit_softcapping
                logits = torch.tanh(logits)
                logits = logits * config.final_logit_softcapping
            all_logits.append(logits[0, 0])  # [vocab]

            kv_len += 1

            # 下一 token 的第一遍需要读取本轮第一遍的中间层残差？否——
            # 第一遍残差仅用于本轮混合；下轮第一遍会用 cache 正常前向
            # 但注意：本轮第二遍覆盖了 cache 中 d..top 层的 K/V（正确，
            # 因为后续 token 的 attention 应看到 recirculated 状态）

            if verbose and (t + 1) % 128 == 0:
                print(f"    [tok {t+1}/{seq}] kv={kv_len} elapsed={time.time()-t0:.1f}s")

    logits = torch.stack(all_logits)  # [seq, vocab]
    return logits


# ---------------------------------------------------------------------------
# 困惑度评估
# ---------------------------------------------------------------------------

def perplexity_from_logits(logits: torch.Tensor, target_ids: torch.Tensor) -> float:
    """给定 logits [seq, vocab] 与目标 token [seq]，返回困惑度。"""
    logits = logits.float()
    target = target_ids.long()
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
    return math.exp(nll.mean().item())


def eval_baseline_ppl(bundle: ModelBundle, input_ids: torch.Tensor, verbose: bool = False) -> float:
    """标准并行 prefill 的 baseline 困惑度（用于校准实现）。"""
    input_ids = input_ids.to(bundle.device)
    with torch.no_grad():
        out = bundle.model(input_ids.unsqueeze(0))
    logits = out.logits[0]  # [seq, vocab]
    # 预测位置 t 的目标是 t+1 的 token；最后一位无目标
    return perplexity_from_logits(logits[:-1], input_ids[1:])


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

    # baseline
    t0 = time.time()
    ppl_b = eval_baseline_ppl(bundle, ids)
    print(f"[baseline] ppl={ppl_b:.4f}  ({time.time()-t0:.2f}s)")

    # recirculation
    params = RecircParams(source=args.source, dest=args.dest, alpha=args.alpha)
    t0 = time.time()
    logits = recirc_logits(bundle, ids, params, verbose=True)
    ppl_r = perplexity_from_logits(logits[:-1], ids[1:])
    print(f"[recirc ] ppl={ppl_r:.4f}  ({time.time()-t0:.2f}s)")
    print(f"[对比  ] 相对变化 {(ppl_r/ppl_b - 1)*100:+.2f}%")
