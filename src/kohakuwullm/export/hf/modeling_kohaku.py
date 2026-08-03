"""HF modeling code for the Kohaku decoder. Ships inside an exported repository.

Standalone by construction: loaded with ``trust_remote_code=True`` on machines
without kohakuwullm, so nothing here may import it. Routed experts stay stacked
as ``(E, out, in)``, the layout training uses, so an export is a copy rather than
a reshape. See docs/guides/hf-export.md.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel

from .configuration_kohaku import KohakuConfig


class KohakuRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.normalized_shape = (dim,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(q, k, cos, sin):
    """``cos``/``sin`` are ``(B, S, head_dim)``; q/k are ``(B, S, H, head_dim)``."""
    cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class KohakuRotary(nn.Module):
    def __init__(self, config: KohakuConfig) -> None:
        super().__init__()
        inv = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, config.head_dim, 2, dtype=torch.int64).float()
                / config.head_dim
            )
        )
        self.register_buffer("inv_freq", inv, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        freqs = position_ids[:, :, None].float() * self.inv_freq[None, None, :]
        angles = torch.cat((freqs, freqs), dim=-1)
        return angles.cos().to(x.dtype), angles.sin().to(x.dtype)


class KohakuAttention(nn.Module):
    """GQA with per-head QK-norm applied before RoPE."""

    def __init__(self, config: KohakuConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        q_out = self.heads * self.head_dim
        kv_out = self.kv_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, q_out, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_out, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_out, bias=False)
        self.o_proj = nn.Linear(q_out, config.hidden_size, bias=False)
        if config.qk_norm:
            self.q_norm = KohakuRMSNorm(self.head_dim, config.rms_norm_eps)
            self.k_norm = KohakuRMSNorm(self.head_dim, config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x, cos, sin, attention_mask=None, past_key_values=None, **kwargs):
        b, s, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(b, s, self.heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).view(b, s, self.kv_heads, self.head_dim))
        v = self.v_proj(x).view(b, s, self.kv_heads, self.head_dim)
        q, k = apply_rope(q, k, cos, sin)

        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            scale=self.scale,
            is_causal=attention_mask is None and s > 1,
            enable_gqa=self.kv_heads != self.heads,
        )
        return self.o_proj(out.transpose(1, 2).reshape(b, s, -1))


class KohakuMLP(nn.Module):
    """SwiGLU. ``gate_proj``/``up_proj`` are the two halves of the trained ``w_in``."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class KohakuMoE(nn.Module):
    """One shared expert plus top-k of ``n_routed_experts``, experts kept stacked.

    Selection adds ``expert_bias`` to the sigmoid scores; the weights come from
    the unbiased scores, which is what makes the balancer auxiliary-loss-free.
    """

    def __init__(self, config: KohakuConfig) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        e, d, h = (
            config.n_routed_experts,
            config.hidden_size,
            config.moe_intermediate_size,
        )
        self.gate = nn.Linear(d, e, bias=False)
        self.register_buffer("expert_bias", torch.zeros(e), persistent=True)
        self.gate_proj = nn.Parameter(torch.empty(e, h, d))
        self.up_proj = nn.Parameter(torch.empty(e, h, d))
        self.down_proj = nn.Parameter(torch.empty(e, d, h))
        self.shared_expert = KohakuMLP(d, h * config.n_shared_experts)

    def score(self, logits: torch.Tensor) -> torch.Tensor:
        if self.scoring_func == "softmax":
            return logits.softmax(-1)
        if self.scoring_func == "sqrtsoftplus":
            return F.softplus(logits).sqrt()
        return logits.sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        # Router runs wholly in fp32, matching training.
        scores = self.score(F.linear(flat.float(), self.gate.weight.float()))
        index = (scores + self.expert_bias).topk(self.top_k, dim=-1).indices
        weight = scores.gather(1, index)
        if self.norm_topk_prob and self.top_k > 1:
            weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-9)
        weight = (weight * self.routed_scaling_factor).to(x.dtype)

        out = torch.zeros_like(flat)
        for expert in index.unique():
            rows, slot = (index == expert).nonzero(as_tuple=True)
            taken = flat[rows]
            hidden = F.silu(taken @ self.gate_proj[expert].T) * (
                taken @ self.up_proj[expert].T
            )
            out.index_add_(
                0, rows, (hidden @ self.down_proj[expert].T) * weight[rows, slot, None]
            )
        return (out + self.shared_expert(flat)).view(shape)


class KohakuDecoderLayer(nn.Module):
    def __init__(self, config: KohakuConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = KohakuAttention(config, layer_idx)
        self.mlp = (
            KohakuMLP(config.hidden_size, config.intermediate_size)
            if layer_idx < config.first_k_dense
            else KohakuMoE(config)
        )
        self.input_layernorm = KohakuRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = KohakuRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(self, x, cos, sin, attention_mask=None, past_key_values=None, **kwargs):
        x = x + self.self_attn(
            self.input_layernorm(x), cos, sin, attention_mask, past_key_values, **kwargs
        )
        return x + self.mlp(self.post_attention_layernorm(x))


class KohakuPreTrainedModel(PreTrainedModel):
    config_class = KohakuConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["KohakuDecoderLayer"]
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module) -> None:
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, KohakuRMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, KohakuMoE):
            for p in (module.gate_proj, module.up_proj, module.down_proj):
                p.data.normal_(mean=0.0, std=std)


class KohakuModel(KohakuPreTrainedModel):
    def __init__(self, config: KohakuConfig) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            KohakuDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        )
        self.norm = KohakuRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = KohakuRotary(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        seen = (
            past_key_values.get_seq_length()
            if isinstance(past_key_values, Cache)
            else 0
        )
        if position_ids is None:
            length = inputs_embeds.shape[1]
            position_ids = torch.arange(
                seen, seen + length, device=inputs_embeds.device
            ).unsqueeze(0)

        mask = None
        if attention_mask is not None and attention_mask.dim() == 2:
            total = seen + inputs_embeds.shape[1]
            pad = attention_mask[:, None, None, :total].bool()
            queries = torch.arange(seen, total, device=inputs_embeds.device)[:, None]
            keys = torch.arange(total, device=inputs_embeds.device)[None, :]
            mask = pad & (keys <= queries)[None, None]
        elif attention_mask is not None:
            mask = attention_mask

        cos, sin = self.rotary_emb(inputs_embeds, position_ids)
        hidden = inputs_embeds
        for layer in self.layers:
            hidden = layer(hidden, cos, sin, mask, past_key_values, **kwargs)
        return BaseModelOutputWithPast(
            last_hidden_state=self.norm(hidden), past_key_values=past_key_values
        )


class KohakuForCausalLM(KohakuPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: KohakuConfig) -> None:
        super().__init__(config)
        self.model = KohakuModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value) -> None:
        self.lm_head = value

    def forward(self, input_ids=None, labels=None, **kwargs):
        out = self.model(input_ids=input_ids, **kwargs)
        logits = self.lm_head(out.last_hidden_state)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=out.past_key_values
        )


__all__ = ["KohakuConfig", "KohakuModel", "KohakuForCausalLM", "KohakuPreTrainedModel"]
