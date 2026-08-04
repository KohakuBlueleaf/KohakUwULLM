"""Export a trained checkpoint as a transformers repository.

Writes safetensors weights, a ``config.json`` carrying ``auto_map``, the
tokenizer, and the standalone modeling code, so the result loads with
``AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)`` on a
machine that has never seen kohakuwullm.

    kogine run scripts/tools/to_hf.py --config configs/lm/tipo/tipo_moe_1b_uwupipe.py \
        --set CKPT=out/ckpt/tipo-moe-1b-uwupipe/step-64000.ckpt \
        --set OUT=out/hf/kohaku-moe-1b

See docs/guides/hf-export.md.
"""

import json
import os
import shutil

import torch
from safetensors.torch import save_file

from kohakuwullm.models.presets import get_preset

CKPT = ""
OUT = ""
TOKENIZER = "models/tokenizer"
VOCAB_SIZE = 65536
PRESET = "Kohaku-MoE-1B"
ARCH_OVERRIDES: dict = {}
PARAM_DTYPE = ""
# Empty follows the checkpoint's own PARAM_DTYPE; set it only to override.
OUT_DTYPE = ""

_DTYPE_ALIAS = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}


def _resolve_dtype() -> str:
    """Export dtype: ``OUT_DTYPE`` if set, else the config's ``PARAM_DTYPE``."""
    name = OUT_DTYPE or PARAM_DTYPE or "bfloat16"
    return _DTYPE_ALIAS.get(name, name)


_EXPORT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..",
    "src",
    "kohakuwullm",
    "export",
    "hf",
)


def load_state(path: str) -> dict:
    """A trainer checkpoint's whole-model tensors, under backbone names."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    state = ckpt.get("state_dict", ckpt)
    return {k.removeprefix("module."): v for k, v in state.items()}


def moe_hidden(config) -> int:
    """Per-expert feed-forward width, read off a built block rather than derived."""
    from kohakuwullm.models import LMBackbone

    with torch.device("meta"):
        block = LMBackbone.build_block(config, config.depth - 1)
    return int(block.mlp.hidden)


def convert(state: dict, config) -> dict:
    """Backbone tensors under the names ``modeling_kohaku`` declares."""
    out = {
        "model.embed_tokens.weight": state["embed.weight"],
        "model.norm.weight": state["final_norm.weight"],
        "lm_head.weight": state["head.weight"],
    }
    for i in range(config.depth):
        src, dst = f"blocks.{i}.", f"model.layers.{i}."
        out[dst + "input_layernorm.weight"] = state[src + "norm_attn.weight"]
        out[dst + "post_attention_layernorm.weight"] = state[src + "norm_mlp.weight"]
        for a, b in (
            ("q_proj", "q_proj"),
            ("k_proj", "k_proj"),
            ("v_proj", "v_proj"),
            ("o_proj", "o_proj"),
        ):
            out[dst + f"self_attn.{b}.weight"] = state[src + f"attn.{a}.weight"]
        if config.qk_norm:
            out[dst + "self_attn.q_norm.weight"] = state[src + "attn.q_norm.weight"]
            out[dst + "self_attn.k_norm.weight"] = state[src + "attn.k_norm.weight"]

        if src + "mlp.w_in.weight" in state:
            gate, up = state[src + "mlp.w_in.weight"].chunk(2, dim=0)
            out[dst + "mlp.gate_proj.weight"] = gate
            out[dst + "mlp.up_proj.weight"] = up
            out[dst + "mlp.down_proj.weight"] = state[src + "mlp.w_out.weight"]
            continue

        gate, up = state[src + "mlp.w_in"].chunk(2, dim=1)
        out[dst + "mlp.gate_proj"] = gate
        out[dst + "mlp.up_proj"] = up
        out[dst + "mlp.down_proj"] = state[src + "mlp.w_out"]
        out[dst + "mlp.gate.weight"] = state[src + "mlp.router.weight"]
        bias = state.get(src + "mlp.router.expert_bias")
        out[dst + "mlp.expert_bias"] = (
            bias if bias is not None else torch.zeros(config.moe_num_experts)
        )
        s_gate, s_up = state[src + "mlp.shared.w_in.weight"].chunk(2, dim=0)
        out[dst + "mlp.shared_expert.gate_proj.weight"] = s_gate
        out[dst + "mlp.shared_expert.up_proj.weight"] = s_up
        out[dst + "mlp.shared_expert.down_proj.weight"] = state[
            src + "mlp.shared.w_out.weight"
        ]
    return out


def write_config(path: str, config) -> None:
    """``config.json`` with the ``auto_map`` that points at the shipped modules."""
    kwargs = config.moe_router_kwargs or {}
    payload = {
        "architectures": ["KohakuForCausalLM"],
        "model_type": "kohaku",
        "auto_map": {
            "AutoConfig": "configuration_kohaku.KohakuConfig",
            "AutoModel": "modeling_kohaku.KohakuModel",
            "AutoModelForCausalLM": "modeling_kohaku.KohakuForCausalLM",
        },
        "vocab_size": config.vocab_size,
        "hidden_size": config.dim,
        "num_hidden_layers": config.depth,
        "num_attention_heads": config.heads,
        "num_key_value_heads": config.kv_heads,
        "head_dim": config.head_dim,
        "intermediate_size": int(config.mlp_hidden),
        "moe_intermediate_size": moe_hidden(config),
        "n_routed_experts": config.moe_num_experts,
        "n_shared_experts": config.moe_num_shared,
        "num_experts_per_tok": config.moe_top_k,
        "first_k_dense": config.moe_first_dense,
        "norm_topk_prob": kwargs.get("norm_topk_prob", True),
        "routed_scaling_factor": kwargs.get("routed_scaling_factor", 1.0),
        "scoring_func": kwargs.get("score_func", "sigmoid"),
        "max_position_embeddings": config.max_position,
        "rope_theta": config.rope_theta,
        "rms_norm_eps": config.norm_eps,
        "qk_norm": config.qk_norm,
        "tie_word_embeddings": config.tie_embeddings,
        "dtype": _resolve_dtype(),
    }
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def main() -> None:
    if not CKPT or not OUT:
        raise SystemExit("set CKPT and OUT")
    config = get_preset(PRESET, vocab_size=VOCAB_SIZE, **ARCH_OVERRIDES)
    os.makedirs(OUT, exist_ok=True)

    tensors = convert(load_state(CKPT), config)
    dtype = getattr(torch, _resolve_dtype())
    tensors = {k: v.to(dtype).contiguous() for k, v in tensors.items()}
    save_file(
        tensors, os.path.join(OUT, "model.safetensors"), metadata={"format": "pt"}
    )

    write_config(OUT, config)
    for name in ("configuration_kohaku.py", "modeling_kohaku.py"):
        shutil.copy(os.path.join(_EXPORT_DIR, name), os.path.join(OUT, name))
    for name in os.listdir(TOKENIZER):
        shutil.copy(os.path.join(TOKENIZER, name), os.path.join(OUT, name))

    total = sum(v.numel() for v in tensors.values())
    print(
        f"wrote {OUT}: {len(tensors)} tensors, {total / 1e6:.1f}M params, {OUT_DTYPE}"
    )


if __name__ == "__main__":
    main()
