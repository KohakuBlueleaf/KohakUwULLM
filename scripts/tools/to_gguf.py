"""Convert a trained checkpoint to GGUF, for llama.cpp and llama-cpp-python.

The architecture is emitted as ``dots1``, which composes exactly this model's
layers: RMSNorm, per-head Q/K norm, GQA + RoPE, SwiGLU, leading dense blocks,
routed experts with a shared expert, and a router bias for aux-loss-free
balancing. See docs/guides/gguf.md.

    kogine run scripts/tools/to_gguf.py --config configs/lm/tipo/tipo_moe_1b_150k.py \
        --set CKPT=out/ckpt/tipo-moe-1b-150k/step-2000.ckpt \
        --set OUT=out/gguf/kohaku-moe-1b-f16.gguf

``gguf`` is imported from ``ref/llama.cpp/gguf-py`` when it is not installed.
"""

import json
import os
import sys

import torch

from kohakuwullm.models.presets import get_preset

sys.path.insert(0, os.path.join("ref", "llama.cpp", "gguf-py"))

import gguf  # noqa: E402

CKPT = ""
OUT = ""
TOKENIZER = "models/tokenizer"
VOCAB_SIZE = 65536
PRESET = "Kohaku-MoE-1B"
ARCH_OVERRIDES: dict = {}
# "f32", "f16" or "bf16": the dtype every tensor is written in.
OUT_DTYPE = "f16"
# Replicate each KV head to one per query head. dots1 builds K and V at
# n_head, not n_head_kv, so a GQA model only loads once expanded.
EXPAND_KV = True
# Written as the EOT id when the tokenizer carries it.
TURN_END_TOKEN = "<|im_end|>"
# False exports a plain completion model: no chat template and no turn-stop token.
WRITE_CHAT_TEMPLATE = True

_FILE_TYPE = {
    "f32": gguf.LlamaFileType.ALL_F32,
    "f16": gguf.LlamaFileType.MOSTLY_F16,
    "bf16": gguf.LlamaFileType.MOSTLY_BF16,
}
_TORCH_DTYPE = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}


def load_state(path: str) -> dict:
    """A trainer checkpoint's whole-model tensors, under backbone names."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    state = ckpt.get("state_dict", ckpt)
    return {k.removeprefix("module."): v for k, v in state.items()}


def write_hparams(w: gguf.GGUFWriter, config, dense_ff: int) -> None:
    """Everything ``llama_model_dots1::load_arch_hparams`` reads, plus the basics."""
    w.add_context_length(config.max_position)
    w.add_embedding_length(config.dim)
    w.add_block_count(config.depth)
    w.add_feed_forward_length(dense_ff)
    w.add_head_count(config.heads)
    w.add_head_count_kv(config.heads if EXPAND_KV else config.kv_heads)
    w.add_key_length(config.head_dim)
    w.add_value_length(config.head_dim)
    w.add_rope_freq_base(config.rope_theta)
    w.add_rope_dimension_count(config.head_dim)
    w.add_layer_norm_rms_eps(config.norm_eps)
    w.add_vocab_size(config.vocab_size)
    w.add_file_type(_FILE_TYPE[OUT_DTYPE])

    w.add_expert_count(config.moe_num_experts)
    w.add_expert_used_count(config.moe_top_k)
    w.add_expert_shared_count(config.moe_num_shared)
    w.add_leading_dense_block_count(config.moe_first_dense)
    w.add_expert_feed_forward_length(moe_hidden(config))
    kwargs = config.moe_router_kwargs or {}
    w.add_expert_weights_norm(kwargs.get("norm_topk_prob", True))
    w.add_expert_weights_scale(kwargs.get("routed_scaling_factor", 1.0))
    w.add_expert_gating_func(
        {
            "sigmoid": gguf.ExpertGatingFuncType.SIGMOID,
            "softmax": gguf.ExpertGatingFuncType.SOFTMAX,
            "sqrtsoftplus": gguf.ExpertGatingFuncType.SQRTSOFTPLUS,
        }[kwargs.get("score_func", "sigmoid")]
    )


def moe_hidden(config) -> int:
    """Per-expert feed-forward width, read off a built block rather than derived."""
    from kohakuwullm.models import LMBackbone

    with torch.device("meta"):
        block = LMBackbone.build_block(config, config.depth - 1)
    return int(block.mlp.hidden)


def write_tokenizer(w: gguf.GGUFWriter, path: str) -> None:
    """The BPE vocabulary and merges, in the layout llama.cpp's gpt2 path reads."""
    with open(os.path.join(path, "tokenizer.json"), encoding="utf-8") as fh:
        spec = json.load(fh)
    with open(os.path.join(path, "tokenizer_config.json"), encoding="utf-8") as fh:
        conf = json.load(fh)

    vocab = spec["model"]["vocab"]
    added = {a["id"]: a for a in spec.get("added_tokens", [])}
    size = max(max(vocab.values()), max(added, default=0)) + 1
    tokens = [f"[UNUSED{i}]" for i in range(size)]
    types = [gguf.TokenType.UNUSED] * size
    for text, index in vocab.items():
        tokens[index] = text
        types[index] = gguf.TokenType.NORMAL
    for index, entry in added.items():
        tokens[index] = entry["content"]
        types[index] = (
            gguf.TokenType.CONTROL
            if entry.get("special")
            else gguf.TokenType.USER_DEFINED
        )

    merges = spec["model"].get("merges", [])
    merges = [" ".join(m) if isinstance(m, list) else m for m in merges]

    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("default")
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges(merges)

    ids = {t["content"]: i for i, t in added.items()}
    for name, adder in (
        ("bos_token", w.add_bos_token_id),
        ("eos_token", w.add_eos_token_id),
        ("pad_token", w.add_pad_token_id),
        ("unk_token", w.add_unk_token_id),
    ):
        text = conf.get(name)
        text = text.get("content") if isinstance(text, dict) else text
        if text in ids:
            adder(ids[text])
    # Every training document starts with BOS. See docs/guides/gguf.md.
    w.add_add_bos_token(True)
    w.add_add_eos_token(False)

    if not WRITE_CHAT_TEMPLATE:
        return
    template = conf.get("chat_template")
    if template:
        w.add_chat_template(template)
    # EOT joins EOS as end-of-generation. See docs/guides/gguf.md.
    if TURN_END_TOKEN in ids:
        w.add_eot_token_id(ids[TURN_END_TOKEN])


def expand_kv(weight: torch.Tensor, config) -> torch.Tensor:
    """A ``(kv_heads * head_dim, dim)`` projection repeated to one head per query.

    Query head ``h`` reads KV head ``h // group``, so repeat_interleave over the
    head axis reproduces GQA exactly. See docs/guides/gguf.md.
    """
    if not EXPAND_KV or config.kv_heads == config.heads:
        return weight
    group = config.heads // config.kv_heads
    rows = weight.reshape(config.kv_heads, config.head_dim, weight.shape[-1])
    return rows.repeat_interleave(group, dim=0).reshape(-1, weight.shape[-1])


def tensor_map(state: dict, config, tmap: gguf.TensorNameMap):
    """Yield ``(gguf_name, tensor)`` for every tensor llama.cpp expects.

    Fused ``w_in`` is split gate-first, matching ``GLUMLP``'s ``chunk(2, -1)``,
    and the stacked expert tensors already carry GGUF's ``(expert, out, in)``
    layout. See docs/guides/gguf.md.
    """

    def name(key: str, suffix: str = ".weight") -> str:
        got = tmap.get_name(key, try_suffixes=(".weight", ".bias"))
        if got is None:
            raise KeyError(f"no GGUF name for {key}")
        return got + suffix

    yield name("token_embd"), state["embed.weight"]
    yield name("output_norm"), state["final_norm.weight"]
    yield name("output"), state["head.weight"]

    for i in range(config.depth):
        p = f"blocks.{i}."
        yield name(f"blk.{i}.attn_norm"), state[p + "norm_attn.weight"]
        yield name(f"blk.{i}.attn_q"), state[p + "attn.q_proj.weight"]
        yield name(f"blk.{i}.attn_k"), expand_kv(
            state[p + "attn.k_proj.weight"], config
        )
        yield name(f"blk.{i}.attn_v"), expand_kv(
            state[p + "attn.v_proj.weight"], config
        )
        yield name(f"blk.{i}.attn_output"), state[p + "attn.o_proj.weight"]
        yield name(f"blk.{i}.attn_q_norm"), state[p + "attn.q_norm.weight"]
        yield name(f"blk.{i}.attn_k_norm"), state[p + "attn.k_norm.weight"]
        yield name(f"blk.{i}.ffn_norm"), state[p + "norm_mlp.weight"]

        if p + "mlp.w_in.weight" in state:
            gate, up = state[p + "mlp.w_in.weight"].chunk(2, dim=0)
            yield name(f"blk.{i}.ffn_gate"), gate
            yield name(f"blk.{i}.ffn_up"), up
            yield name(f"blk.{i}.ffn_down"), state[p + "mlp.w_out.weight"]
            continue

        gate, up = state[p + "mlp.w_in"].chunk(2, dim=1)
        yield name(f"blk.{i}.ffn_gate_exps"), gate
        yield name(f"blk.{i}.ffn_up_exps"), up
        yield name(f"blk.{i}.ffn_down_exps"), state[p + "mlp.w_out"]
        yield name(f"blk.{i}.ffn_gate_inp"), state[p + "mlp.router.weight"]

        shared_gate, shared_up = state[p + "mlp.shared.w_in.weight"].chunk(2, dim=0)
        yield name(f"blk.{i}.ffn_gate_shexp"), shared_gate
        yield name(f"blk.{i}.ffn_up_shexp"), shared_up
        yield name(f"blk.{i}.ffn_down_shexp"), state[p + "mlp.shared.w_out.weight"]

        bias = state.get(p + "mlp.router.expert_bias")
        if bias is not None:
            yield name(f"blk.{i}.exp_probs_b", ".bias"), bias


def main() -> None:
    if not CKPT or not OUT:
        raise SystemExit("set CKPT and OUT")
    config = get_preset(PRESET, vocab_size=VOCAB_SIZE, **ARCH_OVERRIDES)
    state = load_state(CKPT)
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)

    writer = gguf.GGUFWriter(OUT, gguf.MODEL_ARCH_NAMES[gguf.MODEL_ARCH.DOTS1])
    dense_ff = state["blocks.0.mlp.w_out.weight"].shape[1]
    write_hparams(writer, config, dense_ff)
    write_tokenizer(writer, TOKENIZER)

    tmap = gguf.get_tensor_name_map(gguf.MODEL_ARCH.DOTS1, config.depth)
    dtype = _TORCH_DTYPE[OUT_DTYPE]
    count = 0
    for gguf_name, tensor in tensor_map(state, config, tmap):
        keep = torch.float32 if tensor.ndim == 1 else dtype
        writer.add_tensor(gguf_name, tensor.to(keep).contiguous().numpy())
        count += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"wrote {OUT}: {count} tensors, arch=dots1, {OUT_DTYPE}")


if __name__ == "__main__":
    main()
