"""HF config for the Kohaku decoder. Ships inside an exported repository.

Standalone by construction: an exported repo is loaded with
``trust_remote_code=True`` on machines that do not have kohakuwullm installed, so
nothing here may import it. See docs/guides/hf-export.md.
"""

from transformers.configuration_utils import PretrainedConfig


class KohakuConfig(PretrainedConfig):
    """Kohaku: GQA + per-head QK-norm, SwiGLU, and DeepSeek-style sparse MLPs.

    Layers below ``first_k_dense`` use a dense SwiGLU; the rest use one shared
    expert plus ``num_experts_per_tok`` of ``n_routed_experts``, selected on
    sigmoid scores offset by a selection-only bias and weighted by the unbiased
    score.
    """

    model_type = "kohaku"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 65536,
        hidden_size: int = 768,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 2,
        head_dim: int = 64,
        intermediate_size: int = 2048,
        moe_intermediate_size: int = 384,
        n_routed_experts: int = 64,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 8,
        first_k_dense: int = 1,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        scoring_func: str = "sigmoid",
        max_position_embeddings: int = 4096,
        rope_theta: float = 100000.0,
        rms_norm_eps: float = 1e-6,
        qk_norm: bool = True,
        tie_word_embeddings: bool = False,
        bos_token_id: int | None = 64000,
        eos_token_id: int | None = 64001,
        pad_token_id: int | None = 64002,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense = first_k_dense
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring_func = scoring_func
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.qk_norm = qk_norm
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
