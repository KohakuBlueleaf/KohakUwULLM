"""The one tiny architecture every model test is built on.

Not a test module. Four suites instantiate the same 2-layer, 128-wide config,
and four copies of it is how two of them end up disagreeing about whether the
fixture ties its embeddings -- which decides the parameter census in
``test_models_mxfp8_swap.py`` and nothing at all in the others.
"""

from kohakuwullm import LMArchConfig


def tiny_config(**overrides):
    base = dict(
        vocab_size=512,
        dim=128,
        depth=2,
        heads=4,
        kv_heads=2,
        head_dim=32,
        mlp_hidden=256,
        tie_embeddings=True,
    )
    return LMArchConfig(**{**base, **overrides})
