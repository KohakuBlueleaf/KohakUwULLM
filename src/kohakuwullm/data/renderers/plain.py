"""Renderer for general-pretrain text: a document, unmasked.

Returns ``(user_text, output_text)``; ``encode_sample`` masks the user half, so
a plain document puts everything in ``output_text`` and carries loss on every
token. See docs/internals/data.md.
"""

from kohakuwullm.registry import RENDERER


@RENDERER.register("plain")
class PlainRenderer:
    """Render a document as a single unmasked example.

    Args:
        field: record key holding the document text.
    """

    def __init__(self, field: str = "text") -> None:
        self.field = field

    def __call__(self, rec: dict, rng=None, **_) -> tuple[str, str]:
        return "", rec.get(self.field) or ""
