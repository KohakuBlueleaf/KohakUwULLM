"""Renderers for general-pretrain text: plain documents and chat turns.

Both return ``(user_text, output_text)``; ``encode_sample`` masks the user half,
so a plain document puts everything in ``output_text`` and carries loss on every
token. See docs/internals/data.md.
"""

from kohakuwullm.registry import RENDERER

DEFAULT_ROLES = {"system": "system", "user": "user", "assistant": "assistant"}

# Assigned contiguously from 64017. See internal/chat-template-design.md section 4.
CHAT_SPECIALS = [
    "<|im_start|>",
    "<|im_end|>",
    "<|think|>",
    "<|/think|>",
    "<|tool_call|>",
    "<|/tool_call|>",
    "<|tool_result|>",
    "<|/tool_result|>",
]

CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


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


@RENDERER.register("chat")
class ChatRenderer:
    """Render a conversation, training only on assistant turns.

    Expects ``rec["messages"]`` as ``[{"role": ..., "content": ...}]``, or a
    flat ``instruction`` / ``response`` pair. Everything up to the final
    assistant turn becomes context; that turn is the target.

    Args:
        template: format string taking ``role`` and ``content``.
        generation_prefix: emitted before the assistant turn's content.
    """

    def __init__(
        self,
        template: str = "<|im_start|>{role}\n{content}<|im_end|>\n",
        generation_prefix: str = "<|im_start|>assistant\n",
    ) -> None:
        self.template = template
        self.generation_prefix = generation_prefix

    def _messages(self, rec: dict) -> list[dict]:
        messages = rec.get("messages")
        if messages:
            return messages
        prompt = rec.get("instruction") or rec.get("prompt") or ""
        reply = rec.get("response") or rec.get("output") or ""
        out = []
        if rec.get("system"):
            out.append({"role": "system", "content": rec["system"]})
        if prompt:
            out.append({"role": "user", "content": prompt})
        out.append({"role": "assistant", "content": reply})
        return out

    def __call__(self, rec: dict, rng=None, **_) -> tuple[str, str]:
        messages = self._messages(rec)
        if not messages:
            return "", ""
        last = len(messages) - 1
        while last > 0 and messages[last].get("role") != "assistant":
            last -= 1
        context = "".join(
            self.template.format(
                role=m.get("role", "user"), content=m.get("content", "")
            )
            for m in messages[:last]
        )
        target = messages[last].get("content", "")
        return context + self.generation_prefix, target + "<|im_end|>"
