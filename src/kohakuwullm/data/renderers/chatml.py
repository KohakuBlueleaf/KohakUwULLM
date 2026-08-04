"""ChatML renderers: conversations, passage-grounded QA, and a prompt wrapper.

Each returns ``[(text, is_target), ...]``: one masked segment per context turn,
and a header/content split on every trained turn.
See internal/chat-template-design.md.
"""

import re

from kohakuwullm.registry import RENDERER, build

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

TURN_OPEN = "<|im_start|>{role}\n"
TURN_END = "<|im_end|>"
TURN_CLOSE = TURN_END + "\n"
ASSISTANT_OPEN = "<|im_start|>assistant\n"
TRAIN_ROLES = ("assistant",)


def chat_segments(
    messages: list[dict],
    train_roles: tuple[str, ...] = TRAIN_ROLES,
    mask_context: bool = True,
) -> list[tuple[str, bool]]:
    """``[{role, content}]`` -> ChatML segments, one or two per turn.

    A trained turn splits after its ``<|im_start|>{role}\\n`` header, which stays
    context; the content and its ``<|im_end|>`` are the target. Every other turn
    is one masked segment. ``mask_context`` off makes the whole rendering a
    target.
    """
    segments: list[tuple[str, bool]] = []
    for message in messages:
        role = message.get("role") or "user"
        header = TURN_OPEN.format(role=role)
        body = (message.get("content") or "") + TURN_CLOSE
        if not mask_context:
            segments.append((header + body, True))
        elif role in train_roles:
            segments.append((header, False))
            segments.append((body, True))
        else:
            segments.append((header + body, False))
    return segments


@RENDERER.register("chat")
class ChatRenderer:
    """Render a conversation as ChatML, training on every turn of a trained role.

    Reads ``rec[field]`` as ``[{"role": ..., "content": ...}]``, or a flat
    ``system`` / ``instruction`` / ``response`` triple.

    Args:
        field: record key holding the message list.
        train_roles: roles whose content carries loss.
        mask_context: mask every other role, and every turn header.
        min_turns: render nothing below this many messages.
    """

    def __init__(
        self,
        field: str = "messages",
        train_roles: tuple[str, ...] = TRAIN_ROLES,
        mask_context: bool = True,
        min_turns: int = 2,
    ) -> None:
        self.field = field
        self.train_roles = tuple(train_roles)
        self.mask_context = mask_context
        self.min_turns = min_turns

    def _messages(self, rec: dict) -> list[dict]:
        messages = rec.get(self.field)
        if messages:
            stripped = (
                {
                    "role": m.get("role") or "user",
                    "content": (m.get("content") or "").strip(),
                }
                for m in messages
            )
            return [m for m in stripped if m["content"]]
        prompt = rec.get("instruction") or rec.get("prompt") or ""
        reply = rec.get("response") or rec.get("output") or ""
        out = []
        if rec.get("system"):
            out.append({"role": "system", "content": rec["system"]})
        if prompt:
            out.append({"role": "user", "content": prompt})
        if reply:
            out.append({"role": "assistant", "content": reply})
        return out

    def __call__(self, rec: dict, rng=None, **_) -> list[tuple[str, bool]]:
        messages = self._messages(rec)
        if len(messages) < self.min_turns:
            return []
        return chat_segments(messages, self.train_roles, self.mask_context)


@RENDERER.register("chatml")
class ChatMLRenderer:
    """Wrap another renderer's ``(user, output)`` in one user/assistant exchange.

    ``inner`` is a renderer spec. Give it a zero prompt-folding probability, or
    its user half arrives empty and the user turn renders blank.

    Args:
        inner: renderer spec whose output is nested inside the turns.
        mask_context: mask the user turn and both headers.
    """

    def __init__(self, inner="tipo", mask_context: bool = True) -> None:
        self.inner = build(inner, RENDERER)
        self.mask_context = mask_context

    def __call__(self, rec: dict, rng=None, **kwargs) -> list[tuple[str, bool]]:
        user, out = self.inner(rec, rng=rng, **kwargs)
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": out},
        ]
        return chat_segments(messages, TRAIN_ROLES, self.mask_context)


@RENDERER.register("qa_chatml")
class QAChatMLRenderer:
    """A passage followed by ``Question:``/``Answer:`` pairs, as a grounded chat.

    The passage becomes a system turn and each pair a user/assistant exchange. A
    record with no marker falls through as one unmasked plain document.

    Args:
        field: record field holding the text.
        question / answer: the markers that delimit a pair.
        mask_context: mask the passage, the questions and the turn headers.
    """

    def __init__(
        self,
        field: str = "text",
        question: str = "Question:",
        answer: str = "Answer:",
        mask_context: bool = True,
    ) -> None:
        self.field = field
        self.pattern = re.compile(re.escape(question))
        self.answer = answer
        self.mask_context = mask_context

    def _split(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        chunks = self.pattern.split(text)
        pairs = []
        for chunk in chunks[1:]:
            question, sep, answer = chunk.partition(self.answer)
            if sep:
                pairs.append((question.strip(), answer.strip()))
        return chunks[0].strip(), pairs

    def __call__(self, rec: dict, rng=None, **_) -> list[tuple[str, bool]]:
        text = rec.get(self.field) or ""
        passage, pairs = self._split(text)
        if not pairs:
            return [(text, True)]

        messages = [{"role": "system", "content": passage}] if passage else []
        for question, answer in pairs:
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
        return chat_segments(messages, TRAIN_ROLES, self.mask_context)
