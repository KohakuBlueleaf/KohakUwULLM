"""ChatML renderers: a wrapper around a prompt renderer, and passage-grounded QA.

Both return ``(user_text, output_text)`` like every renderer; ``mask_prompt``
decides whether the prompt half is masked or folded into the target.

See internal/chat-template-design.md.
"""

import re

from kohakuwullm.registry import RENDERER, build

TURN = "<|im_start|>{role}\n{content}<|im_end|>\n"
ASSISTANT_OPEN = "<|im_start|>assistant\n"
TURN_END = "<|im_end|>"


@RENDERER.register("chatml")
class ChatMLRenderer:
    """Wrap another renderer's ``(user, output)`` in one user/assistant exchange.

    ``inner`` is a renderer spec. Give it a zero prompt-folding probability, or
    its user half arrives empty and the user turn renders blank.

    Args:
        inner: renderer spec whose output is nested inside the turns.
        mask_prompt: return the prompt separately so the packer masks it.
    """

    def __init__(self, inner="tipo", mask_prompt: bool = False) -> None:
        self.inner = build(inner, RENDERER)
        self.mask_prompt = mask_prompt

    def __call__(self, rec: dict, rng=None, **kwargs) -> tuple[str, str]:
        user, out = self.inner(rec, rng=rng, **kwargs)
        prompt = TURN.format(role="user", content=user) + ASSISTANT_OPEN
        target = out + TURN_END
        if self.mask_prompt:
            return prompt, target
        return "", prompt + target


@RENDERER.register("qa_chatml")
class QAChatMLRenderer:
    """A passage followed by ``Question:``/``Answer:`` pairs, as a grounded chat.

    The passage becomes a system turn and each pair a user/assistant exchange. A
    record with no marker falls through as a plain document.

    Args:
        field: record field holding the text.
        question / answer: the markers that delimit a pair.
        mask_prompt: mask everything but the final answer.
    """

    def __init__(
        self,
        field: str = "text",
        question: str = "Question:",
        answer: str = "Answer:",
        mask_prompt: bool = False,
    ) -> None:
        self.field = field
        self.pattern = re.compile(re.escape(question))
        self.answer = answer
        self.mask_prompt = mask_prompt

    def _split(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        chunks = self.pattern.split(text)
        pairs = []
        for chunk in chunks[1:]:
            question, sep, answer = chunk.partition(self.answer)
            if sep:
                pairs.append((question.strip(), answer.strip()))
        return chunks[0].strip(), pairs

    def __call__(self, rec: dict, rng=None, **_) -> tuple[str, str]:
        text = rec.get(self.field) or ""
        passage, pairs = self._split(text)
        if not pairs:
            return "", text

        turns = TURN.format(role="system", content=passage) if passage else ""
        for question, answer in pairs[:-1]:
            turns += TURN.format(role="user", content=question)
            turns += TURN.format(role="assistant", content=answer)
        last_q, last_a = pairs[-1]
        prompt = turns + TURN.format(role="user", content=last_q) + ASSISTANT_OPEN
        target = last_a + TURN_END
        if self.mask_prompt:
            return prompt, target
        return "", prompt + target
