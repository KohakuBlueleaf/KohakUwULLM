"""TIPO prompt rendering: a normalized record -> ``(user_text, output_text)``.

One record produces many different training examples; everything random goes
through the injected ``rng``. The task vocabulary is unchanged from the original
TIPO training. See docs/internals/data.md for the protocol and the output layout.
"""

import random

from kohakuwullm.registry import RENDERER

MAX_LEN = {"very_short": 18, "short": 36, "long": 48, "very_long": 72}
MIN_LEN = {"very_short": 0, "short": 18, "long": 36, "very_long": 48}
# (paragraphs, sentences per paragraph) budget for the natural-language fields
MAX_LEN_NL = {
    "very_short": (1, 2),
    "short": (1, 4),
    "long": (2, 4),
    "very_long": (3, 6),
}

TASKS = [
    "tag_to_long",
    "long_to_tag",
    "short_to_tag",
    "short_to_long",
    "tag_to_short_to_long",
    "short_to_tag_to_long",
    "short_to_long_to_tag",
    "gen_meta",
]

SPECIAL_TOKENS = [
    "<|empty|>",
    *(f"<|{length}|>" for length in MIN_LEN),
    *(f"<|{task}|>" for task in TASKS),
]


def _random_choose(items: list, n: int, rng) -> list:
    """``n`` items sampled without replacement, original order preserved."""
    idx = list(range(len(items)))
    rng.shuffle(idx)
    return [items[i] for i in sorted(idx[:n])]


def _trim_sentences(text: str, max_sentences: int, rng) -> str:
    """Drop sentences at random until ``text`` fits the target length bucket."""
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if len(sentences) > max_sentences:
        # Always keep the first sentence, which carries the subject.
        sentences = [sentences[0]] + _random_choose(
            sentences[1:], max_sentences - 1, rng
        )
    return ". ".join(sentences) + "."


def _trim_paragraphs(text: str, budget: tuple[int, int], rng) -> str:
    """The same trim one level up: paragraphs, then sentences within each."""
    max_paras, max_sentences = budget
    paragraphs = [
        [s.strip() for s in para.split(".") if s.strip()]
        for para in text.split("\n")
        if para.strip()
    ]
    if len(paragraphs) > max_paras:
        paragraphs = [paragraphs[0]] + _random_choose(
            paragraphs[1:], max_paras - 1, rng
        )
    for i, sentences in enumerate(paragraphs):
        if len(sentences) > max_sentences:
            paragraphs[i] = [sentences[0]] + _random_choose(
                sentences[1:], max_sentences - 1, rng
            )
    return "\n".join(". ".join(p) + "." for p in paragraphs)


@RENDERER.register("tipo")
class TIPORenderer:
    """Render a record as a TIPO training example.

    Args:
        separator: tag separator.
        train_on_input_prob: probability of folding the user half into the
            output.
        total_drop_prob: probability of dropping all prior metadata and input
            tags.
        info_drop_prob: probability of dropping a random subset of metadata.
        gen_meta_prob: probability of the ``<|gen_meta|>`` variant, where part of
            the metadata becomes a prediction target.
    """

    def __init__(
        self,
        separator: str = ", ",
        train_on_input_prob: float = 0.7,
        total_drop_prob: float = 0.10,
        info_drop_prob: float = 0.3,
        gen_meta_prob: float = 0.3,
    ) -> None:
        self.separator = separator
        self.train_on_input_prob = train_on_input_prob
        self.total_drop_prob = total_drop_prob
        self.info_drop_prob = info_drop_prob
        self.gen_meta_prob = gen_meta_prob

    def _prior_info(self, rec: dict, total_drop: bool) -> list[str]:
        """The record's metadata lines, skipping the fields it does not have."""
        sep = self.separator

        def joined(category: str) -> str | None:
            return sep.join(rec.get(category) or []) or None

        aspect = None
        if rec.get("width") and rec.get("height"):
            aspect = f"aspect ratio: {rec['width'] / rec['height']:.1f}"
        fields = [
            (f"meta: {joined('meta')}", joined("meta")),
            (f"rating: {joined('rating')}", joined("rating")),
            (f"artist: {joined('artist')}", joined("artist")),
            (aspect, None if total_drop else aspect),
            (f"characters: {joined('character')}", joined("character")),
            (f"quality: {rec.get('quality')}", rec.get("quality")),
            (f"copyrights: {joined('copyright')}", joined("copyright")),
        ]
        return [text for text, present in fields if present]

    def __call__(
        self, rec: dict, rng=None, target_len: str | None = None
    ) -> tuple[str, str]:
        """Draw one training example from ``rec``: ``(user_text, output_text)``.

        Every choice below is a draw from ``rng``. See docs/internals/data.md.
        """
        rng = rng or random
        sep = self.separator

        general = list(rec.get("general") or [])
        total = len(general)
        rng.shuffle(general)

        if target_len is None:
            candidates = [k for k, low in MIN_LEN.items() if low <= total]
            target_len = candidates[-1] if candidates else "very_short"
        nl_budget = MAX_LEN_NL[target_len]

        length = min(MAX_LEN[target_len], total)
        n_input = rng.randint(1, max(length * 3 // 5, 1) + 1)
        total_drop = rng.random() < self.total_drop_prob
        if total_drop:
            n_input = 0
        prompt_input = general[:n_input]
        prompt_output = general[n_input:length]

        drop_info = not total_drop and rng.random() < self.info_drop_prob
        prior_info = self._prior_info(rec, total_drop)
        if drop_info:
            prior_info = [info for info in prior_info if rng.random() > 0.35]
        rng.shuffle(prior_info)

        short, long = rec.get("short"), rec.get("long")
        if short:
            short = _trim_sentences(short, nl_budget[1], rng)
        if long:
            long = _trim_paragraphs(long, nl_budget, rng)

        task, task_str = self._pick_task(short, long, target_len, rng)
        full = {"tag": sep.join(general), "short": short, "long": long}

        before, addon_out = "", ""
        no_drop = not drop_info and not total_drop
        if no_drop and prior_info and rng.random() < self.gen_meta_prob:
            task_str += " <|gen_meta|>"
            n_given = int(
                min(max(1, rng.random() * len(prior_info)), max(1, len(prior_info) - 1))
            )
            before = "\n".join(prior_info[:n_given]) + "\n"
            addon_out = "\n" + "\n".join(prior_info[n_given:])
        elif not total_drop and prior_info:
            before = "\n".join(prior_info) + "\n"

        after, out = "", ""
        order = task.split("_to_") if task else ["tag"]
        if order[0] == "tag":
            after += f"tag: {sep.join(prompt_input)}"
            out += sep + sep.join(prompt_output) + "\n"
        else:
            after += f"{order[0]}: {full[order[0]]}\n"
        for field in order[1:]:
            out += f"{field}: {full[field]}\n"

        user = before + f"target: {task_str}\n" + after
        out = (out.rstrip() + addon_out).rstrip()
        if rng.random() < self.train_on_input_prob:
            user, out = "", user + out
        return user, out

    def _pick_task(self, short, long, target_len, rng):
        """Choose a task the record has the fields for, and its target line.

        Returns ``(None, ...)`` -- a plain length-conditioned continuation --
        with probability ``1 / (len(tasks) + 1)``.
        """
        tasks = []
        if long:
            tasks += ["tag_to_long", "long_to_tag"]
        if short:
            tasks += ["short_to_tag"]
        if long and short:
            # Duplicated so the multi-hop tasks are drawn more often.
            tasks += [
                "tag_to_short_to_long",
                "short_to_long_to_tag",
                "short_to_tag_to_long",
            ] * 2

        if tasks and rng.random() < (len(tasks) / (len(tasks) + 1)):
            task = rng.choice(tasks)
            needs_length = task.startswith("tag_to") or "to_tag" in task
            return task, (
                f"<|{target_len}|> <|{task}|>" if needs_length else f"<|{task}|>"
            )
        return None, f"<|{target_len}|>"
