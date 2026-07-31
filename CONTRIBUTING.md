# Contributing to KohakUwULLM

Short on purpose; read it end-to-end before opening a PR.

**English only** for code, comments, commits, and PR text. Translated docs
(`docs/zh/`, `README.zh.md`) are the exception and are welcome.

## Before you open a PR

Open an issue or ping the community channels first. This is not gatekeeping --
it stops you from writing something that collides with in-flight work.

- **QQ group**: 1097666427
- **Discord**: https://discord.gg/xWYrkyvJ2s

**Feature PRs require prior approval.** A feature PR adds functionality, changes
a public API, alters core architecture, or changes user-visible behavior
non-trivially. It needs a public discussion trail and an explicit maintainer
go-ahead *before* the PR is opened. Submitting an issue and a PR at the same time
does not count. PRs without traceable approval get closed with a pointer here,
and can be reopened immediately once alignment happens.

**Bug fixes, docs, tests and small single-module refactors** can go straight to
PR. Still worth a two-line message first.

## Local pre-flight

```bash
uv pip install -e ".[dev,bench]"

ruff check src/ scripts/ tests/
black --check src/ scripts/ tests/
.venv/bin/python -m pytest tests/ -q
```

If you touched a kernel, also run its benchmark and paste the before/after into
the PR:

```bash
.venv/bin/python scripts/bench/kernel/kernels.py --out out/bench/kernels
```

## Code conventions

The full set lives in [CLAUDE.md](CLAUDE.md). The non-negotiables:

- **Python 3.10+.** Modern hints: `list`, `dict`, `X | None`. Never `List`,
  `Optional`, `Union` from `typing`.
- **No imports inside functions**, except optional deps and lazy imports that
  avoid slow startup (Lightning in `training/__init__.py` is the example).
- **Import grouping**: built-in, third-party, `kohakuwullm.*`; `import` before
  `from`; shorter dotted paths first; alphabetical.
- **Max 600 lines per file** (hard cap 1000).
- **Comments explain *why*, not *what*.** Where a decision was made against a
  plausible alternative, say which alternative and why it lost -- that is the only
  kind of comment that survives the code changing.
- **Prefer `match-case`** over deep `if-elif-else`.
- Never use `sys.path` hacks; import from the installed package.

## Rules specific to this repo

**Select, don't dispatch.** Config resolves a concrete class once, at build time,
via `build(spec, REGISTRY)`. If you are adding an `if mode == ...` inside a
training or sampling loop, it belongs in `__init__` instead.

**Every Triton kernel needs a precision test and a benchmark row.** The test goes
in the matching `tests/test_kernels*.py` (they are split by subject), compares
against an fp64 reference in *both* fp16 and bf16, forward and backward, and
states tolerance in ULP. If the kernel has a CPU fallback, that test goes in
`tests/test_kernels_cpu_fallback.py`, the one file with no `requires CUDA` mark.
The benchmark goes in `scripts/bench/kernel/kernels.py` against whatever the
kernel replaces. A kernel that does not beat its baseline should be deleted, not
kept.

**Never trust a low-precision scalar reduction.** Summing 16k bf16 terms loses
several percent. Reduce in fp32.

**Attention must not cross document boundaries** in the packed layout. If your
backend cannot take `cu_seqlens`, build the block-diagonal mask. There is a test
that pins this; do not weaken it.

## Audit loop (anything larger than one file)

Do not stop at "tests pass". Loop until it converges: implement → write tests
that pin the behaviour (negative cases count more) → run the suite and lint →
audit the diff for clear bugs, integrity bugs and behavior bugs → if a bug
slipped past the tests, fix the *test* first, confirm it fails on the unfixed
code, then fix the bug → loop.

Audit your **benchmarks** the same way. A measurement you have not audited is not
evidence -- this repo has already had three benchmark bugs that made correct code
look broken.

## PR description

- short summary, and the PR type
- exact commands you ran to validate
- linked issue / discussion (mandatory for feature PRs)
- any skipped check, with a concrete reason

## License

Apache-2.0. By contributing you agree your contributions are licensed the same.
