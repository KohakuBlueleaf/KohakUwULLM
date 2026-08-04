# LM configs

Every file here is a KohakuEngine config: a flat namespace of UPPER_CASE globals
that override the same-named defaults in a training script. Run one with

```bash
kogine run scripts/train/lm_pipe.py --config configs/lm/<group>/<name>.py
```

`scripts/train/lm_pipe.py` is the pipeline-parallel path and `scripts/train/lm.py`
the Lightning one; a config names neither, so read its docstring for which script
it expects. See docs/guides/writing-configs.md.

| directory | what lives there |
|---|---|
| `general/` | the production general-language pretrain |
| `tipo/` | TIPO caption/tag recipes, dense and MoE, across the parameter ladder |
| `sweep/` | throughput sweeps: one microbatch shape per file, sixty steps, no checkpoint |
| `smoke/` | short runs that prove a path works before a long one commits to it |
| `retired/` | recipes kept only because their docstring records why they were dropped |

Filenames keep their group prefix (`sweep/sweep_08192x32.py`, not
`sweep/08192x32.py`) so every stem stays a valid Python identifier: KohakuEngine
loads a script by file stem to keep objects picklable across `spawn` workers, and
a leading digit breaks that.
