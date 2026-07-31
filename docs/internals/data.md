# Data

This document goes from the raw databases to the tensors a training step consumes.
Read it end to end and you should be able to plug a new corpus in, or write a
renderer, without reading the loader.

The pipeline is three stages, each swappable on its own:

```
sources/vault.py     KohakuVault db  ->  normalized record dict
renderers/tipo.py    record          ->  (user_text, output_text)
packing.py           text            ->  tokens + loss mask  ->  packed batch
loader/              the four registered loaders over all of it
```

`packing.py` sits at the top of the package because it is the batching contract
every other part shares. `sources/` and `renderers/` are separate directories
because they are two different registries and two different extension points —
adding a corpus and adding a prompt format are unrelated tasks. Everything under
`loader/` is framework plumbing that knows nothing about this corpus.

`build_dataset` / `build_loader` wire the stages together from a config-shaped
spec, which is what the training script calls.

---

## 1. The corpus

KohakuVault databases, copied to local NVMe at `/xg7/caption-datasets` (115 GB).

The copy is not incidental. Random reads over NFS run at **723 rec/s**
single-process against **4,047 rec/s** locally — 5.6x — and the gap persists under
worker parallelism (8.6k against 16.9k rec/s at 16 workers). See
`scripts/bench/data/data.py`.

| source | records | tags from | caption |
|---|---|---|---|
| `danbooru` | 11.8M | danbooru ground truth, 5-way split | Qwen 3.5 2B |
| `danbooru_tagger` | 11.8M | tagger predictions, 3-way + confidences | Qwen 3.5 2B |
| `nozomi` | large | tagger predictions | Qwen 3.5 2B |
| `cc12m` / `coyo11m` / `laion_coco` | ~12M each | tagger predictions | Qwen 3.5 2B |
| `imagenet` | small | embedded in the caption record | Qwen 3.5 2B |

`danbooru` and `danbooru_tagger` cover the same images but are *different views*:
ground-truth tags with artist/copyright/meta categories, versus predicted tags with
confidences. Training on both is not duplication.

---

## 2. Record sources

There are two shapes of source and one normalized record, so the renderer never
branches on provenance.

**`DanbooruRecords`** joins `raw/danbooru_meta.db` with `qwen-caption/danbooru.db`
by post id. Tags are danbooru ground truth in the full 5-way split
(general / artist / character / copyright / meta), plus score, dimensions and year.
The score is bucketed into a quality word (`masterpiece` … `worst quality`) by
thresholds that live in `sources/vault.py` rather than baked into the export, so
they stay a training knob.

**`PathKeyedRecords`** joins `qwen-caption/<name>.db` with `tagger/<name>_tags.db`
by image path, 1:1. Tags are *tagger predictions* with confidences in a 3-way split,
filtered by a confidence `threshold`.

Coverage is uneven, and the fallbacks are the interesting part:

- no tagger db at all (`imagenet`) → use the tags embedded in the caption record;
- caption present, tags empty → caption-only tasks;
- a danbooru id past the caption db's older id range → tags only.

None of these is an error. A record with either half alone is trainable, and the
renderer picks a task the record has the fields for.

`empty_record()` builds every field a normalized record can have, all absent, and
each source fills in what it has. Building it in full rather than per source is what
lets the renderer ask any source for any field and get `None` or `[]` instead of a
`KeyError`.

### Two operational rules, both learned the hard way

**SQLite handles are not fork-safe.** A handle opened in the parent and used from a
`num_workers > 0` worker returns *corrupt rows* rather than failing loudly. Every
source here holds its handles in `_ForkSafeVaults`, which reopens them on PID change.
Any source you write must do the same.

**`KVault.keys(prefix=None, limit=10000)` takes `limit` as a hard total, not a page
size**, and `items()` / `values()` call it with no argument. A full scan written as
`for k, v in vault.items()` silently covers the first 10,000 rows and looks like it
worked. Every scan here drives off `ids.npy` or passes an explicit limit.

Materializing 21M keys for a path-keyed vault costs minutes and gigabytes, so
`KeyIndex` caches the key list next to the db as a newline-delimited file and reloads
it on later runs, writing through a `.part` file and `os.replace` so a crashed scan
cannot leave a truncated cache behind.

---

## 3. Rendering is the augmentation

One record produces many different training examples. `TIPORenderer` draws, from an
injected `rng`:

- a **task** (`tag_to_long`, `short_to_tag_to_long`, …) from those the record's
  available fields support;
- a **length bucket** (`very_short` … `very_long`) from the tag count;
- how many tags are revealed as input versus withheld as target;
- which caption variant to use, and how many sentences and paragraphs to keep;
- whether to drop all metadata (10%), a random subset (30%), or none;
- whether to fold the user half into the output (70%) — which is what teaches the
  model to *generate* a prompt rather than only continue one.

All randomness goes through that `rng`, seeded from `(seed, epoch, index)`. That is
what makes a run reproducible *and* makes a repeated record render differently each
pass, which is why `{"name": "danbooru", "repeat": 3}` in a config is a re-render,
not a duplicate.

With probability `1 / (len(tasks) + 1)` no task is drawn at all and the example is a
plain length-conditioned continuation. That stays in the mix at every record because
it is what the model needs when given no field to convert from.

Multi-hop tasks (`tag_to_short_to_long`, `short_to_long_to_tag`,
`short_to_tag_to_long`) are entered twice in the candidate list so they are drawn more
often. They are the ones that actually require the model to chain fields.

### The protocol

The task vocabulary is **unchanged** from the original TIPO training, so a model
trained here speaks the same protocol.

```
<|very_short|> <|short|> <|long|> <|very_long|>   target length bucket
<|tag_to_long|> <|long_to_tag|> ...               which field predicts which
<|gen_meta|>                                      also generate the metadata
```

Output layout:

```
<prior metadata lines>
target: <|length|> <|task|>
<input field>: ...
<output field>: ...
```

Sentence and paragraph trimming always keeps the *first* sentence of a caption,
which carries its subject, and samples the rest without replacement in original
order.

---

## 4. Tokenization and loss masking

`encode_sample(tokenizer, user_text, output_text, max_length)` returns `input_ids`
and `labels` of equal length. Three rules:

- the user half **and the BOS** are context, not targets: their labels are `-100`
  (`IGNORE_INDEX`);
- `labels` is the *unshifted* target — the shift happens once, at pack time;
- truncation is applied to the concatenated ids, so a long caption loses its tail
  rather than its prompt.

The shift is deferred deliberately. Doing it at pack time is what makes it a shift
**inside each document**, so the last position of a document never predicts the first
token of its neighbour.

---

## 5. Packing

Samples are concatenated into one flat token axis with `cu_seqlens` marking
boundaries. Nothing is padded. `collate_packed` returns a `PackedBatch`:

| field | meaning |
|---|---|
| `tokens` | `(T,)` int64, all documents concatenated |
| `labels` | `(T,)` int64, already shifted, `-100` where masked |
| `seq_info` | boundaries + per-document position ids |
| `num_tokens` | total tokens — the denominator for throughput |
| `num_trained` | tokens with a real label — the denominator for the loss |

This matters more than usual for TIPO-shaped data. Rendered samples run roughly
50–600 tokens against a 2048+ context, so a padded batch would be ~80% padding and
packing is close to a 4x throughput multiplier before any kernel work.

**`pad_to_multiple`** appends a single fully-masked filler document to round the
packed length up. That is purely to keep `torch.compile` from re-specializing on every
distinct total; the filler contributes no gradient because all of its labels are
ignored.

### Splitting for gradient accumulation

`split_packed` splits on **document boundaries**, never at a token offset — cutting a
document in half would hand the remainder a wrong position offset. Documents are
dealt longest-first onto whichever chunk is currently smallest, which keeps per-chunk
token counts close even when the length distribution is skewed. Uneven chunks make
the backward passes uneven and stall accumulation.

### Padded batches

`collate_padded` produces the `(B, S)` layout and also returns a `PackedBatch`, so
`seq_info.packed` is the only thing that ever branches. Packing stays the default;
padded batches earn their place for three other reasons:

- **Comparability.** Reproducing a published recipe means matching its batch
  semantics, and most are specified in sequences x length, not tokens.
- **Debuggability.** A `(B, S)` tensor is inspectable; a packed one requires reading
  `cu_seqlens` to know what you are looking at.
- **Kernels that need it.** Anything without a varlen path takes `(B, S)`.

Its `num_tokens` counts padding, because otherwise a padded run's tokens/s would look
identical to a packed one's while doing several times the work.

---

## 6. Loaders

`LOADER_KIND` selects one and `LOADER_KWARGS` configures it.

| kind | batches by | sharding | resumable |
|---|---|---|---|
| `torch`, `map` | sample count | torch's sampler | no |
| `iterative` | token budget | caller passes `rank` | no |
| `ddp` | token budget | one shard per rank, from the launcher | yes |
| `pipeline` | `count` x exactly `k` tokens | data-parallel group only | yes |

### Why a token budget

The map-style path batches a *fixed number of documents*, so a step's token count is
a sum of `B` random lengths and swings with whatever happened to land together. Since
the optimizer steps once per batch, that swing lands directly in the effective batch
size.

The token-budget loaders batch to a budget instead. Documents are appended until the
running total would cross `k`; the document that did not fit is queued and opens the
next batch. Nothing is padded, so the batch is still varlen, and the per-batch total
lands in `(k - m, k]` rather than anywhere within a couple of thousand tokens of the
mean.

Three thresholds, and they are not interchangeable:

```
k        per-batch token budget, e.g. 262144 (= 256 x 1024)
m        "long content" threshold; the retry rule's target, and the bound on
         how far under budget a batch may land
ctx_max  hard truncation for a single document, set by the position encoding
```

**`m` is what buys the bound.** A document longer than `m` is *long content*: ending
the batch on it would leave a hole the size of the document itself, so instead it is
queued and the fill continues until the document that ends the batch is short
(`<= m`). The deficit is then bounded by that last short document, hence
`total > k - m`.

In practice `m` only bites below the deficit a plain greedy fill already produces —
505 tokens on this corpus, so `m=512` is a no-op and `m=128` is where the retry starts
earning its keep. See the ablation panel in `scripts/bench/data/data.py`.

### The edge cases

All handled in `pack_to_budget` and pinned by `tests/test_iterative_loader.py`:

- **A document longer than `k`.** An empty batch accepts unconditionally, so such a
  document is emitted alone in an over-budget batch instead of blocking the stream
  forever. Keeping `ctx_max <= k` makes this unreachable, and the dataset raises if
  you do not.
- **Several queued documents.** The retry queues one document per rejection, so a
  batch can end with many queued. They drain in draw order at the head of the *next*
  batch, and the batch under construction never re-reads its own queue — rejections go
  to a separate list. One shared queue would let two mutually-non-fitting documents
  bounce forever.
- **A retry that never finds a short document.** Bounded by `max_retry`; past it the
  batch ends regardless, and only then can the total fall below `k - m`.
- **End of stream.** The queue is drained by further batches after the stream runs
  dry. Each drains at least one document (the unconditional first accept), so the
  drain terminates.

A batch is always yielded with the queue drained, which is what lets a caller
reconstruct the packer's entire state as *drawn minus emitted*. Resume depends on that.

### Sharding

The token-budget loaders shard **twice over, nested**: shard
`rank * num_workers + worker_id` of `world_size * num_workers`, every shard striding
one permutation of `(seed, epoch)`.

Every shard draws the *same* permutation and takes a different stride of it. That is
what makes the shards disjoint and jointly complete — an `IterableDataset` is copied
wholesale into each worker, so a dataset that does not do this hands every worker the
same data.

Striding a shared permutation costs `4n` bytes transiently per worker (47 MB at 11.8M
records). The cheap alternative — give worker `w` the indices `i % num_shards == w`
and shuffle only those — is `O(n / num_shards)` but *freezes the partition*: two
records in different residue classes could then never share a batch, in any epoch, for
the whole run.

Batches are built **per worker**: each worker fills its own budget from its own shard
and the loader round-robins between them, so `k` stays the per-step budget while the
workers are what parallelize.

### `batches_per_epoch` is close to mandatory above one rank

Shards hold equal *document* counts but not equal *token* counts, so ranks disagree on
how many batches an epoch has, and the short rank leaves the collective early. That
surfaces as a **hang in the next all-reduce**, not as an error in the loader.
`build_ddp_loader` warns when you omit it. Set it below the expected count,
`tokens / (k * world_size * num_workers)`.

### Epochs

With `persistent_workers=True` a worker re-enters `__iter__` per epoch and advances
its own `_pass` counter, which is what keeps a persistent worker from replaying one
epoch forever — the worker holds its own copy of the dataset and never sees
`set_epoch`. Without persistence, workers are re-forked from a parent whose counter
never moved and every epoch silently replays the first, so `make_loader` ties
`persistent_workers` to `num_workers` rather than exposing it.

Call `set_epoch` on the loader between epochs regardless; it is what the shuffle keys
off.

---

## 7. The pipeline loader

`torch.distributed.pipelining` fixes the boundary activation's shape when the stage is
built (`PipelineStage(input_args=...)`), and every send after that is a bare tile
matched against that declaration. So the loader owes it an *identical* token count per
microbatch, not a nearly identical one.

`MicroBatchedDataset` gets that by setting `pad_to_multiple = k`: the packer never
exceeds `k`, so the filler is exactly the deficit and every microbatch is exactly `k`
tokens. The pad region contributes nothing to the loss — its labels are
`IGNORE_INDEX` and the trainer normalizes by trained tokens.

**Padding rather than splitting is the deliberate choice.** Cutting a document at the
budget would give the shape for free, but a TIPO example is a prompt and its
completion: half of one is a different training example, and the half that lands in the
next microbatch has no prompt at all. The retry rule bounds what the padding costs
instead — the deficit is at most `m`, so at `k=8192, m=512` the pad is under 6% worst
case and well under that in practice.

A trailing partial step is dropped rather than padded out with empty microbatches: the
schedule's microbatch count is fixed when the stage is built, and a step of a different
width deadlocks the ranks expecting the declared one.

Sizes are not free parameters. Measured on 4x5090: **8192 tokens per microbatch, 32
microbatches per 262144-token step**. Throughput declines monotonically above 8192 —
the bubble grows faster than GEMM efficiency improves — and collapses below it, where
the schedule is launch-bound.

All pipeline stages must see the *same* data, so this loader does not shard across the
pipeline dimension: stage 0 needs its tokens and the last stage needs its labels.
Under pure PP leave `rank` / `world_size` at their defaults; under PP+DDP pass the
**data-parallel** group's rank and size, never the global ones.

`MicroBatchedStep` keeps `tokens` and `labels` flat over the whole step so torch's
schedules can chunk them along dim 0 and get precisely the microbatch boundaries back;
`microbatches` is the same data as views for a runtime that wants the split done for
it. `seq_infos` is per microbatch with offsets local to it, because sequence metadata
is never pipelined — every stage derives its own layout from that list.

---

## 8. Resume

A 50–100B-token run is days long, so a crash at hour 30 must continue rather than start
over: same document order, same pack boundaries, nothing repeated and nothing skipped.
`ddp` and `pipeline` do that. Three properties make it reachable.

**No hidden rng state.** The shuffle comes from `(seed, epoch)` and each render from
`(seed, epoch, index)`, so a position is just: a pass number, an offset into the shard,
and the handful of documents the packer has drawn but not yet emitted.

**The position is pushed, not pulled.** torch offers the main process no way to ask a
live worker where it is, so every batch arrives carrying the position of the worker
that produced it and `ResumableLoader` records it on the way past. What a checkpoint
holds must be the position of the last batch the *trainer consumed*, never one a worker
or the fetcher had run ahead to draw — that one dies with the process, and a position
describing it would skip it on resume.

**Every rank's position is in every rank's checkpoint.** Lightning writes only rank 0's,
so `state_dict` all-gathers first. Restoring rank 0's cursor into rank 1's shard would
land at a plausible-looking offset in the wrong data, which is the failure mode with no
symptom.

### The one-batch lag

The position is committed one batch **behind** the hand-off when, and only when,
`batches_per_epoch` is unset. Lightning's fetcher pre-pulls exactly one batch from a
loader with no `__len__` so it can see the end coming, and none at all from one that has
a length (`_PrefetchDataFetcher.__iter__`). Committing on hand-off would then describe a
batch still sitting in that queue, and the resume would skip it. The lag is selected
once, from whether the loader has a length at all — the same condition the fetcher
prefetches on — and closed by `_flush` when the underlying loader is exhausted.

### The rotation

A restart resumes each shard, but a fresh `DataLoader` iterator always begins its
round-robin at worker 0, and a crash rarely lands on a cycle boundary. `_rotation`
computes the offset — the next batch belongs to the worker with the fewest delivered,
since the round-robin hands earlier workers one more each before it wraps — and each
cycle is rotated by it for the rest of the epoch.

The whole cycle is buffered before any of it is handed on, rather than only the batches
that move: a cycle that ends short must be left in arrival order, and which cycle that is
cannot be known until it is short. It costs one cycle of queue for one epoch; the next
starts in phase again. Only *ordering* rides on the rotation — it reorders delivery and
drops nothing, so even where a strict round-robin stops holding, no document is skipped
or repeated.

### Do not rely on Lightning to restore it

`state_dict` / `load_state_dict` are the names Lightning looks for, and it *does* save
them into the fit loop's state. It does **not** reliably restore them.
`_load_combined_loader_states` runs from `_FitLoop.setup_data`, which returns early once
the combined loader exists, and anything that read `trainer.estimated_stepping_batches`
has already built it — before `restore_training_state` ever runs.

That was measured, not inferred, on a schedule with `end: -1`: the position reached the
checkpoint and was never applied, and the resumed run re-trained on batches it had already
seen. `LMTrainer` therefore saves and restores it itself, from a hook that always precedes
`setup_data`. Both paths may apply the same position, so `load_state_dict` must stay
idempotent.

### Mismatches raise

Resuming onto a different rank or worker count re-cuts the shards, so the saved offsets
would point into data this run never had. It would run, and it would quietly replay some
documents and skip others for the rest of the epoch. All three mismatches — world size,
worker count, a missing per-rank entry — raise instead.

---

## 9. Tokenizer

DeepSeek-V4's BPE, pruned to 64,000 ordinary tokens plus 1,536 special slots =
**65,536** exactly.

```
[0     , 64000)   ordinary BPE tokens, kept in merge order
[64000 , 64017)   <|bos|> <|eos|> <|pad|> <|unk|> + the TIPO task vocabulary
[64017 , 65536)   <|reserved_N|> placeholders
```

```bash
.venv/bin/python scripts/tokenizer/build_tokenizer.py --out models/tokenizer \
    --check-samples 1000
```

**Why prune.** A general-purpose 128k vocabulary spends most of its table on tokens this
corpus never emits (other scripts, code, rare Unicode), while the embedding and head cost
scales linearly with it. At `dim=1280` a 128k vocab is 164M tied parameters; 64k is 82M.
On a 500M model that is the difference between the embedding being a third of the model
and a sixth.

**Why ordered truncation.** A trained BPE assigns ids in merge order, so id `i` depends
only on tokens with smaller ids. Truncating to the first `N` ids therefore yields a
*closed* vocabulary — no surviving merge can reference a dropped token — and the byte-level
alphabet occupies the first 256 ids and is never at risk. Frequency-aware pruning (keep
the `N` tokens this corpus actually uses) would fit the domain better but breaks that
closure: dropping a mid-rank token orphans every later merge built on it. Ordered
truncation is correct by construction, which is what makes it the default.

**Why reserved slots come last.** Ids stay stable: token id `k` means the same thing
before and after adding a new task token, so introducing one is an id assignment rather
than a re-embedding, and old checkpoints keep their rows.

Measured cost on real records: **+3.9% tokens per sample, 0 unknown tokens**. Cheap for
halving the embedding.

---

## 10. Throughput, and one measured negative result

The loader is not the bottleneck. Full pipeline (read, render, tokenize, pack) on local
NVMe, measured after draining the prefetch buffer:

| workers | tokens/s |
|---|---|
| 0 | 230k |
| 4 | 745k |
| 8 | 1.04M |
| **16** | **1.11M** |
| 32 | 876k |

Against the ~400k tokens/s a 500M dense model on four 5090s consumes, that is ~2.8x
headroom. **Use 16 workers**; past that the vault read rate flattens and per-worker
overhead starts to cost.

**Measurement note.** Time more batches than the prefetch buffer holds. With 32 workers
and `prefetch_factor=4` the loader has 128 batches queued, so timing 30 of them measures
how fast a queue empties, not how fast it refills — which is how an earlier version of
this benchmark reported a fictional 12.4M tokens/s.

### The per-sample DataLoader is already near-optimal. Do not group.

Measured cost of one sample (danbooru, local NVMe):

```
read + render          0.227 ms
tokenize individually  0.621 ms
tokenize batched(32)   0.554 ms   -- only 1.12x, not the multiple assumed
```

Tokenization is 73% of the cost, so batching the tokenizer call looks like the obvious
win. It is not: the Rust tokenizer's per-call overhead is small relative to the work, so
batching 32 strings saves 12%.

And *grouping to enable that batching costs far more than it saves*. torch's default
per-sample map-style dataset shards individual indices across workers, so a 32-sample
batch is built by up to 32 workers in parallel. A grouped dataset whose `__getitem__`
returns a whole batch puts all 32 renders and reads on **one** worker. Measured end to
end: **3.1M tok/s per-sample against 226k tok/s grouped — a 12x regression to buy a 1.12x
tokenizer win.**

`BatchRenderedDataset` and `_grouped_loader_do_not_use` are kept, unwired, so that
result stays reproducible in an afternoon rather than a week. `BatchRenderedDataset` is
also the right shape for a *pre-tokenization* pass — one process, writing shards to disk
— where there is no DataLoader parallelism to lose.

### Sources are held as a lazy view

`_ConcatRepeated` is an indexable *view* over several record sources, not a
materialized list. The sources are lazy — `DanbooruRecords` holds an id array and hits
the database per `__getitem__` — so materializing them costs one query per record and
holds every record in host memory. On the production corpus that was **25 minutes and
12 GB** before a single token was tokenized, which defeats the whole point of a
streaming loader. A repeat lists the source again rather than duplicating its records,
since the renderer is seeded per draw.

---

## 11. Writing a dataset for this framework

A record source is any object with `__len__` and `__getitem__` returning a normalized
record dict, or `None` for a missing row. That is the whole interface. Register it or
pass the instance directly.

The checklist:

1. **Return `empty_record()` filled in**, so the renderer can ask for any field.
   Missing is `None` or `[]`, never a `KeyError`.
2. **Reopen database handles on PID change.** Use `_ForkSafeVaults`. An inherited
   SQLite handle returns corrupt rows rather than raising.
3. **`None` means missing, and the loaders handle it differently on purpose.** The
   map-style path emits an empty sample so the dataset length is preserved; the
   iterative path skips the row outright, because it has no fixed length to preserve.
4. **Never scan with a defaulted `limit`.** Drive off an id array or pass an explicit
   one.
5. **If it is expensive to index, cache the index next to the db**, written through a
   temp file and `os.replace`.

A renderer is a callable `(record, rng) -> (user_text, output_text)`. One rule, and it
is not optional: **take all randomness from the injected `rng`.** The dataset seeds it
from `(seed, epoch, index)`, which is what makes a repeated record render differently
each pass while the run stays reproducible. A renderer that calls the global `random`
module breaks both properties.
