"""Sweep-row provenance and the contention filter.

These pin the decisions that have no visible symptom when they break: a
contended row silently kept reads as a real measurement, a never-run cell
silently zeroed reads as an OOM, and a bf16 run silently labelled fp32 puts two
precisions in one bar. Every case here is one comparison or one fallback wide.
"""

import json

from kohakuwullm.bench import (
    best_cell,
    classify,
    falling_throughput,
    load_sweep,
    ordered_presets,
    parse_tag,
    resolve_dtype,
    strategy_coverage,
)


def _row(per_micro=8192, tokens_per_s=1e5, spread=0.01, ok=True, peak=10.0):
    return dict(
        per_micro=per_micro,
        n_micro=262144 // per_micro,
        tokens_per_s=tokens_per_s,
        peak_gib=peak,
        step_spread=spread,
        ok=ok,
    )


def _write(tmp_path, name, payload):
    (tmp_path / f"{name}.json").write_text(json.dumps(payload))


def test_parse_tag_reads_every_driver_tag():
    assert parse_tag("pp4")["strategy"] == "pp4"
    assert parse_tag("pp4_ckpt")["strategy"] == "pp4_ckpt"
    assert parse_tag("ddp4_ckpt")["strategy"] == "ddp4_ckpt"

    # dtype must not become part of the strategy, or pp4 and pp4_bf16 merge into
    # one series and a best is picked across precisions.
    bf16 = parse_tag("pp4_bf16")
    assert (bf16["strategy"], bf16["dtype"]) == ("pp4", "bf16")

    # The regex this replaced matched `\w+` (which includes underscores) for its
    # trailing group, so `micro_muon_ckpt` was consumed whole and the optimizer
    # group never fired -- every muon row was reported as adamw.
    micro = parse_tag("pp4_micro_muon_ckpt")
    assert micro["optimizer"] == "muon"
    assert micro["strategy"] == "pp4_ckpt"
    assert set(micro["flags"]) == {"micro", "ckpt"}
    assert micro["unknown"] == []

    full = parse_tag("pp4_bf16_adamw_ckpt")
    assert (full["dtype"], full["optimizer"], full["strategy"]) == (
        "bf16",
        "adamw",
        "pp4_ckpt",
    )

    # An unknown part is surfaced, never folded into a guess.
    assert parse_tag("pp4_head")["unknown"] == ["head"]


def test_resolve_dtype_prefers_record_and_admits_ignorance():
    recorded = resolve_dtype({"param_dtype": "bf16"}, parse_tag("pp4"))
    assert recorded == ("bf16", None)

    # Recorded config beats a contradicting tag.
    assert resolve_dtype({"param_dtype": "fp16"}, parse_tag("pp4_bf16"))[0] == "fp16"

    # No record, no tag, no micro: the driver did not override, so it is default.
    assert resolve_dtype({}, parse_tag("pp4_ckpt"))[0] == "fp32"

    # No record and a micro tag: e2e_micro.sh forces bf16 without tagging it, so
    # guessing fp32 here would mislabel a bf16 run. It must return None.
    dtype, note = resolve_dtype({}, parse_tag("pp4_micro_muon_ckpt"))
    assert dtype is None
    assert "not recoverable" in note


def test_classify_treats_unknown_spread_as_contended():
    assert classify(_row(spread=0.049), 0.05) == "clean"
    assert classify(_row(spread=0.05), 0.05) == "clean"
    assert classify(_row(spread=0.051), 0.05) == "contended"
    assert classify(_row(spread=0.741), 0.05) == "contended"
    assert classify(_row(ok=False, spread=0.0), 0.05) == "oom"

    # An absent or NaN spread is not evidence of a quiet card. This is the case a
    # naive `spread <= max` gets wrong in the permissive direction, because NaN
    # fails the comparison and a bare `not (spread > max)` would keep it.
    assert classify({"per_micro": 8192, "ok": True}, 0.05) == "contended"
    assert classify(_row(spread=float("nan")), 0.05) == "contended"


def test_load_sweep_classifies_and_reports(tmp_path):
    _write(
        tmp_path,
        "Nano-500M_pp4",
        dict(preset="Nano-500M", world=4, rows=[_row(spread=0.01), _row(spread=0.30)]),
    )
    _write(
        tmp_path,
        "Nano-500M_pp4_micro_muon_ckpt",
        dict(preset="Nano-500M", world=4, rows=[_row()]),
    )
    _write(
        tmp_path,
        "Nano-1B_ddp4_bf16",
        dict(
            preset="Nano-1B",
            world=4,
            param_dtype="bf16",
            optimizer="adamw",
            device="NVIDIA GeForce RTX 5090",
            rows=[_row(ok=False)],
        ),
    )
    rows, notes, devices = load_sweep(str(tmp_path), max_spread=0.05)

    assert len(rows) == 4
    assert sorted(r["status"] for r in rows) == [
        "clean",
        "clean",
        "contended",
        "oom",
    ]
    assert devices == {"NVIDIA GeForce RTX 5090"}
    # The unrecoverable dtype is reported, not silently defaulted.
    assert any("not recoverable" in n for n in notes)
    micro = next(r for r in rows if "micro" in r["tag"])
    assert micro["dtype"] is None and micro["optimizer"] == "muon"
    # Recorded config is used, and an OOM row is never counted as ok.
    oom = next(r for r in rows if r["status"] == "oom")
    assert oom["dtype"] == "bf16" and not oom["ok"] and not oom["clean"]


def test_load_sweep_skips_payloads_it_cannot_attribute(tmp_path):
    _write(tmp_path, "legacy", [dict(preset="Nano-500M", tokens_per_s=1.0)])
    _write(tmp_path, "Nano-500M_pp4", dict(preset="Nano-500M", rows=[_row()]))
    rows, notes, _ = load_sweep(str(tmp_path))
    assert len(rows) == 1
    assert any("no preset/rows" in n for n in notes)


def test_best_cell_prefers_clean_and_keeps_never_run_distinct():
    rows = [
        dict(
            preset="Nano-1B",
            tag="pp4",
            strategy="pp4",
            per_micro=8192,
            tokens_per_s=9e4,
            peak_gib=20.0,
            spread=0.30,
            status="contended",
            ok=True,
            clean=False,
        ),
        dict(
            preset="Nano-1B",
            tag="pp4",
            strategy="pp4",
            per_micro=4096,
            tokens_per_s=6e4,
            peak_gib=17.0,
            spread=0.01,
            status="clean",
            ok=True,
            clean=True,
        ),
    ]
    # The contended row is faster. Picking a max over everything would report it.
    best = best_cell(rows, "Nano-1B", "pp4")
    assert best["status"] == "clean" and best["tokens_per_s"] == 6e4

    # No clean row: keep the least-contended one and say it is contended, rather
    # than dropping the cell and making it look never-run.
    only_bad = best_cell([rows[0]], "Nano-1B", "pp4")
    assert only_bad["status"] == "contended" and only_bad["tokens_per_s"] == 9e4

    # Never run is None, which callers must not render as a zero.
    assert best_cell(rows, "Nano-1B", "ddp4_ckpt") is None

    all_oom = [dict(rows[1], ok=False, clean=False, status="oom")]
    assert best_cell(all_oom, "Nano-1B", "pp4")["tokens_per_s"] == 0.0


def test_falling_throughput_catches_uniform_contention():
    def clean(per_micro, tokens_per_s, tag="pp4"):
        return dict(
            preset="Nano-1B",
            tag=tag,
            per_micro=per_micro,
            tokens_per_s=tokens_per_s,
            clean=True,
        )

    # The real Nano-1B pp4 case: 4.2% spread passes the filter, but throughput
    # nearly halves as the microbatch doubles, which no healthy run does.
    hits = falling_throughput([clean(4096, 66901), clean(8192, 38737)])
    assert len(hits) == 1
    assert hits[0][0] == "Nano-1B" and hits[0][3]["per_micro"] == 8192

    # A normal rising curve, and a small dip within tolerance, are both quiet.
    assert falling_throughput([clean(4096, 60000), clean(8192, 66000)]) == []
    assert falling_throughput([clean(4096, 66000), clean(8192, 65000)]) == []

    # Curves from different runs must not be compared against each other.
    mixed = [clean(4096, 66901, "pp4"), clean(8192, 38737, "pp4_bf16")]
    assert falling_throughput(mixed) == []


def test_ordered_presets_puts_dense_before_moe():
    # `Kohaku-MoE-1B` is in the list because a `startswith("MoE")` test files it
    # under dense, which is what the Kohaku ladder's naming would have done to
    # every sparse rung in a sweep figure.
    rows = [
        {"preset": p}
        for p in (
            "MoE-8B-A1B",
            "Nano-500M",
            "MoE-1B-A120M",
            "Nano-1B",
            "Kohaku-MoE-1B",
            "Kohaku-500M",
        )
    ]
    assert ordered_presets(rows) == [
        "Kohaku-500M",
        "Nano-1B",
        "Nano-500M",
        "Kohaku-MoE-1B",
        "MoE-1B-A120M",
        "MoE-8B-A1B",
    ]


def test_strategy_coverage_counts_presets_not_rows():
    rows = [
        {"preset": "Nano-1B", "strategy": "pp4"},
        {"preset": "Nano-1B", "strategy": "pp4"},
        {"preset": "Nano-500M", "strategy": "pp4"},
        {"preset": "Nano-1B", "strategy": "ddp4"},
    ]
    coverage = strategy_coverage(rows)
    assert coverage["pp4"] == {"Nano-1B", "Nano-500M"}
    assert coverage["ddp4"] == {"Nano-1B"}
