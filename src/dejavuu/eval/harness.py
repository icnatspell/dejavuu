"""Shared benchmark infrastructure for the Spec-Bench (text) and MMSpec (vision)
harnesses: the method registry, per-prompt aggregation, the drafter factory (with
optional static datastore), and the comparison table."""

from __future__ import annotations

import csv
import json
import os
import platform
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, stdev

from loguru import logger
from rich.console import Console
from rich.table import Table

from dejavuu.core.engine import GenResult

# The method registry (DRAFTERS / make_drafter) lives in the library, not here -- see
# dejavuu.drafters. Re-exported so existing `from dejavuu.eval.harness import DRAFTERS`
# call sites keep working.
from dejavuu.drafters import DRAFTERS, make_drafter

__all__ = [
    "DRAFTERS",
    "Agg",
    "benchmark_metadata",
    "load_datastore",
    "make_drafter",
    "render_table",
    "write_run_manifest",
]


def benchmark_metadata(
    *,
    dataset: str,
    model: str,
    provider: str,
    threads: int,
    budget: int,
    tree: bool,
    width: int,
    max_new: int,
) -> dict[str, object]:
    """Describe the hardware and decode settings required to compare runs fairly."""
    return {
        "schema_version": 1,
        "benchmark": dataset,
        "model": model,
        "runtime": {"provider": provider, "threads": threads},
        "decode": {"budget": budget, "max_new": max_new, "tree": tree, "width": width},
        "host": {
            "cpu_count": os.cpu_count(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
    }


def write_run_manifest(csv_path: Path, metadata: dict[str, object]) -> Path:
    """Write stable, machine-readable run metadata beside a benchmark CSV.

    A leaderboard must compare like-for-like runs: the CSV holds measured decode
    metrics, while this sidecar records the model artifact and execution settings
    that materially affect them. Callers provide explicit values rather than relying
    on filenames or ambient process state.
    """
    manifest = csv_path.with_suffix(".manifest.json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return manifest


def load_datastore(path: Path, tokenizer) -> list[list[int]]:
    """Tokenize a corpus file (one document per line) into token-id docs to seed the
    static store of REST / SAM-Decoding. Uses the model's own tokenizer so ids line up."""
    docs = Path(path).read_text().splitlines()
    return [tokenizer(d)["input_ids"] for d in docs if d.strip()]


@dataclass
class Agg:
    tokens: int = 0
    steps: int = 0
    drafted: int = 0
    accepted: int = 0
    gen_s: float = 0.0
    draft_s: float = 0.0
    verify_s: float = 0.0
    prefill_s: float = 0.0
    exact: bool = True  # strict gate (text): any per-prompt mismatch flips it
    cmp_tok: int = 0  # tokens compared vs baseline (non-strict / VLM match %)
    match_tok: int = 0
    ratios: list[float] = field(default_factory=list)  # per-prompt tps/baseline_tps
    # per-prompt samples for mean +/- std (tps, accept len/%, the four ms timings);
    # pooled sums above stay for speedup(batch) and the exactness gate.
    series: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def add(self, r: GenResult, dt: float) -> None:
        self.tokens += len(r.tokens)
        self.steps += r.steps
        self.drafted += r.drafted
        self.accepted += r.accepted
        self.gen_s += dt
        self.draft_s += r.draft_s
        self.verify_s += r.verify_s
        self.prefill_s += r.prefill_s
        ddt = dt - r.prefill_s  # decode-only wallclock
        st = r.steps or 1
        s = self.series
        if ddt:
            s["tps"].append(len(r.tokens) / ddt)
        s["alen"].append(len(r.tokens) / st)
        if r.drafted:
            s["apct"].append(r.accepted / r.drafted)
        if r.root_proposals:
            s["root_top1"].append(r.root_top1 / r.root_proposals)
            s["root_top5"].append(r.root_top5 / r.root_proposals)
        if len(r.tokens):
            s["ms_out"].append(ddt / len(r.tokens) * 1e3)
            s["submitted_out"].append(r.drafted / len(r.tokens))
            s["verified_out"].append((r.drafted + r.steps) / len(r.tokens))
        s["prefill"].append(r.prefill_s / st * 1e3)
        s["draft"].append(r.draft_s / st * 1e3)
        s["verify"].append(r.verify_s / st * 1e3)
        s["overhead"].append((ddt - r.draft_s - r.verify_s) / st * 1e3)

    def speedups(self, dt: float, n_tok: int, base_tps: float) -> None:
        """Record this prompt's speedup vs its own baseline (for the per-prompt mean,
        as opposed to the pooled aggregate ratio). Call once per spec result."""
        if dt and base_tps:
            self.ratios.append((n_tok / dt) / base_tps)

    def compare(self, out: list[int], baseline: list[int]) -> None:
        """Record exactness vs the baseline (both the strict bool and token-match
        counts). Position-wise over the longer sequence, so divergence in length
        or content both count as misses."""
        if out != baseline:
            self.exact = False
        self.cmp_tok += max(len(out), len(baseline))
        self.match_tok += sum(a == b for a, b in zip(out, baseline, strict=False))


def _decode_tps(a: Agg) -> float:
    """tok/s over decode time only -- prefill is a one-time prompt tax, orthogonal to
    speculative decoding, so including it unfairly dilutes the decode-loop speedup."""
    d = a.gen_s - a.prefill_s
    return a.tokens / d if d else 0.0


def _mstd(vals: list[float]) -> tuple[float, float] | None:
    """(mean, std) over per-prompt samples; std is 0 for a single prompt, None if empty."""
    if not vals:
        return None
    return fmean(vals), (stdev(vals) if len(vals) > 1 else 0.0)


def _fmt(ms: tuple[float, float] | None, prec: int = 1, suffix: str = "") -> str:
    """Render a (mean, std) as 'mean ± std' for the rich table; '-' if no samples."""
    if ms is None:
        return "-"
    m, sd = ms
    return f"{m:.{prec}f} ± {sd:.{prec}f}{suffix}"


def _pair(vals: list[float], prec: int) -> tuple[str, str]:
    """(mean, std) as a pair of formatted CSV cells; ('', '') if no samples."""
    ms = _mstd(vals)
    return ("", "") if ms is None else (f"{ms[0]:.{prec}f}", f"{ms[1]:.{prec}f}")


def render_table(
    title: str,
    methods: list[str],
    aggs: dict[str, dict[str, Agg]],
    save: Path | None = None,
    strict: bool = True,
    csv_path: Path | None = None,
) -> None:
    """Per-category speed/accept/exactness comparison; shared by text and VLM benches.
    `aggs` is keyed category -> method -> Agg; speedup is computed against *that
    category's* baseline (no mean across categories -- each category is a row-group).
    `strict=True` shows the bit-exact gate (✓/✗ LOSSY); `strict=False` shows a
    token-match % vs baseline (for VLM, whose quantized genai decoder isn't
    length-invariant so spec-decode isn't strictly lossless -- see model contract).
    If `save` is given, also append the rendered table (plain text) to that file."""
    table = Table(title=title)
    last = "exact" if strict else "tok match"
    cols = (
        "category",
        "method",
        "tok/s",
        "speedup(batch)",
        "speedup(prompt)",
        "accept len",
        "accept %",
        "draft/out",
        "verify in/out",
        "ms/out",
        "root top1",
        "root top5",
        "prefill ms",
        "draft ms",
        "verify ms",
        "overhead ms",
        last,
    )
    for col in cols:
        table.add_column(col, justify="right", no_wrap=True)
    table.caption = (
        "tok/s and speedups are decode-only (prefill excluded — it's a one-time "
        "prompt tax, orthogonal to spec decoding; see prefill ms).   "
        "speedup(batch) = Σtokens/Σdecode-time ÷ baseline — throughput over the whole "
        "category, long outputs weighted more.   "
        "speedup(prompt) = mean of per-prompt speedup ratios — each prompt weighted "
        "equally, reflects the typical single-prompt experience."
    )
    for cat, cat_aggs in sorted(aggs.items()):
        base = cat_aggs.get("baseline")
        base_tps = _decode_tps(base) if base else None
        for mi, m in enumerate(methods):
            a = cat_aggs[m]
            s = a.series
            # speedup(batch) stays the pooled aggregate (no per-prompt distribution);
            # every other numeric column is reported as per-prompt mean ± std.
            speedup = f"{_decode_tps(a) / base_tps:.2f}x" if base_tps else "-"
            # per-step time split: total = prefill + draft + verify + overhead.
            # prefill is one-time (amortized over steps; shrinks with longer output);
            # draft = index lookup; verify = model forward; overhead = accept/KV remainder.
            if m == "baseline":
                match = "-"
            elif strict:
                match = "✓" if a.exact else "✗ LOSSY"
            else:
                match = f"{a.match_tok / a.cmp_tok:.1%}" if a.cmp_tok else "-"
            table.add_row(
                cat if mi == 0 else "",
                m,
                _fmt(_mstd(s["tps"])),
                speedup,
                _fmt(_mstd(a.ratios), prec=2, suffix="x"),
                _fmt(_mstd(s["alen"]), prec=2),
                _fmt(_mstd([p * 100 for p in s["apct"]]), prec=0, suffix="%"),
                _fmt(_mstd(s["submitted_out"]), prec=2),
                _fmt(_mstd(s["verified_out"]), prec=2),
                _fmt(_mstd(s["ms_out"]), prec=1),
                _fmt(_mstd([p * 100 for p in s["root_top1"]]), prec=0, suffix="%"),
                _fmt(_mstd([p * 100 for p in s["root_top5"]]), prec=0, suffix="%"),
                _fmt(_mstd(s["prefill"]), prec=0),
                _fmt(_mstd(s["draft"]), prec=0),
                _fmt(_mstd(s["verify"]), prec=0),
                _fmt(_mstd(s["overhead"]), prec=0),
                match,
                end_section=mi == len(methods) - 1,
            )
    # Fixed wide width when logging so nothing truncates to an 80-col default
    # (no terminal to detect); on-screen print auto-fits the terminal.
    console = Console(record=save is not None, width=200 if save else None)
    console.print(table)
    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        with save.open("a") as f:
            f.write(console.export_text())
        logger.info("results -> {}", save)
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not csv_path.exists()
        with csv_path.open("a", newline="") as f:
            w = csv.writer(f)
            # per-prompt metrics get a value + _std pair; speedup_batch is the single
            # pooled aggregate (no distribution).
            if is_new:
                w.writerow(
                    [
                        "title",
                        "category",
                        "method",
                        "tok_s",
                        "tok_s_std",
                        "speedup_batch",
                        "speedup_prompt",
                        "speedup_prompt_std",
                        "accept_len",
                        "accept_len_std",
                        "accept_pct",
                        "accept_pct_std",
                        "draft_per_output",
                        "draft_per_output_std",
                        "verified_per_output",
                        "verified_per_output_std",
                        "ms_per_output",
                        "ms_per_output_std",
                        "root_top1",
                        "root_top1_std",
                        "root_top5",
                        "root_top5_std",
                        "prefill_ms",
                        "prefill_ms_std",
                        "draft_ms",
                        "draft_ms_std",
                        "verify_ms",
                        "verify_ms_std",
                        "overhead_ms",
                        "overhead_ms_std",
                        "match",
                    ]
                )
            for cat, cat_aggs in sorted(aggs.items()):
                base = cat_aggs.get("baseline")
                base_tps = _decode_tps(base) if base else None
                for m in methods:
                    a = cat_aggs[m]
                    s = a.series
                    speedup = f"{_decode_tps(a) / base_tps:.4f}" if base_tps else ""
                    if m == "baseline":
                        match = ""
                    elif strict:
                        match = "exact" if a.exact else "lossy"
                    else:
                        match = f"{a.match_tok / a.cmp_tok:.4f}" if a.cmp_tok else ""
                    w.writerow(
                        [
                            title,
                            cat,
                            m,
                            *_pair(s["tps"], 4),
                            speedup,
                            *_pair(a.ratios, 4),
                            *_pair(s["alen"], 4),
                            *_pair(s["apct"], 4),
                            *_pair(s["submitted_out"], 4),
                            *_pair(s["verified_out"], 4),
                            *_pair(s["ms_out"], 4),
                            *_pair(s["root_top1"], 4),
                            *_pair(s["root_top5"], 4),
                            *_pair(s["prefill"], 2),
                            *_pair(s["draft"], 2),
                            *_pair(s["verify"], 2),
                            *_pair(s["overhead"], 2),
                            match,
                        ]
                    )
        logger.info("csv -> {}", csv_path)
