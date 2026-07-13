#!/usr/bin/env python3
"""Rank methods and auto-diagnose their bottleneck from benchmark CSVs.

Post-processor over the CSVs `dejavuu.eval.specbench` / `mmspec` already write. It
aggregates the per-category rows into one row per method, ranks by decode speedup, and
names the dominant *reducible* phase for anything that isn't clearly winning -- turning
the phase split (draft / verify / learn / overhead) into a to-do list of gaps.

One table per input CSV, so different run configs (greedy vs sampled, warm vs cold) stay
separate -- do not pool passes into a single table. Usage:

    python scripts/gap_rank.py results/specbench_greedy_warm.csv [more.csv ...]
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict

# Gap thresholds -- first-guess heuristics; tune once real spread is known.
WIN = 1.10  # speedup at/above this is a clear win, no gap to report
ACCEPT_FLOOR = 1.30  # accept_len below this = drafts basically never land
PHASE_RATIO = 0.20  # a non-verify phase costing >20% of verify is worth cutting


def _favg(rows: list[dict[str, str]], key: str) -> float:
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    return sum(vals) / len(vals) if vals else 0.0


def _spread(rows: list[dict[str, str]], key: str) -> tuple[float, float]:
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    return (min(vals), max(vals)) if vals else (0.0, 0.0)


def diagnose(spd: float, acc: float, verify: float, learn: float, ovh: float) -> str:
    """Name the dominant reducible gap for a non-winning method."""
    if spd >= WIN:
        return "win"
    tag = "marginal" if spd >= 1.0 else "loss"
    if acc < ACCEPT_FLOOR and verify:
        return f"{tag} -> acceptance (drafts rarely land: draft-quality)"
    if learn > PHASE_RATIO * verify:
        return f"{tag} -> learn tax (O(vocab) observe)"
    if ovh > PHASE_RATIO * verify:
        return f"{tag} -> engine/accept overhead"
    return f"{tag} -> verify-bound (draft too long for accept_len / budget)"


def rank_csv(path: str) -> None:
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    title = path
    with open(path) as f:
        for row in csv.DictReader(f):
            by_method[row["method"]].append(row)
            title = row.get("title") or title

    ranked = []
    for method, rows in by_method.items():
        spd = _favg(rows, "speedup_batch")
        acc = _favg(rows, "accept_len")
        verify, learn, ovh = (_favg(rows, k) for k in ("verify_ms", "learn_ms", "overhead_ms"))
        lo, hi = _spread(rows, "speedup_batch")
        match = "exact" if all(r.get("match") in ("exact", "") for r in rows) else "LOSSY"
        verdict = "ref" if method == "baseline" else diagnose(spd, acc, verify, learn, ovh)
        ranked.append(
            (spd, method, acc, _favg(rows, "draft_ms"), verify, learn, ovh, lo, hi, match, verdict)
        )

    print(f"\n=== {title} ===")
    hdr = f"{'method':16}{'spd':>7}{'range':>13}{'acc':>6}{'drft':>6}{'vfy':>7}{'lrn':>6}{'ovh':>6} {'match':6} verdict / gap"
    print(hdr)
    print("-" * len(hdr))
    for spd, m, acc, drf, vfy, lrn, ovh, lo, hi, match, verdict in sorted(ranked, reverse=True):
        rng = f"{lo:.2f}-{hi:.2f}" if hi - lo > 0.01 else ""
        print(
            f"{m:16}{spd:>6.2f}x{rng:>13}{acc:>6.2f}{drf:>6.2f}{vfy:>7.1f}"
            f"{lrn:>6.2f}{ovh:>6.1f} {match:6} {verdict}"
        )


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/gap_rank.py <bench.csv> [more.csv ...]")
    for path in sys.argv[1:]:
        rank_csv(path)


if __name__ == "__main__":
    main()
