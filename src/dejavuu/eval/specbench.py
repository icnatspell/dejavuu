"""Compatibility helpers for the unified benchmark runner.

New code should use :mod:`dejavuu.eval.bench` and :mod:`dejavuu.eval.datasets`.
"""

from __future__ import annotations

from pathlib import Path

from dejavuu.decoders.text import download
from dejavuu.eval.datasets import (
    SPEC_WORKLOADS,
    SPECBENCH_REVISION,
    load_specbench_cases,
)

SPEC_BENCH_REVISION = SPECBENCH_REVISION
SPEC_BENCH_URL = (
    f"https://raw.githubusercontent.com/hemingkx/Spec-Bench/{SPEC_BENCH_REVISION}/"
    "data/spec_bench/question.jsonl"
)
CACHE = Path.home() / ".cache" / "dejavuu"
WORKLOADS = SPEC_WORKLOADS


def resolve_model_root(model_path: Path | None, variant: str) -> Path:
    """Return an explicit built decoder directory or the bundled text snapshot."""
    if model_path is not None:
        return model_path
    if variant == "fp32":
        raise ValueError("--variant fp32 requires --model-path to a built decoder directory")
    return download(variant)


def load_specbench(workload: str, n: int, per_category: int = 0) -> list[tuple[str, str]]:
    """Legacy first-turn tuple view over the pinned conversation adapter."""
    cases = load_specbench_cases(n=n, per_category=per_category, workload=workload)
    return [(case.category, case.prompt) for case in cases]


def main() -> None:
    from dejavuu.eval.bench import main as unified_main

    unified_main("specbench")


if __name__ == "__main__":
    main()
