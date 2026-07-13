"""Compatibility helpers for MMSpec through the unified benchmark runner."""

from __future__ import annotations

from pathlib import Path

from dejavuu.eval.datasets import MMSPEC_REVISION, load_mmspec_cases

RAW = (
    f"https://raw.githubusercontent.com/killthefullmoon/MMSpec/{MMSPEC_REVISION}/"
    "dataset/MMSpec/testmini"
)
CACHE = Path.home() / ".cache" / "dejavuu" / "mmspec"


def load_mmspec(n: int, per_category: int = 0) -> list[tuple[str, str, Path]]:
    """Legacy first-turn tuple view over the pinned multimodal conversation adapter."""
    cases = load_mmspec_cases(n=n, per_category=per_category)
    return [(case.category, case.prompt, case.turns[0].images[0]) for case in cases]


def main() -> None:
    from dejavuu.eval.bench import main as unified_main

    unified_main("mmspec")


if __name__ == "__main__":
    main()
