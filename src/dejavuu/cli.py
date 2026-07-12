"""Single-generation CLI: `dejavuu "<prompt>" --method ...`."""

import time
from enum import Enum
from typing import Annotated

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from dejavuu.config import GenerationConfig
from dejavuu.core import Sampler, generate
from dejavuu.decoders.text import Model, download
from dejavuu.drafters import METHODS, make_drafter

Method = Enum("Method", {k: k for k in METHODS})
Variant = Enum("Variant", {"q4": "q4", "int8": "int8"})
Provider = Enum("Provider", {"cpu": "cpu", "cuda": "cuda"})


def main(
    prompt: Annotated[str, typer.Argument()] = "def fib(n):\n    return",
    method: Method = Method.pld,
    variant: Variant = Variant.q4,
    provider: Provider = Provider.cpu,
    max_new: int = 64,
    budget: int = 8,
    temperature: Annotated[float, typer.Option(help="0 = greedy")] = 0.0,
    top_p: float = 1.0,
    seed: int = 0,
) -> None:
    cfg = GenerationConfig(
        method=method.value,
        max_new=max_new,
        budget=budget,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
    logger.info("loading {} ({}, {})", cfg.method, variant.value, provider.value)
    root = download(variant.value)
    model = Model(root, variant.value, provider.value)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(root)
    ids = tok(prompt)["input_ids"]
    drafter = make_drafter(cfg.method)
    sampler = Sampler(cfg.temperature, cfg.top_p, cfg.seed) if cfg.temperature > 0 else None

    bar = tqdm(total=cfg.max_new, unit="tok", desc=cfg.method)
    t = time.time()
    res = generate(
        model,
        ids,
        cfg.max_new,
        drafter,
        cfg.budget,
        tok.eos_token_id,
        on_emit=lambda _t, _acc: bar.update(1),
        sampler=sampler,
    )
    dt = time.time() - t
    bar.close()

    console = Console()
    console.print(tok.decode(res.tokens))
    acc = res.accepted / res.drafted if res.drafted else 0.0
    table = Table(title=f"{method.value} / {variant.value}")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("tokens", str(len(res.tokens)))
    table.add_row("verify steps", str(res.steps))
    table.add_row("tokens/s", f"{len(res.tokens) / dt:.1f}")
    table.add_row("mean accept len", f"{len(res.tokens) / res.steps:.2f}")
    table.add_row("accept rate", f"{acc:.0%} ({res.accepted}/{res.drafted})")
    console.print(table)


def app() -> None:
    typer.run(main)


if __name__ == "__main__":
    app()
