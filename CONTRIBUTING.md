# Contributing

Thanks for helping out. This is a research-grade library with a hard correctness bar:
speculative decoding must be **lossless**, so most changes come with a test that proves
bit-exactness against the plain baseline.

## Setup

You need Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras --index-strategy unsafe-best-match   # library + vlm/hf/build extras
uv run prek install                                       # git pre-commit hooks
```

## The gate

Run the full suite before pushing. CI runs the same thing and is the final word.

```bash
uv run prek run --all-files
```

That is: `ruff` (lint + format), `pyrefly` (types), `deptry` (dependency hygiene),
`uv lock --check`, the offline test suite, and coverage (must stay >= 90%). Security
auditing (`pip-audit`) runs as a separate CI job.

Tests that need a downloaded model are marked `model` and are opt-in:

```bash
uv run pytest                 # fast, offline (default)
uv run pytest -m model        # also runs the real-model correctness checks
```

## Adding a drafter

A drafter proposes candidate tokens from raw token ids only -- never model internals --
so one instance drives every backend and both text and vision.

1. Subclass `Drafter` (`dejavuu/drafters/base.py`); implement `propose` (chain). Override
   `propose_tree` if you can branch; otherwise it defaults to the chain.
2. Register it in `DRAFTERS` (`dejavuu/drafters/__init__.py`). That alone wires it into
   the CLI, the `DejaVu` API, both benches, and the conformance suite.
3. `tests/test_conformance.py` then requires it to emit a valid `DraftTree` and stay
   bit-exact with the baseline under **both** chain and tree verification. If it can't
   pass both, it isn't done.

Losslessness is the verifier's job, not the drafter's. A drafter may propose anything;
never weaken the accept rule to make one "work".

## Adding a backend

A backend implements the `Verifier` contract (`dejavuu/core/verifier.py`): `forward`,
`empty_kv`, and -- if its KV isn't a numpy list -- `rollback_kv`/`gather_kv`. See
`decoders/hf.py` for the transformers backend as a worked example. New backends should
carry a lossless test on a small real model (mirror `tests/test_hf_backend.py`).

## Conventions

`AGENTS.md` is the working agreement (the drafter contract, cost tiers, and the "never
call a technique dead on an unoptimized implementation" rule). Use
[Conventional Commits](https://www.conventionalcommits.org/), keep `CHANGELOG.md`
current under `[Unreleased]`, and follow [SemVer](https://semver.org/) for releases.
