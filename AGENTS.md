# Working agreement

## Issue-first integration

Before starting any integration work, check GitHub for an existing issue with
`gh issue list --state all`. Link the work to that issue when one exists; otherwise,
create an issue that describes the method or integration, intended benefit, expected
drafter/verifier impact, and primary paper or code reference before implementation.
Do this before creating an implementation branch or changing code.

## Engineering standards

- Write comments, docstrings, and benchmark explanations for an AI/ML engineer.
  When audience expertise is uncertain, target a junior AI/ML engineer: define
  spec-decoding terms, explain why a measurement or constraint matters, and avoid
  unexplained runtime jargon.
- Use [Conventional Commits](https://www.conventionalcommits.org/) for commit
  messages and [Semantic Versioning](https://semver.org/) for released versions.
- Run `uv run prek run --all-files` before handing off changes. CI runs the same
  suite (plus `pip-audit`) and is the final enforcement layer.

## The drafter contract is the standardization boundary

Every drafter is a drop-in: it works on raw token-id lists only (never model
internals) and emits a `DraftTree`, so the same instance drives the LLM (`Model`)
and the VLM (`VLM`) through the shared `Verifier`/engine. Preserve this:

- A new drafter subclasses `Drafter` and is registered in `eval.harness.DRAFTERS`.
  Registration alone wires it into the CLI, both benches, and the conformance
  suite -- no per-drafter test scaffolding needed.
- **Every method must be usable under both chain and tree verification.**
  `propose` is the chain path; `propose_tree` is the branching path (default: the
  chain, so tree support is free unless you have ranked candidates to branch on).
  `tests/test_conformance.py` enforces this: it runs every `DRAFTERS` entry
  through a model-free verifier under `tree=False` and `tree=True` and asserts a
  valid tree shape plus bit-exact output vs the autoregressive baseline. If your
  drafter can't pass both, it isn't done.
- Losslessness is the verifier's job, not the drafter's. A drafter may propose
  anything; correctness comes from greedy accept / seeded-sample coupling in the
  engine. Never weaken that gate to make a drafter "work".

## Classify every cost into one tier

1. **Offline:** export, conversion, quantization, tree-capable re-export.
2. **Online once:** session setup, artifact load, prompt prefill / KV precompute.
3. **Online hot path:** per-step draft, verify forward, accept, KV composition,
   sampling.

Never present a tier-one or tier-two saving as a tier-three latency/TPS gain, and
report `tok/s`/speedup as decode-only (prefill excluded -- see `render_table`).

## Never call a technique dead on an unoptimized implementation

A negative result only indicts the version you measured. Before writing "no-go",
"dead", or any verdict that stops a line of work, confirm you are assessing an
*adequately optimized* implementation of the idea.

- Separate "this **implementation** is slow/low quality" from "the **technique**
  cannot work here". Only the second justifies stopping; most first attempts are
  the first.
- State the optimization level alongside the result (e.g. "brute-force cosine, no
  ANN"). A number without that label is not a verdict.
- Prefer "conditional / needs X" over "no-go" whenever a named lever is untried.
  Reserve "dead" for when the theoretical ceiling itself is unfavorable.

## Test-first delivery

Use a vertical red-green-refactor loop: write one public-interface test for the
highest-value behavior, watch it fail, implement the minimum to pass, repeat one
behavior at a time. Tests assert observable behavior, not private internals. For
performance-sensitive code, test correctness and protocol invariants separately
from machine-specific timing thresholds.
