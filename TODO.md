# TODO

## HF backend: remaining

- [ ] **GPU (`device="cuda"`) smoke.** The backend is validated lossless on CPU for both
  a random tiny model and a real one (SmolLM2-135M, chain + tree), and SDPA is validated
  lossless too. Only the CUDA device path is unexercised (no GPU in CI). Run a real
  checkpoint on GPU, chain + tree, and confirm bit-exact vs HF greedy.
- Done: `attn_implementation` is exposed (default `"eager"`, `"sdpa"` validated lossless
  for GPU perf; flash-attn is out since it can't take an arbitrary tree mask).

## Release / publishing

- [x] **Tag-triggered PyPI publish workflow.** `.github/workflows/publish.yml` runs on
  `v*` tag pushes, gates `uv build`/`uv publish` behind the checks job, and uses PyPI
  **trusted publishing** (OIDC, no stored token).
- [ ] Configure a pending PyPI trusted publisher for project `dejavuu`, repository
  `icnatspell/dejavuu`, workflow `publish.yml`, and environment `pypi` before the first
  tag.
- [ ] Do a TestPyPI upload and clean-environment install before the production tag.
- [ ] Tag and publish `v0.1.0` after the trusted publisher and release validation are
  complete.
