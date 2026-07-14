# Benchmark runbook

Reproducible steps to set up the environment, regenerate the ONNX models, and run the
full speculative-decoding benchmark on **SpecBench** and **SpeedBench** for both
reference models:

| Short name | HF id | Arch | Params |
|---|---|---|---|
| `qwen3-0.6b` | `Qwen/Qwen3-0.6B` | Qwen3ForCausalLM | 0.6 B |
| `danube3-500m` | `h2oai/h2o-danube3-500m-base` | LlamaForCausalLM | 0.5 B |

Both are conventional causal LMs, so the same tooling and drafter contract apply. All
commands assume the repo root and the `uv` toolchain.

> **What "the benchmark" measures.** A drafter proposes tokens; the verifier's accept
> rule keeps only the tokens greedy decoding would have produced, so the output is
> lossless against *the same backend*. We report decode-only `tok/s`, speedup, accepted
> length, acceptance rate, the `prefill/draft/verify/learn/overhead` phase split, and
> proposal quality (root top-1/top-5). See [`methods.md`](methods.md) for the drafters.

---

## 1. Environment

Decode is heavily **launch-bound** on these small models, so GPU is 2–3× faster per step
than CPU and makes a full sweep practical (≈13 ms/step int4 vs ≈44 ms fp32-CPU). CI and
correctness (`tests/test_conformance.py`) run on **CPU**; the GPU setup below is a local,
reversible overlay that never touches `pyproject.toml` / `uv.lock`.

### 1a. CPU (default / CI)

```bash
uv sync                      # onnxruntime 1.27 CPU, the validated baseline
uv run pytest -q             # conformance + unit tests
```

### 1b. GPU (RTX 3080 class, CUDA 12, WSL2)

The pinned `onnxruntime==1.27` (needed for `MatMulNBitsQuantizer(bits=…)` at build time)
has no CUDA wheel on PyPI that is import-stable on WSL2. Install the CUDA-12 GPU build
from Microsoft's dedicated index, isolated with `--no-deps` so it cannot pull the cu13
wheel:

```bash
# GPU runtime (replaces the CPU onnxruntime in the venv)
uv pip uninstall onnxruntime
uv pip install onnxruntime-gpu==1.27.0 --no-deps \
  --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# CUDA-12 runtime libs (skip any already present)
uv pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 \
  nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cuda-nvrtc-cu12
```

**Import-order fix.** The GPU pybind extension resolves `libcudart.so.12` at
`import onnxruntime`, *before* `ort.preload_dlls()` can run. Preload the cu12 libs at
Python startup with a local-only `sitecustomize.py` in the venv site-packages
(`python -c "import site; print(site.getsitepackages()[0])"`):

```python
# .venv/lib/python3.13/site-packages/sitecustomize.py  (local only, not in the repo)
import ctypes, os
_nvidia = os.path.join(os.path.dirname(__file__), "nvidia")
# RTLD_LOCAL, not GLOBAL: map the sonames so ORT's DT_NEEDED resolves, but keep their
# symbols out of global scope or a co-loaded torch binds these instead of its own cu12
# libs and segfaults.
for comp in ("cuda_runtime", "cuda_nvrtc", "nvjitlink", "cublas", "cufft", "curand", "cudnn"):
    libdir = os.path.join(_nvidia, comp, "lib")
    if os.path.isdir(libdir):
        for name in sorted(os.listdir(libdir)):
            if ".so" in name:
                try:
                    ctypes.CDLL(os.path.join(libdir, name), mode=ctypes.RTLD_LOCAL)
                except OSError:
                    pass
```

**Verify both invariants hold** (CUDA provider registers *and* the build-time `bits` API
is present):

```bash
uv run python -c "import onnxruntime as ort; \
  assert 'CUDAExecutionProvider' in ort.get_available_providers(); \
  from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer as Q; \
  import inspect; assert 'bits' in inspect.signature(Q.__init__).parameters; print('GPU env OK')"
```

### 1c. Restore the CPU env

```bash
uv pip uninstall onnxruntime-gpu
rm -rf "$(uv run python -c 'import onnxruntime,os;print(os.path.dirname(onnxruntime.__file__))')"
uv pip install onnxruntime==1.27.0
rm -f "$(uv run python -c 'import site;print(site.getsitepackages()[0])')/sitecustomize.py"
```

---

## 2. Regenerate the models

Each model needs **two** exports, because no single ONNX graph is both tree-capable and
GPU-fast (fused GPU attention is causal-only — see `tools/build_tree_decoder.py`):

| Export | Tool | Variants | Attention | Use |
|---|---|---|---|---|
| **Eager** | `build_decoder` | `fp32`, `int8`, `q4` | eager (4D tree mask + hidden_states) | tree verification, PLD+/AdaPLD, CPU, fidelity reference |
| **genai** | `build_genai_chain` | `fp16_genai`, `int4_genai` | fused GroupQueryAttention (chain-only) | fast GPU **chain** floor |

Run both per model (the eager `fp32` is the fidelity yardstick the genai validator scores
against, so build it first):

```bash
# ---- qwen3-0.6b ----
uv run --extra vlm --with onnx_ir --index-strategy unsafe-best-match \
  python -m dejavuu.tools.build_decoder --model Qwen/Qwen3-0.6B \
  --out ~/.cache/dejavuu/qwen3-0.6b
uv run python -m dejavuu.tools.build_genai_chain --model Qwen/Qwen3-0.6B \
  --out ~/.cache/dejavuu/qwen3-0.6b          # needs the GPU env (validates on CUDA)

# ---- danube3-500m ----
uv run --extra vlm --with onnx_ir --index-strategy unsafe-best-match \
  python -m dejavuu.tools.build_decoder --model h2oai/h2o-danube3-500m-base \
  --out ~/.cache/dejavuu/danube3-500m
uv run python -m dejavuu.tools.build_genai_chain --model h2oai/h2o-danube3-500m-base \
  --out ~/.cache/dejavuu/danube3-500m
```

`build_decoder` writes `onnx/model_{fp32,int8,q4}.onnx` + a `manifest.json` with a
teacher-forced fidelity gate. `build_genai_chain` adds `onnx/{fp16_genai,int4_genai}/`
and merges manifest entries, gating each on **batched-vs-incremental** self-consistency
(the check the old CPU q4 failed) and scoring top-1 fidelity vs the eager fp32.

> danube3 is `LlamaForCausalLM`; both the eager exporter and the genai builder support it.
> If the genai builder ever rejects an arch, only the fast GPU-chain variant is lost — the
> eager export (and thus the whole benchmark) still runs.

---

## 3. Run the benchmark

### Method set (all 22 drafters + `baseline`)

```bash
METHODS=baseline,pld,copyspec,pld_plus,adapld,anpd,cacheback,lookahead,logit_spec,ngram_trie,token_recycling,suffix_recycle,suffix_recycle_merge,suffix_recycle_tail,pld_recycle,rest,suffix_decoding,sam_decoding,stand,asam,asam_verify,asd,asd_verify
```

### Two comparison configs (pick per the question)

Hold **one artifact + budget + chain/tree** fixed across all methods (CLAUDE.md: compare
like-for-like). Two configs answer different questions:

- **A — GPU chain floor** (`--variant int4_genai --provider cuda`, chain): fastest decode.
  Every method runs in chain mode; tree/`hidden_states`-dependent methods (PLD+, AdaPLD)
  degrade gracefully to their chain form. This is the **speedup leaderboard**.
- **B — full-capability** (`--variant int8 --provider cuda --tree --width 2`, eager):
  slower per step but exercises tree verification and hidden-state reranking, so
  tree/PLD+ methods show their real behaviour.

### Commands (full sweep: every prompt, `--per-category 0`, `--repetitions 1`)

Repetitions are **1 on the full dataset**: decoding is greedy + seeded, so a re-run yields
identical accepted tokens — reps would only re-measure wall-clock jitter, which already
averages out across the many prompts. (Use `--repetitions 3` only when sampling *few*
prompts per category.)

```bash
mkdir -p results

# ---- config A: GPU chain floor, int4_genai ----
for MODEL in qwen3-0.6b danube3-500m; do
  for DS in speedbench specbench; do
    uv run --extra bench python -m dejavuu.eval.bench \
      --dataset $DS --protocol conversation --per-category 0 \
      --model-path ~/.cache/dejavuu/$MODEL --variant int4_genai --provider cuda \
      --budget 4 --max-new 128 --warmups 1 --repetitions 1 \
      --methods $METHODS --out results/${DS}-${MODEL}-int4genai
  done
done

# ---- config B: full-capability, eager int8 tree ----
for MODEL in qwen3-0.6b danube3-500m; do
  for DS in speedbench specbench; do
    uv run --extra bench python -m dejavuu.eval.bench \
      --dataset $DS --protocol conversation --per-category 0 \
      --model-path ~/.cache/dejavuu/$MODEL --variant int8 --provider cuda \
      --tree --width 2 --budget 4 --max-new 128 --warmups 1 --repetitions 1 \
      --methods $METHODS --out results/${DS}-${MODEL}-int8tree
  done
done
```

Each run writes a bundle: `summary.csv` (per-method/-category rows), `responses.jsonl`
(with `scores.text_similarity`), and `divergences.jsonl`. Divergence from the baseline is
**diagnostic, never a failure** — the run keeps all latency/quality metrics.

### Time budget

Per generation ≈ **2–2.7 s** (max-new 128, config A; config B is slower per step). Sizes:
SpeedBench qualitative = **880** prompts (11 cats × 80), SpecBench = **480**.

| Scope (per model, per dataset, 23 methods) | Prompts | ≈ Wall (config A) |
|---|---|---|
| **Full** (`--per-category 0`) SpeedBench | 880 | ~11 h |
| **Full** SpecBench | 480 | ~6 h |
| `--per-category 30` SpeedBench | 330 | ~4 h |
| `--per-category 15` SpeedBench | 165 | ~2 h |

**Full × 2 datasets × 2 models × 2 configs ≈ 60+ h.** For a first trustworthy leaderboard,
`--per-category 30` gives error bars within ~1.4× of the full set (`se ∝ 1/√N`) at a
fraction of the cost; deepen only borderline pairs afterward. Prefer `run_in_background`
or a `tmux`/`nohup` session for long runs.

---

## 4. Read the results

`render_table` (printed per run) and `summary.csv` carry every column: decode `tok/s`,
speedup (batch + per-prompt), accepted length, acceptance rate, submitted/verified counts,
time per emitted token, the `prefill/draft/verify/learn/overhead` phase split, and root
top-1/top-5 target agreement. `±` is the standard deviation **across prompts** in a
category; `repeat_std` (CSV) is jitter across repetitions.

- **Significance.** A method beats `baseline` only when its speedup interval clears the
  baseline's — the table marks this (`*` = significant at the reported spread). Treat
  overlapping intervals as "within noise", not a ranking.
- **Rank + next levers.** `uv run python scripts/gap_rank.py results/<bundle>/summary.csv`
  ranks methods by speedup and labels each one's dominant reducible gap
  (acceptance / learn / overhead / verify-bound) — that ranked list *is* the to-do list.
- **Draft-source base rate** (hybrid methods): `grep 'hybrid source\[' <run-log>`.
- **Cold vs warm** memory/cache are separate modes (`--model-memory cold|warm`,
  `--cache-scope request|run`); report them apart.

> Compare methods only within the same bundle (same artifact, provider, precision, budget,
> chain/tree). A quantized run vs an fp32 baseline is a backend difference, not a drafter
> regression.
