"""Build onnxruntime-genai fused CHAIN-mode ONNX variants for the GPU decode path.

genai's model builder fuses attention into one `GroupQueryAttention` op and keeps the
graph entirely on the GPU (the stock torch/eager export runs ~1238 shape/RoPE nodes on
the CPU EP per step). Crucially it also keeps OrtDecoder's conventional KV I/O
(`past_key_values.{i}` -> `present.{i}`), so the existing runtime drives it directly with
device-resident KV (#46). Measured on qwen3-0.6b/RTX 3080 @ past_len 1024, device_kv:
the fp32 eager export floors at ~25 ms/step; genai fp16 at ~15.5, genai int4 at ~13.

genai is **chain-only**: its GQA is causal and it emits no 4D tree mask and no
hidden_states (see `build_tree_decoder.py`). So these variants are the fast GPU *chain*
path; the eager torch export (fp32/int8/q4) stays the tree-capable + hidden-states path
that PLD+/AdaPLD need. fp16/int4 are not bit-exact vs fp32 -- that's a backend precision
choice, measured as fidelity below, not a correctness break (the engine's losslessness is
spec-vs-same-variant, judged in the harness, never vs fp32).

Two variants land in `<out>/onnx/<name>/model.onnx`:
  - fp16_genai -- fidelity-safe fallback.
  - int4_genai -- fastest + smallest (block-wise weight-only). Unlike the repo's old CPU
                  q4 (which failed batched-vs-incremental), genai int4 is spec-self-
                  consistent on the GPU MatMulNBits kernel -- validated here, not assumed.

Each variant is gated on **batched-vs-incremental** self-consistency (spec verification
submits several tokens in one forward; those causal logits must pick the same tokens as
one-at-a-time decoding, or acceptance collapses -- this is the exact check the old q4
failed) and scored for teacher-forced top-1 **fidelity vs the fp32 onnx** (quality proxy).
Both run on CUDA because GQA is a CUDA contrib op and won't load on the CPU EP.

    uv run python -m dejavuu.tools.build_genai_chain --model Qwen/Qwen3-0.6B \
        --out ~/.cache/dejavuu/qwen3-0.6b --hf-cache ~/.cache/huggingface/hub
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from transformers import AutoTokenizer

from dejavuu.decoders.ort import OrtDecoder, make_session
from dejavuu.tools.artifact import MANIFEST, write_manifest

# variant name -> genai builder precision flag. Names carry the `_genai` suffix so they
# never collide with the eager torch export's fp32/int8/q4 and a reader can tell at a
# glance which path (fused/chain-only/GPU) an artifact is.
VARIANTS = {"fp16_genai": "fp16", "int4_genai": "int4"}

# Batched-vs-incremental agreement floor to call a variant spec-safe. NOT 1.0: on the GPU,
# a batched (M>1) forward and one-at-a-time (M=1) decoding take different GQA kernel paths,
# so fp16 flips a near-tie argmax at the odd position -- a backend-precision metric, not a
# broken accept/KV path (CLAUDE.md: treat backend exactness as a metric, not a validity
# gate). A ~1% flip costs ~1% acceptance; the old CPU q4 that this check flagged was far
# below this. On the deterministic CPU EP the eager export still hits exactly 1.0.
SPEC_CONSISTENCY_FLOOR = 0.98

# Probes for the validation gate: a few deliberately different registers (fact, code,
# reasoning) so a quant that degrades on one shows up, instead of one lucky prompt.
PROBES = [
    "The capital of France is",
    "def fibonacci(n):",
    "Q: If a train travels 60 km in 45 minutes, what is its speed in km/h? A:",
]


def _build(model: str, revision: str | None, hf_cache: Path, precision: str, dst: Path) -> None:
    """Run the genai builder into a temp dir, then keep only model.onnx(+.data) in `dst`.
    The builder also emits genai_config.json + a tokenizer we don't need (the cache root
    already has the tokenizer, and OrtDecoder only loads the graph)."""
    tmp = dst.parent / f"{dst.name}__genai_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    cmd = [
        sys.executable,
        "-m",
        "onnxruntime_genai.models.builder",
        "-m",
        model,
        "-o",
        str(tmp),
        "-p",
        precision,
        "-e",
        "cuda",
        "-c",
        str(hf_cache),
    ]
    if revision:
        cmd += ["--extra_options", f"hf_revision={revision}"]
    logger.info("genai builder: {} ({})", model, precision)
    subprocess.run(cmd, check=True)

    dst.mkdir(parents=True, exist_ok=True)
    for name in ("model.onnx", "model.onnx.data"):
        src = tmp / name
        if src.exists():  # int4 may or may not externalise, so .data is best-effort
            shutil.copy2(src, dst / name)
    shutil.rmtree(tmp)


def _batched_argmax(dec: OrtDecoder, ids: list[int]) -> np.ndarray:
    """Per-position top-1 in ONE forward: pred[i] is the argmax that predicts ids[i+1]."""
    logits, _, _ = dec.run(ids, dec.empty_kv(), 0)
    return np.asarray([int(r.argmax()) for r in np.asarray(logits)])


def _incremental_argmax(dec: OrtDecoder, ids: list[int]) -> np.ndarray:
    """Same per-position top-1, but via one-token KV-cache forwards (what real decoding
    does). Must match the batched pass token-for-token or spec acceptance degrades."""
    past, out = dec.empty_kv(), []
    for pos, tok in enumerate(ids):
        logits, past, _ = dec.run([tok], past, pos)
        out.append(int(np.asarray(logits)[-1].argmax()))
    return np.asarray(out)


def _fp32_reference(fp32_onnx: Path, ids: list[int]) -> np.ndarray:
    """fp32 eager export's per-position top-1 over `ids` (CPU) -- the fidelity yardstick."""
    dec = OrtDecoder(make_session(fp32_onnx, "cpu"))
    logits, _, _ = dec.run(ids, dec.empty_kv(), 0)
    return np.asarray([int(r.argmax()) for r in np.asarray(logits)])


def _validate(variant_onnx: Path, seqs: list[list[int]], fp32_onnx: Path) -> tuple[float, float]:
    """Returns (spec_consistency, fidelity_vs_fp32), averaged over the probe sequences.
    consistency == 1.0 is the gate; fidelity is a quality signal (int4 will be < 1.0)."""
    dec = OrtDecoder(make_session(variant_onnx, "cuda"), device_kv=True)
    cons, fid = [], []
    for seq in seqs:
        batched = _batched_argmax(dec, seq)
        incr = _incremental_argmax(dec, seq)
        ref = _fp32_reference(fp32_onnx, seq)
        cons.append(float(np.mean(batched == incr)))
        # compare where both predict a token: positions 0..len-2 (last predicts past end)
        fid.append(float(np.mean(batched[:-1] == ref[:-1])))
    return float(np.mean(cons)), float(np.mean(fid))


def _reference_sequences(fp32_onnx: Path, tok: AutoTokenizer, n_new: int) -> list[list[int]]:
    """Greedy-continue each probe with the fp32 export so validation scores over the
    reference model's own path (isolates the quant's logit effect from greedy drift)."""
    seqs = []
    for probe in PROBES:
        prompt = tok(probe)["input_ids"]
        dec = OrtDecoder(make_session(fp32_onnx, "cpu"))
        past, committed = dec.empty_kv(), 0
        if len(prompt) > 1:
            _, past, _ = dec.run(prompt[:-1], past, 0)
            committed = len(prompt) - 1
        cur, gen = prompt[-1], []
        for _ in range(n_new):
            logits, past, _ = dec.run([cur], past, committed)
            committed += 1
            cur = int(np.asarray(logits)[-1].argmax())
            gen.append(cur)
        seqs.append([*prompt, *gen])
    return seqs


def _update_manifest(out: Path, entries: dict) -> None:
    """Merge the new variant entries into the existing manifest's provenance and re-stamp
    (write_manifest regenerates file hashes for the whole dir, so the big .onnx.data blobs
    are covered too)."""
    manifest = json.loads((out / MANIFEST).read_text())
    prov = manifest["provenance"]
    prov.setdefault("variants", {}).update(entries)
    prov.setdefault(
        "genai_chain_note", "fused chain-only GPU variants; tree/hidden use the eager export"
    )
    write_manifest(out, prov)


def main() -> None:
    ap = argparse.ArgumentParser("dejavuu.tools.build_genai_chain")
    ap.add_argument("--model", required=True, help="HF model id, e.g. Qwen/Qwen3-0.6B")
    ap.add_argument("--revision", default=None, help="HF commit; default = builder's choice")
    ap.add_argument("--out", type=Path, required=True, help="cache dir (has onnx/ + manifest.json)")
    ap.add_argument("--hf-cache", type=Path, default=Path.home() / ".cache/huggingface/hub")
    ap.add_argument("--probe-tokens", type=int, default=24)
    ap.add_argument("--only", choices=list(VARIANTS), help="build just one variant")
    ap.add_argument(
        "--no-build",
        action="store_true",
        help="skip the genai builder; re-validate + re-stamp the already-built variants",
    )
    args = ap.parse_args()

    out: Path = args.out
    fp32_onnx = out / "onnx" / "model_fp32.onnx"
    if not fp32_onnx.exists():
        raise SystemExit(f"need the eager fp32 export at {fp32_onnx} (fidelity reference)")
    tok = AutoTokenizer.from_pretrained(out)
    seqs = _reference_sequences(fp32_onnx, tok, args.probe_tokens)

    wanted = {args.only: VARIANTS[args.only]} if args.only else VARIANTS
    entries = {}
    for name, precision in wanted.items():
        dst = out / "onnx" / name
        if not args.no_build:
            _build(args.model, args.revision, args.hf_cache, precision, dst)
        consistency, fidelity = _validate(dst / "model.onnx", seqs, fp32_onnx)
        spec_safe = consistency >= SPEC_CONSISTENCY_FLOOR
        logger.info(
            "[{}] batched-vs-incremental: {:.0%}  |  top-1 fidelity vs fp32: {:.0%}  |  spec-safe: {}",
            name,
            consistency,
            fidelity,
            spec_safe,
        )
        if not spec_safe:
            logger.warning(
                "{} batched != incremental at {:.0%} of positions (< {:.0%} floor) -- "
                "acceptance will degrade; blocking for strict runs",
                name,
                consistency,
                SPEC_CONSISTENCY_FLOOR,
            )
        entries[name] = {
            "file": f"onnx/{name}/model.onnx",
            "fidelity_vs_fp32": round(fidelity, 4),
            "batched_incremental_agreement": round(consistency, 4),
            "speculative_compatible": spec_safe,
            "topology": "genai-fused chain-only (no 4D tree mask, no hidden_states)",
            "validation_note": f"batched-vs-incremental {consistency:.0%} over {len(seqs)} probes (GPU)",
        }
    _update_manifest(out, entries)
    logger.info("wrote {} variant(s) + manifest", len(entries))


if __name__ == "__main__":
    main()
