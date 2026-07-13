"""Convert + quantize any conventional HF causal LM into a tree-capable ONNX decoder.

The export speaks `OrtDecoder`'s I/O so the runtime gets everything the drafters can
use, on *any* conventional causal LM (Qwen3, Llama, Mistral, ...), not just the SmolVLM
backbone that `build_tree_decoder` targets:

    input_ids + position_ids + 4D additive mask + past KV  ->  logits + hidden_states + present KV

That single graph gives the engine both chain **and** tree verification (the 4D mask)
plus per-token hidden states (so PLD+/AdaPLD rerank instead of silently degrading to
plain PLD). `OrtDecoder` auto-detects all of it from the graph I/O -- no runtime wiring.

Three variants land in `<out>/onnx/`, named to match `decoders.text.Model`
(`model_fp32.onnx`, `model_int8.onnx`, `model_q4.onnx`), alongside the tokenizer:
  - fp32          -- reference, exported with eager attention so the 4D mask is honoured.
  - int8          -- dynamic weight-only, ~4x smaller.
  - q4 (int4)     -- block-wise weight-only body + int8 lm-head (the big vocab
                     projection stays int8 to protect logit/hidden fidelity).

Every variant is then **validated** with a teacher-forced fidelity gate: over the fp32
greedy sequence, it scores per-position top-1 agreement against the torch fp32 reference
(given the same prefix, does the variant predict the same next token?). That isolates
the quant's logit effect from greedy divergence -- so it doesn't cry wolf when a
perfectly coherent quant simply takes a different-but-valid path -- and it predicts the
speculative match% the variant will show. A free-running decode is also printed so
incoherent garbage stays human-visible. This is quant *fidelity vs fp32*; it is NOT the
benchmark's spec-vs-same-variant losslessness (judged in the eval harness, never vs fp32).

    uv run --extra vlm --with onnx_ir --index-strategy unsafe-best-match \
        python -m dejavuu.tools.build_decoder --model Qwen/Qwen3-0.6B \
        --out ~/.cache/dejavuu/qwen3-0.6b
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import transformers
from huggingface_hub import HfApi
from loguru import logger
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from dejavuu.decoders.ort import OrtDecoder, make_session
from dejavuu.tools.artifact import write_manifest

# Which hidden-state layer PLD+/AdaPLD rerank on. -1 (final) matches build_tree_decoder;
# untuned -- a representation-aware drafter can pick a different layer if it exports more.
HIDDEN_LAYER = -1
OPSET = 17

# Teacher-forced per-position top-1 agreement floors vs torch fp32 (see validate()).
# fp32 below its floor means the *export* is broken (wrong mask/positions/KV) -> hard
# error. int8/q4 below theirs means the quant shifted logits too far -> loud warning.
FIDELITY_FLOOR = {"fp32": 0.98, "int8": 0.85, "q4": 0.70}


@dataclass
class Variant:
    name: str  # fp32 / int8 / q4
    path: Path
    fidelity: float | None = None  # top-1 agreement vs torch fp32 over the probe
    sequence_length_agreement: float | None = None
    speculative_compatible: bool | None = None


class _Wrapper(nn.Module):
    """Flat-tensor I/O around a `*ForCausalLM` so `torch.onnx.export` emits the
    conventional KV-cache names, honouring explicit `position_ids` + a 4D additive
    attention mask (the tree mask) and returning the chosen hidden-state layer."""

    def __init__(self, lm: nn.Module) -> None:
        super().__init__()
        self.lm = lm.eval()
        self.n = lm.config.num_hidden_layers

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_bias: torch.Tensor,
        *past: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        cache = DynamicCache(config=self.lm.config)
        for i in range(self.n):
            cache.update(past[2 * i], past[2 * i + 1], i)
        out = self.lm(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attn_bias,  # 4D additive -> consumed as-is by eager attention
            past_key_values=cache,
            use_cache=True,
            cache_position=position_ids[0],
            output_hidden_states=True,
        )
        present: list[torch.Tensor] = []
        for i in range(self.n):
            present += [cache.layers[i].keys, cache.layers[i].values]
        return (out.logits, out.hidden_states[HIDDEN_LAYER], *present)


def load_lm(model_id: str, revision: str) -> nn.Module:
    """Load a conventional causal LM with eager attention (required so the exported
    graph consumes our 4D additive mask directly rather than rebuilding a causal one)."""
    lm = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    lm.config._attn_implementation = "eager"
    return lm.eval()


def export_fp32(lm: nn.Module, out: Path) -> None:
    """torch.onnx.export the tree+hidden decoder to `out` (fp32)."""
    cfg = lm.config
    n_layers = cfg.num_hidden_layers
    n_kv = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

    past_len, n_new = 2, 3  # representative shapes so KV-cat / mask broadcasting trace
    total = past_len + n_new
    input_ids = torch.zeros(1, n_new, dtype=torch.long)
    position_ids = torch.arange(past_len, total, dtype=torch.long)[None]
    attn_bias = torch.zeros(1, 1, n_new, total, dtype=torch.float32)
    past = [
        torch.zeros(1, n_kv, past_len, head_dim, dtype=torch.float32) for _ in range(2 * n_layers)
    ]

    past_in = [f"past_key_values.{i}.{kv}" for i in range(n_layers) for kv in ("key", "value")]
    present_out = [f"present.{i}.{kv}" for i in range(n_layers) for kv in ("key", "value")]
    dynamic: dict[str, dict[int, str]] = {
        "input_ids": {1: "N"},
        "position_ids": {1: "N"},
        "attn_bias": {2: "N", 3: "total"},
        "logits": {1: "N"},
        "hidden_states": {1: "N"},
    }
    for name in past_in:
        dynamic[name] = {2: "past"}
    for name in present_out:
        dynamic[name] = {2: "total"}

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        _Wrapper(lm),
        (input_ids, position_ids, attn_bias, *past),
        str(out),
        input_names=["input_ids", "position_ids", "attn_bias", *past_in],
        output_names=["logits", "hidden_states", *present_out],
        dynamic_axes=dynamic,
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,  # legacy exporter: handles *past varargs + dynamic_axes cleanly
    )
    logger.info("fp32 tree+hidden decoder -> {}", out)


def quantize_int8(fp32: Path, out: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(fp32, out, weight_type=QuantType.QInt8)
    logger.info("int8 (dynamic weight-only) -> {}", out)


def quantize_q4(fp32: Path, out: Path) -> bool:
    """Block-wise int4 body + int8 lm-head. Keeping the vocab projection out of int4
    protects logit/hidden fidelity. Needs the build-only `onnx_ir` dep for the int4
    quantizer; returns False (skips) if it is missing."""
    import onnx

    try:
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    except ImportError:
        logger.warning(
            "q4 skipped: MatMulNBitsQuantizer needs `onnx_ir` (re-run with --with onnx_ir)"
        )
        return False
    from onnxruntime.quantization import QuantType, quantize_dynamic

    lm_head = next(
        n.name
        for n in onnx.load(str(fp32), load_external_data=False).graph.node
        if n.op_type == "MatMul" and "logits" in n.output
    )
    q = MatMulNBitsQuantizer(
        onnx.load(str(fp32)), block_size=32, is_symmetric=True, bits=4, nodes_to_exclude=[lm_head]
    )
    q.process()
    tmp = out.parent / "_q4_body.onnx"
    q.model.save_model_to_file(str(tmp), use_external_data_format=False)
    quantize_dynamic(tmp, out, weight_type=QuantType.QInt8, nodes_to_quantize=[lm_head])
    tmp.unlink(missing_ok=True)
    logger.info("q4 (int4 body + int8 lm-head) -> {}", out)
    return True


def _greedy_onnx(path: Path, prompt_ids: list[int], n_new: int, threads: int) -> list[int]:
    """Greedy-decode `n_new` tokens through the ONNX decoder in chain mode (mirrors the
    engine's baseline: no drafts, one token per forward)."""
    dec = OrtDecoder(make_session(path, "cpu", threads))
    past = dec.empty_kv()
    committed = 0
    if len(prompt_ids) > 1:
        _, past, _ = dec.run(prompt_ids[:-1], past, 0)
        committed = len(prompt_ids) - 1
    tok = prompt_ids[-1]
    out: list[int] = []
    for _ in range(n_new):
        logits, past, _ = dec.run([tok], past, committed)
        committed += 1
        tok = int(np.asarray(logits)[-1].argmax())
        out.append(tok)
    return out


def _greedy_torch(lm: nn.Module, prompt_ids: list[int], n_new: int) -> list[int]:
    with torch.no_grad():
        gen = lm.generate(
            torch.tensor([prompt_ids]),
            max_new_tokens=n_new,
            do_sample=False,
            use_cache=True,
            pad_token_id=lm.config.eos_token_id,
        )
    return gen[0, len(prompt_ids) :].tolist()


def _next_token_argmax_onnx(path: Path, ids: list[int], threads: int) -> list[int]:
    """Per-position top-1 over `ids` in a single chain forward: pred[i] predicts ids[i+1]."""
    dec = OrtDecoder(make_session(path, "cpu", threads))
    logits, _, _ = dec.run(ids, dec.empty_kv(), 0)
    return [int(row.argmax()) for row in np.asarray(logits)]


def _incremental_argmax_onnx(path: Path, ids: list[int], threads: int) -> list[int]:
    """Next-token argmax for each position using one-token KV-cache forwards.

    Speculative verification submits several tokens at once. Those batched causal
    logits must select the same tokens as incremental decoding from the same graph;
    some aggressive quantizations violate that invariant even when both executions
    are individually well-formed.
    """
    dec = OrtDecoder(make_session(path, "cpu", threads))
    past = dec.empty_kv()
    predictions: list[int] = []
    for position, token_id in enumerate(ids):
        logits, past, _ = dec.run([token_id], past, position)
        predictions.append(int(np.asarray(logits)[-1].argmax()))
    return predictions


def _next_token_argmax_torch(lm: nn.Module, ids: list[int]) -> list[int]:
    with torch.no_grad():
        out = lm(torch.tensor([ids]))
    return [int(row.argmax()) for row in out.logits[0]]


def validate(
    variant: Variant, lm: nn.Module, tok: AutoTokenizer, probe: str, n_new: int, threads: int
) -> None:
    """Teacher-forced fidelity gate. Over the fp32 greedy reference sequence, score
    per-position top-1 agreement between the ONNX variant and torch fp32 -- i.e. given
    the *same* prefix, does the variant predict the same next token? This isolates the
    quant's logit effect from greedy divergence (one shifted token cascades into a
    different-but-valid continuation, so free-running overlap wildly understates quality)
    and it is exactly what predicts speculative match% against a same-variant baseline.

    A free-running decode is also printed purely so incoherent 'garbage' is human-visible.
    fp32 below floor = broken export (raises); a quant below floor = degraded quant (warns).
    """
    prompt_ids = tok(probe)["input_ids"]
    seq = [*prompt_ids, *_greedy_torch(lm, prompt_ids, n_new)]  # score over fp32's own path
    ref = _next_token_argmax_torch(lm, seq)
    got = _next_token_argmax_onnx(variant.path, seq, threads)
    incremental = _incremental_argmax_onnx(variant.path, seq, threads)
    start = len(prompt_ids) - 1  # first position whose prediction is a generated token
    pairs = list(zip(ref[start:-1], got[start:-1], strict=False))
    agree = float(np.mean([a == b for a, b in pairs])) if pairs else 0.0
    variant.fidelity = agree
    consistency = float(np.mean(np.asarray(got) == np.asarray(incremental)))
    variant.sequence_length_agreement = consistency
    variant.speculative_compatible = consistency == 1.0

    free_run = _greedy_onnx(variant.path, prompt_ids, n_new, threads)
    text = tok.decode(free_run, skip_special_tokens=True).replace("\n", " ")
    logger.info(
        "[{}] top-1 vs fp32: {:.0%}  |  batched-vs-incremental: {:.0%}  |  free-run: {!r}",
        variant.name,
        agree,
        consistency,
        text[:70],
    )
    if not variant.speculative_compatible:
        message = (
            f"{variant.name} batched causal predictions agree with incremental decoding "
            f"at only {consistency:.0%} of probe positions"
        )
        if variant.name == "fp32":
            raise RuntimeError(f"broken export: {message}")
        logger.warning("{} -- artifact will be blocked for strict speculative runs", message)
    floor = FIDELITY_FLOOR[variant.name]
    if agree < floor:
        msg = f"{variant.name} teacher-forced fidelity {agree:.0%} < floor {floor:.0%}"
        if variant.name == "fp32":
            raise RuntimeError(f"broken export: {msg} (mask / position_ids / KV wrong)")
        logger.warning("{} -- quant shifted logits further than expected; inspect before use", msg)


def main() -> None:
    ap = argparse.ArgumentParser("dejavuu.tools.build_decoder")
    ap.add_argument("--model", required=True, help="HF model id, e.g. Qwen/Qwen3-0.6B")
    ap.add_argument("--revision", default=None, help="HF commit; default resolves current commit")
    ap.add_argument("--out", type=Path, required=True, help="output dir; ONNX lands in <out>/onnx/")
    ap.add_argument("--quant", choices=["int8", "q4", "both", "none"], default="both")
    ap.add_argument("--no-validate", action="store_true", help="skip the generation fidelity gate")
    ap.add_argument(
        "--probe", default="The capital of France is", help="prompt for the fidelity gate"
    )
    ap.add_argument("--probe-tokens", type=int, default=32)
    ap.add_argument(
        "--threads", type=int, default=0, help="ORT intra-op threads for validation (0=default)"
    )
    args = ap.parse_args()

    onnx_dir = args.out / "onnx"
    revision = args.revision or HfApi().model_info(args.model).sha
    logger.info("loading {}@{} (eager attention)", args.model, revision)
    lm = load_lm(args.model, revision)
    tok = AutoTokenizer.from_pretrained(args.model, revision=revision)

    fp32 = onnx_dir / "model_fp32.onnx"
    export_fp32(lm, fp32)
    variants = [Variant("fp32", fp32)]
    if args.quant in ("int8", "both"):
        int8 = onnx_dir / "model_int8.onnx"
        quantize_int8(fp32, int8)
        variants.append(Variant("int8", int8))
    if args.quant in ("q4", "both") and quantize_q4(fp32, onnx_dir / "model_q4.onnx"):
        variants.append(Variant("q4", onnx_dir / "model_q4.onnx"))

    tok.save_pretrained(args.out)  # so decoders.text.Model + AutoTokenizer load from <out>
    logger.info("tokenizer -> {}", args.out)

    if not args.no_validate:
        for v in variants:
            validate(v, lm, tok, args.probe, args.probe_tokens, args.threads)

    write_manifest(
        args.out,
        {
            "model_kind": "text_onnx",
            "source_model": args.model,
            "source_revision": revision,
            "architecture": type(lm).__name__,
            "primary_input": "input_ids",
            "hidden_states_layer": HIDDEN_LAYER,
            "opset": OPSET,
            "exporter": "torch.onnx legacy (dynamo=False)",
            "variants": {
                v.name: {
                    "file": v.path.relative_to(args.out).as_posix(),
                    "fidelity_vs_fp32": v.fidelity,
                    "sequence_length_agreement": v.sequence_length_agreement,
                    "speculative_compatible": v.speculative_compatible,
                }
                for v in variants
            },
            "transformers": transformers.__version__,
            "torch": torch.__version__,
        },
    )
    logger.info("done: {} variant(s) in {}", len(variants), onnx_dir)


if __name__ == "__main__":
    main()
