"""Export a tree-capable ONNX decoder: input_ids + explicit position_ids + a 4D
additive attention mask + KV cache, with eager attention that consumes the 4D mask.

This is the tree export gate. Every shipped export has a 2D mask and no position_ids,
so `OrtDecoder.supports_tree` is False and `--tree` falls back to chain. The genai
builder can't help (its GQA is causal-only). So we go straight to torch.onnx.export
over the SmolVLM2 Llama text backbone -- the same eager forward `tools/eval_tree.py`
already proves lossless -- flattening KV into per-layer tensors so the output speaks
OrtDecoder's conventional I/O (past_key_values.{i}.key|value -> present.{i}.key|value).

Emits fp32 + int8 + (with the build-only `onnx_ir` dep) int4 block-wise weight-only
with the lm-head kept in fp32. Run (int4 needs the extra dep):
    uv run --extra vlm --with onnx_ir --index-strategy unsafe-best-match \
        python -m dejavuu.tools.build_tree_decoder
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import transformers
from loguru import logger
from torch import nn
from transformers import DynamicCache, LlamaForCausalLM

from dejavuu.decoders.vlm import REPO, VLM_TREE_DECODER
from dejavuu.tools.artifact import write_manifest

# reuse the backbone extraction the tree harness already uses
from dejavuu.tools.eval_tree import _load_backbone

TREE_DECODER = Path.home() / ".cache" / "dejavuu" / "smolvlm2_decoder_tree" / "model.onnx"
# VLM tree decoder (VLM_TREE_DECODER) lives in decoders/vlm.py so the runtime can
# prefer it without importing torch; it takes inputs_embeds (vision-splice).

HIDDEN_LAYER = -1  # which hidden-state layer PLD+/AdaPLD rerank on (last, untuned)


class _Wrapper(nn.Module):
    """Flat-tensor I/O around LlamaForCausalLM so torch.onnx.export emits the
    conventional KV-cache names. Honours position_ids + a 4D additive mask."""

    def __init__(self, lm: LlamaForCausalLM, embeds: bool = False):
        super().__init__()
        self.lm = lm.eval()
        self.n = lm.config.num_hidden_layers
        self.embeds = embeds  # VLM variant: take inputs_embeds (vision-splice) not ids

    def forward(self, primary, position_ids, attn_bias, *past):
        cache = DynamicCache(config=self.lm.config)
        for i in range(self.n):
            cache.update(past[2 * i], past[2 * i + 1], i)
        cpos = position_ids[0]
        prim = {"inputs_embeds": primary} if self.embeds else {"input_ids": primary}
        out = self.lm(
            **prim,
            position_ids=position_ids,
            attention_mask=attn_bias,  # 4D additive -> used as-is (tree mask)
            past_key_values=cache,
            use_cache=True,
            cache_position=cpos,
            output_hidden_states=True,
        )
        present = []
        for i in range(self.n):
            present += [cache.layers[i].keys, cache.layers[i].values]
        return (out.logits, out.hidden_states[HIDDEN_LAYER], *present)


def _quant_int4(fp32: Path, out: Path) -> bool:
    """Block-wise weight-only int4 (the same quant onnxruntime-genai's model builder
    applies) on the body, with the lm-head at int8 instead. Keeping the big vocab
    projection out of int4 protects logit/hidden-state fidelity; int8 (not fp32) keeps
    it small -- fp32 would be ~half the file on this small-hidden/large-vocab model. The
    genai builder itself can't be used here -- it rebuilds the graph into a causal-only
    topology with no 4D tree mask and no hidden_states output -- so we run its int4
    quantizer directly on our tree export. Needs the build-only `onnx_ir` dep."""
    import onnx

    try:
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    except ImportError:
        logger.warning(
            "int4 skipped: MatMulNBitsQuantizer needs `onnx_ir` -- re-run with "
            "`uv run --extra vlm --with onnx_ir --index-strategy unsafe-best-match "
            "python -m dejavuu.tools.build_tree_decoder`"
        )
        return False
    from onnxruntime.quantization import QuantType, quantize_dynamic

    lm_head = next(
        n.name
        for n in onnx.load(str(fp32), load_external_data=False).graph.node
        if n.op_type == "MatMul" and "logits" in n.output
    )
    # pass 1: int4 the body, leaving the lm-head as-is (fp32)
    q = MatMulNBitsQuantizer(
        onnx.load(str(fp32)),
        block_size=32,
        is_symmetric=True,
        bits=4,
        nodes_to_exclude=[lm_head],
    )
    q.process()
    tmp = out.parent / "_int4_body.onnx"
    q.model.save_model_to_file(str(tmp), use_external_data_format=False)
    # pass 2: int8 just the lm-head
    quantize_dynamic(tmp, out, weight_type=QuantType.QInt8, nodes_to_quantize=[lm_head])
    tmp.unlink(missing_ok=True)
    logger.info("tree decoder (int4 body + int8 lm-head) -> {}", out)
    return True


def _quantize(out: Path, prim_name: str) -> None:
    """Produce the int8 and (int4 body + int8 lm-head) variants next to fp32 `out`,
    and refresh the manifest. Split out so `--quant-only` can regenerate them from an
    existing fp32 model.onnx without the (slow) torch re-export."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    # weight-only INT8: ~4x smaller, +14% tok/s vs fp32 in the chain-vs-tree e2e,
    # still lossless (tree node logits == causal chain on whatever weights verify).
    int8 = out.parent / "model_int8.onnx"
    quantize_dynamic(out, int8, weight_type=QuantType.QInt8)
    logger.info("tree decoder (int8) -> {}", int8)

    have_int4 = _quant_int4(out, out.parent / "model_int4.onnx")

    manifest = write_manifest(
        out.parent,
        {
            "source_repo": REPO,
            "backbone": "LlamaForCausalLM (SmolVLM2 text_model)",
            "primary_input": prim_name,
            "hidden_states_layer": HIDDEN_LAYER,
            "exporter": "torch.onnx legacy (dynamo=False)",
            "opset": 17,
            "quantization": "int8 dynamic (model_int8.onnx)"
            + ("; int4 body + int8 lm-head (model_int4.onnx)" if have_int4 else ""),
            "transformers": transformers.__version__,
            "torch": torch.__version__,
        },
    )
    logger.info("manifest -> {}", manifest)


def _export(lm: LlamaForCausalLM, out: Path, embeds: bool) -> None:
    cfg = lm.config
    n_layers = cfg.num_hidden_layers
    n_kv = cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    prim_name = "inputs_embeds" if embeds else "input_ids"

    # representative shapes so KV-cat / mask broadcasting trace: past_len=2, N=3
    past_len, nN = 2, 3
    total = past_len + nN
    primary = (
        torch.zeros(1, nN, cfg.hidden_size, dtype=torch.float32)
        if embeds
        else torch.zeros(1, nN, dtype=torch.long)
    )
    position_ids = torch.arange(past_len, total, dtype=torch.long)[None]
    attn_bias = torch.zeros(1, 1, nN, total, dtype=torch.float32)
    past = []
    for _ in range(n_layers):
        z = torch.zeros(1, n_kv, past_len, hd, dtype=torch.float32)
        past += [z, z.clone()]

    past_in = [f"past_key_values.{i}.{kv}" for i in range(n_layers) for kv in ("key", "value")]
    present_out = [f"present.{i}.{kv}" for i in range(n_layers) for kv in ("key", "value")]
    dynamic = {
        prim_name: {1: "N"},
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
        _Wrapper(lm, embeds=embeds),
        (primary, position_ids, attn_bias, *past),
        str(out),
        input_names=[prim_name, "position_ids", "attn_bias", *past_in],
        output_names=["logits", "hidden_states", *present_out],
        dynamic_axes=dynamic,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,  # legacy TorchScript exporter: handles *past varargs + dynamic_axes
    )
    logger.info("tree decoder -> {}", out)
    _quantize(out, prim_name)


def main() -> None:
    ap = argparse.ArgumentParser("dejavuu.tools.build_tree_decoder")
    ap.add_argument(
        "--quant-only",
        action="store_true",
        help="skip the torch re-export; just (re)quantize the existing fp32 model.onnx "
        "into int8 + int4 (fast; needs onnx_ir for int4)",
    )
    args = ap.parse_args()
    if args.quant_only:
        _quantize(TREE_DECODER, "input_ids")  # text decoder
        _quantize(VLM_TREE_DECODER, "inputs_embeds")  # VLM (vision-splice) decoder
        return
    lm = _load_backbone()
    lm.config._attn_implementation = "eager"  # 4D mask consumed directly
    _export(lm, TREE_DECODER, embeds=False)  # text: input_ids
    _export(lm, VLM_TREE_DECODER, embeds=True)  # VLM: inputs_embeds (vision-splice)


if __name__ == "__main__":
    main()
