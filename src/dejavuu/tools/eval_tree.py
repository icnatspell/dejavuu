"""Benchmark the engine's tree path on real weights -- chain vs tree.

No ONNX decoder honours the tree contract yet (every shipped export has a 2D mask /
no position_ids -- the tree export gate), and a faithful torch.onnx.export with a 4D
additive mask + KV cache fights transformers 5.x's Cache/masking internals. So to
actually *measure* tree's benefits/drawbacks today we run the SmolVLM2 Llama backbone
directly in eager PyTorch as a tree-capable Verifier. This is a benchmarking harness,
NOT the production path (that stays raw onnxruntime); it exists to answer "what does
tree buy, and what does it cost" with numbers instead of theory.

Run (needs the vlm extra):
    uv run --extra vlm python -m dejavuu.tools.eval_tree
"""

from __future__ import annotations

import time

import numpy as np
import torch
from loguru import logger
from transformers import (
    AutoModelForImageTextToText,
    AutoTokenizer,
    DynamicCache,
    LlamaConfig,
    LlamaForCausalLM,
)

from dejavuu.core.engine import generate
from dejavuu.core.verifier import KVCache, Verifier
from dejavuu.decoders.vlm import REPO
from dejavuu.drafters import STAND, AdaPLD, PLDPlus, SuffixDecoding, TokenRecycling


class TorchTreeDecoder(Verifier):
    """Eager-PyTorch tree-capable decoder: honours explicit position_ids (tree depths)
    + a 4D additive mask (siblings isolated) + KV cache, so the engine's tree path runs
    on real weights. fp32 exact attention -> length-invariant, so chain *and* tree stay
    bit-exact vs baseline (unlike the quantized genai ONNX decoder)."""

    def __init__(self, lm: LlamaForCausalLM):
        self.lm = lm.eval()
        self.cfg = lm.config
        self.n_layers = self.cfg.num_hidden_layers
        self.n_kv = self.cfg.num_key_value_heads
        self.hd = getattr(
            self.cfg, "head_dim", self.cfg.hidden_size // self.cfg.num_attention_heads
        )

    @property
    def supports_tree(self) -> bool:
        return True

    def empty_kv(self) -> KVCache:
        z = np.zeros((1, self.n_kv, 0, self.hd), np.float32)
        return [(z.copy(), z.copy()) for _ in range(self.n_layers)]

    @torch.no_grad()
    def forward(self, token_ids, past, past_len, position_ids=None, attn_bias=None):
        n = len(token_ids)
        cache = DynamicCache(config=self.cfg)
        for i, (k, v) in enumerate(past):
            cache.update(torch.from_numpy(k), torch.from_numpy(v), i)
        cpos = torch.arange(past_len, past_len + n)
        pos = torch.from_numpy(position_ids).long() if position_ids is not None else cpos[None]
        mask = torch.from_numpy(attn_bias).float() if attn_bias is not None else None
        out = self.lm(
            input_ids=torch.tensor([token_ids]),
            position_ids=pos,
            attention_mask=mask,  # None -> model builds causal (chain); 4D -> tree
            past_key_values=cache,
            use_cache=True,
            cache_position=cpos,
            output_hidden_states=True,
        )
        present = [
            (cache.layers[i].keys.numpy(), cache.layers[i].values.numpy())
            for i in range(self.n_layers)
        ]
        return out.logits[0].numpy(), present, out.hidden_states[-1][0].numpy()


def _load_backbone() -> LlamaForCausalLM:
    vlm = AutoModelForImageTextToText.from_pretrained(REPO, dtype=torch.float32)
    text = vlm.model.text_model
    cfg = LlamaConfig(**{**text.config.to_dict(), "architectures": ["LlamaForCausalLM"]})
    lm = LlamaForCausalLM(cfg)
    # strict=False tolerates non-persistent buffers (rotary inv_freq); anything else
    # missing/unexpected means the backbones drifted and the body would load garbage
    # weights -- which the losslessness gate can't catch, so surface it loudly.
    info = lm.model.load_state_dict(text.state_dict(), strict=False)
    drift = [k for k in info.missing_keys if "inv_freq" not in k] + list(info.unexpected_keys)
    if drift:
        logger.warning("backbone weight drift -- keys not transferred: {}", drift)
    lm.lm_head.load_state_dict(vlm.lm_head.state_dict())
    return lm


PROMPTS = [
    "The capital of France is Paris. The capital of France is Paris. The capital of",
    "def add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b\n\ndef",
    "one two three four five one two three four five one two three four",
]


def main(max_new: int = 24, budget: int = 8, width: int = 3) -> None:
    model = TorchTreeDecoder(_load_backbone())
    tok = AutoTokenizer.from_pretrained(REPO)
    eos = tok.eos_token_id

    def run(label, make, tree):
        toks = steps = acc = drafted = 0
        dt = 0.0
        lossless = True
        for p in PROMPTS:
            ids = tok(p)["input_ids"]
            base = generate(model, ids, max_new, None, budget, eos)
            d = make()
            d.reset(ids)
            t = time.perf_counter()
            r = generate(model, ids, max_new, d, budget, eos, tree=tree, width=width)
            dt += time.perf_counter() - t
            toks += len(r.tokens)
            steps += r.steps
            acc += r.accepted
            drafted += r.drafted
            lossless &= r.tokens == base.tokens
        logger.info(
            "{:24s} tok/s={:5.1f}  accept_len={:4.2f}  accept%={:4.0%}  steps={:3d}  lossless={}",
            label,
            toks / dt,
            toks / steps,
            acc / drafted,
            steps,
            lossless,
        )

    logger.info(
        "SmolVLM2 Llama backbone (fp32 torch), {} prompts, max_new={}",
        len(PROMPTS),
        max_new,
    )
    run("suffix chain", lambda: SuffixDecoding(min_match=1), False)
    run("suffix tree", lambda: SuffixDecoding(min_match=1), True)
    run("token_recycling chain", lambda: TokenRecycling(), False)
    run("token_recycling tree", lambda: TokenRecycling(), True)
    run("stand chain", lambda: STAND(order=2), False)
    run("stand tree", lambda: STAND(order=2), True)
    run("pld_plus chain", lambda: PLDPlus(), False)
    run("adapld chain", lambda: AdaPLD(), False)
    run("adapld tree", lambda: AdaPLD(), True)


if __name__ == "__main__":
    main()
