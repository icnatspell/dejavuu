"""Hugging Face transformers as a Verifier backend.

Any `AutoModelForCausalLM` drives the same speculative-decoding engine with no ONNX
export -- the easy on-ramp: point it at a model id and generate. transformers owns the
forward pass and the KV cache (a torch `DynamicCache`); everything else (drafters,
draft construction, the accept rule, sampling) is the shared backend-agnostic core.

Chain and tree verification, both bit-exact with greedy decoding (the accept rule owns
correctness). Tree needs no re-export: eager attention honours the engine's 4D additive
mask + explicit position_ids, so `supports_tree` is True on any causal LM.

Needs the optional `hf` extra (torch). `attn_implementation` defaults to "eager" (the
mask plumbing is deterministic across models); "sdpa" also respects the mask and is
faster on GPU (validated lossless in tests). flash-attention can't take an arbitrary
mask, so it can't serve tree mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dejavuu.core.verifier import Verifier


@dataclass
class HFBackend(Verifier):
    """Wrap a transformers causal LM as a `Verifier`. `device` is required (no
    cuda-vs-cpu guessing); `dtype` is a torch dtype name (e.g. "bfloat16") or None for
    the model's native dtype."""

    model_id: str
    device: str
    dtype: str | None = None
    # "eager" is the safe default: it honours a custom 4D tree mask on every model.
    # "sdpa" is faster on GPU and also respects the mask (validated lossless in tests),
    # so opt into it for perf. flash-attention can't take an arbitrary mask, so it can't
    # serve tree mode.
    attn_implementation: str = "eager"
    emit_hidden: bool = True  # last-layer hidden states for PLD+/AdaPLD (free side channel)
    _model: Any = field(default=None, init=False, repr=False)
    _torch: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        self._torch = torch
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, attn_implementation=self.attn_implementation
        )
        if self.dtype is not None:
            model = model.to(getattr(torch, self.dtype))
        self._model = model.to(self.device).eval()

    @property
    def supports_tree(self) -> bool:
        # eager attention honours a custom 4D additive mask + explicit position_ids, so
        # tree verification works on any HF causal LM with no re-export.
        return True

    def empty_kv(self) -> Any:
        from transformers import DynamicCache

        return DynamicCache()

    def forward(
        self,
        token_ids: list[int],
        past: Any,
        past_len: int,
        position_ids: np.ndarray | None = None,
        attn_bias: np.ndarray | None = None,
    ) -> tuple[np.ndarray, Any, np.ndarray | None]:
        """One forward pass through the HF model. Returns logits[N, vocab] and (optional)
        last-layer hidden[N, H] as numpy, plus the updated torch KV cache. The per-step
        logits/hidden cross to numpy here so the accept rule and seeded sampler stay one
        backend-agnostic implementation.

        Chain: a 2D all-ones mask, HF derives the causal structure from position_ids.
        Tree: the engine's rank-4 additive bias (0 visible / large-negative blocked) and
        explicit position_ids (siblings share a position) go straight through."""
        torch = self._torch
        n = len(token_ids)
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        if position_ids is not None:
            pos = torch.as_tensor(position_ids, dtype=torch.long, device=self.device)
        else:
            pos = torch.arange(past_len, past_len + n, device=self.device).unsqueeze(0)
        if attn_bias is not None:
            # Remap the engine's -1e9 to the model dtype's min so the softmax zeros
            # masked entries exactly, in any precision (fp16/bf16 included).
            b = torch.as_tensor(attn_bias, device=self.device)
            mask: Any = torch.zeros_like(b, dtype=self._model.dtype)
            mask.masked_fill_(b < 0, torch.finfo(self._model.dtype).min)
        else:
            mask = torch.ones((1, past_len + n), dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self._model(
                input_ids=input_ids,
                past_key_values=past,
                position_ids=pos,
                attention_mask=mask,
                use_cache=True,
                output_hidden_states=self.emit_hidden,
            )
        logits = out.logits[0].float().cpu().numpy()
        hidden = out.hidden_states[-1][0].float().cpu().numpy() if self.emit_hidden else None
        return logits, out.past_key_values, hidden

    def rollback_kv(self, kv: Any, committed: int) -> Any:
        """Chain accept: truncate the torch KV cache to `committed` positions in place."""
        kv.crop(committed)
        return kv

    def gather_kv(self, kv: Any, committed: int, path: list[int]) -> Any:
        """Tree accept: keep committed rows + the accepted path's scattered rows, per
        layer, along the KV sequence axis. `path` is the accepted node indices (incl.
        root) into this step's draft; new length is committed + len(path)."""
        torch = self._torch
        rows = torch.tensor(
            [*range(committed), *(committed + p for p in path)],
            dtype=torch.long,
            device=self.device,
        )
        if hasattr(kv, "layers"):  # transformers >= ~4.54 (DynamicLayer.keys/.values)
            for layer in kv.layers:
                layer.keys = layer.keys.index_select(2, rows)
                layer.values = layer.values.index_select(2, rows)
        else:  # legacy key_cache/value_cache lists
            for i in range(len(kv.key_cache)):
                kv.key_cache[i] = kv.key_cache[i].index_select(2, rows)
                kv.value_cache[i] = kv.value_cache[i].index_select(2, rows)
        return kv
