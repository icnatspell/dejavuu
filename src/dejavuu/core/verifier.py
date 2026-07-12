"""The contract the generation engine depends on -- nothing model-specific.

Any decoder (text LLM or VLM) that can expose these three operations is
speculatively decodable. The LLM (model.py) feeds input_ids to one graph; a VLM
(vlm.py) embeds tokens and seeds the KV with image features during prefill. The
engine never knows the difference.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

# The numpy default cache: a list of (key, value) arrays, one pair per layer. Backends
# that keep their KV elsewhere (a torch DynamicCache, ORT device buffers) use their own
# opaque type and override rollback_kv/gather_kv; the engine never inspects it.
KVCache = list[tuple[np.ndarray, np.ndarray]]


def trim_kv(kv: KVCache, length: int) -> KVCache:
    """Keep the first `length` positions along the seq axis (numpy chain rollback)."""
    return [(k[:, :, :length, :], v[:, :, :length, :]) for k, v in kv]


def gather_kv(kv: KVCache, committed: int, path: list[int]) -> KVCache:
    """Keep the committed rows + the accepted path's (scattered) tree rows, gathered into
    contiguous slots (numpy tree rollback). New length == committed + len(path)."""
    rows = np.r_[:committed, committed + np.asarray(path)]
    return [(k[:, :, rows, :], v[:, :, rows, :]) for k, v in kv]


class Verifier(ABC):
    @abstractmethod
    def empty_kv(self) -> KVCache: ...

    def rollback_kv(self, kv: Any, committed: int) -> Any:
        """Chain accept: trim the KV to the `committed` accepted positions. The default is
        numpy; a backend whose KV isn't a numpy list (HF's torch cache) overrides this."""
        return trim_kv(kv, committed)

    def gather_kv(self, kv: Any, committed: int, path: list[int]) -> Any:
        """Tree accept: keep committed rows + the accepted path's scattered tree rows.
        Numpy default; a torch/other backend overrides. `path` is the accepted node
        indices (incl. root); new length is committed + len(path)."""
        return gather_kv(kv, committed, path)

    @property
    def supports_tree(self) -> bool:
        """True iff `forward` honours explicit `position_ids` + a 4D additive
        `attn_bias` -- the contract tree verification needs. Default False: the
        stock 2D-mask exports can't express tree attention, so engine(tree=True)
        falls back to chain. A tree-capable re-export flips this on."""
        return False

    @abstractmethod
    def forward(
        self,
        token_ids: list[int],
        past: KVCache,
        past_len: int,
        position_ids: np.ndarray | None = None,
        attn_bias: np.ndarray | None = None,
    ) -> tuple[np.ndarray, KVCache, np.ndarray | None]:
        """logits[N, vocab], present KV (length past_len+N), and per-token hidden
        states[N, H] (or None if the decoder does not emit them) for N input tokens.
        Chain path: position_ids/attn_bias are None and positions are the contiguous
        past_len..past_len+N-1 (2D causal mask). Tree path (only when supports_tree):
        explicit `position_ids` [1,N] and additive `attn_bias` [1,1,N,past_len+N].
        The hidden states are a side channel for representation-aware drafters
        (PLD+, AdaPLD); they never affect verification/losslessness."""

    def prefill(self, prompt_ids: list[int]) -> tuple[KVCache, int]:
        """Seed the KV from the prompt, leaving prompt[-1] as the uncommitted
        anchor. Returns (past, committed_len). VLMs override to inject image
        features; the default is plain text prefill."""
        if len(prompt_ids) <= 1:
            return self.empty_kv(), 0
        _, past, _ = self.forward(prompt_ids[:-1], self.empty_kv(), 0)
        return past, len(prompt_ids) - 1

    @property
    def is_vlm(self) -> bool:
        """True for a multimodal backend that consumes images via `prepare`. Text
        backends leave this False; the API uses it to route encoding."""
        return False

    def prepare(self, processor_out: dict) -> list[int]:
        """Splice vision features into the prompt embeds and stash them for `prefill`,
        returning the token ids. VLM backends override this; text backends can't."""
        raise NotImplementedError(f"{type(self).__name__} is text-only; no image support")
