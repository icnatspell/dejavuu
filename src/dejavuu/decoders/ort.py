"""One ONNX decoder, contract auto-derived from the graph (not from per-model config).

`Model` (text) and `VLM` (multimodal) differ only in how they produce the decoder's
primary input -- token ids vs spliced image+text embeds -- and in prefill. The decode
itself is identical, so it lives here once. `OrtDecoder` reads `get_inputs()` /
`get_outputs()` to discover everything the engine needs, so any causal ONNX decoder
that speaks the conventional I/O works drop-in, LLM or VLM:

  primary input    `inputs_embeds` if present else `input_ids`
  attention_mask   fed (all-ones) iff the graph declares it
  position_ids     fed iff declared (stock exports derive positions from the mask)
  past / present   `past_key_values.{i}.key|value` in / `present.{i}.key|value` out,
                   matched **by name** (key/value may be interleaved or grouped) --
                   layer count, kv-head count, head_dim and dtype all read off the
                   past-input shape, so there is no config.json dependency
  tree mode        `supports_tree` iff the graph exposes position_ids AND a rank-4
                   float additive-mask input (no shipped export does -- until a
                   tree-capable decoder is exported, tree falls back to chain)
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import onnxruntime as ort

from dejavuu.core.verifier import KVCache, gather_kv, trim_kv

_PAST = re.compile(r"past_key_values\.(\d+)\.(?:key|value)")
_NP = {"tensor(float)": np.float32, "tensor(float16)": np.float16}
_ONNX_DTYPE = {np.float32: 1, np.float16: 10}  # TensorProto.FLOAT / FLOAT16


def make_gather_session(np_dtype: type, providers: list[str]) -> ort.InferenceSession:
    """A one-node Gather(axis=2) graph that trims/reorders a KV tensor [1,h,seq,d] along
    the sequence axis on-device. Both accept paths reduce to this: chain rollback gathers
    the contiguous committed prefix `arange(len)`, tree accept gathers the committed rows
    plus the accepted path's scattered rows. Kept as its own tiny session because a KV
    prefix in [1,h,seq,d] layout is *not* contiguous (heads are the outer axis), so it
    can't be trimmed by a cheap reshape -- it needs an actual gather kernel."""
    from onnx import TensorProto, helper

    tp = _ONNX_DTYPE[np_dtype]
    data = helper.make_tensor_value_info("data", tp, [1, None, None, None])
    idx = helper.make_tensor_value_info("idx", TensorProto.INT64, [None])
    out = helper.make_tensor_value_info("out", tp, [1, None, None, None])
    node = helper.make_node("Gather", ["data", "idx"], ["out"], axis=2)
    graph = helper.make_graph([node], "kv_gather", [data, idx], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    return ort.InferenceSession(model.SerializeToString(), providers=providers)


def device_gather(
    sess: ort.InferenceSession,
    kv: list[tuple[ort.OrtValue, ort.OrtValue]],
    rows: np.ndarray,
    device: str,
) -> list[tuple[ort.OrtValue, ort.OrtValue]]:
    """Gather `rows` along the seq axis of every layer's (key, value), keeping the result
    device-resident. `kv` values are device OrtValues; the output stays on `device` so it
    feeds straight back as the next step's past with no host round-trip.

    ponytail: 2*n_layers tiny Gather launches per accept -- fine here (the forward is
    launch-bound at ~18 ms; a handful of extra micro-kernels is noise). Batch the layers
    into one gather only if a profile shows this bucket growing."""
    idx = ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(rows, np.int64), device, 0)

    def one(t: ort.OrtValue) -> ort.OrtValue:
        io = sess.io_binding()
        io.bind_ortvalue_input("data", t)
        io.bind_ortvalue_input("idx", idx)
        io.bind_output("out", device)
        sess.run_with_iobinding(io)
        return io.get_outputs()[0]

    return [(one(k), one(v)) for k, v in kv]


def _causal_bias(n: int, past_len: int) -> np.ndarray:
    """Additive [1,1,n,past_len+n]: all past visible, causal among the n new tokens.
    What a tree-capable graph (which *requires* a 4D mask input) needs for a plain
    chain step, so the same decoder serves both chain and tree."""
    bias = np.zeros((1, 1, n, past_len + n), np.float32)
    bias[0, 0, :, past_len:] = np.triu(np.full((n, n), -1e9, np.float32), 1)
    return bias


def make_session(
    path: Path,
    provider: str = "cpu",
    threads: int = 0,
    *,
    allow_provider_fallback: bool = False,
) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
    if provider == "cuda" and hasattr(ort, "preload_dlls"):
        # ONNX Runtime GPU builds load their CUDA/cuDNN libs from the nvidia pip packages
        # at runtime; without this the CUDA provider silently fails to register. No-op on
        # CPU-only builds and cheap to call once per session.
        with contextlib.suppress(Exception):  # missing libs fall back to CPU below
            ort.preload_dlls()
    available = ort.get_available_providers()
    if provider == "cuda" and "CUDAExecutionProvider" not in available:
        if not allow_provider_fallback:
            raise RuntimeError(
                "CUDAExecutionProvider is unavailable; install an ONNX Runtime GPU build "
                "or explicitly allow CPU fallback"
            )
        providers = ["CPUExecutionProvider"]
    else:
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if provider == "cuda"
            else ["CPUExecutionProvider"]
        )
    session = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
    if provider == "cuda" and not allow_provider_fallback:
        actual = session.get_providers()
        if not actual or actual[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"requested CUDAExecutionProvider but session uses {actual}")
    return session


@dataclass
class OrtDecoder:
    session: ort.InferenceSession
    device_kv: bool = False
    """Keep the KV cache as device-resident OrtValues across steps via io-binding, instead
    of returning present KV as host numpy and re-uploading past KV each step. Off by default
    so the CPU/numpy path (and conformance) is byte-identical; enabled for `provider=cuda`
    where the host<->device copy dominates the verify forward (issue #46). Works on the CPU
    EP too (device string follows the session), which is how it's tested without a GPU."""

    def __post_init__(self) -> None:
        """Fail clearly at load if the graph isn't a conventional KV-cache causal
        decoder, instead of a cryptic KeyError mid-generation."""
        if not any(_PAST.match(n) for n in self._in):
            raise ValueError(
                f"no past_key_values.* inputs -- not a KV-cache causal decoder. "
                f"inputs={list(self._in)}"
            )
        if "input_ids" not in self._in and "inputs_embeds" not in self._in:
            raise ValueError(
                f"decoder takes neither input_ids nor inputs_embeds. inputs={list(self._in)}"
            )
        missing = [i for i in range(self.n_layers) if f"present.{i}.key" not in self._out]
        if missing:
            raise ValueError(f"decoder missing present.{{i}}.key/value for layers {missing}")

    @cached_property
    def _in(self) -> dict[str, ort.NodeArg]:
        return {i.name: i for i in self.session.get_inputs()}

    @cached_property
    def _out(self) -> list[str]:
        return [o.name for o in self.session.get_outputs()]

    @cached_property
    def primary(self) -> str:
        return "inputs_embeds" if "inputs_embeds" in self._in else "input_ids"

    @property
    def takes_embeds(self) -> bool:
        return self.primary == "inputs_embeds"

    @cached_property
    def n_layers(self) -> int:
        return 1 + max(int(m.group(1)) for n in self._in if (m := _PAST.match(n)))

    @cached_property
    def _kv(self) -> tuple[int, int, type]:
        """(n_kv_heads, head_dim, dtype) from the past-key input shape [b, h, seq, d]."""
        arg = self._in["past_key_values.0.key"]
        return int(arg.shape[1]), int(arg.shape[3]), _NP.get(arg.type, np.float32)

    @cached_property
    def _device(self) -> str:
        """ORT device string for io-binding: 'cuda' when the session runs on the CUDA
        provider, else 'cpu'. Lets the device_kv path (and its Gather session) run on the
        CPU EP for testing without a GPU."""
        return "cuda" if "CUDAExecutionProvider" in self.session.get_providers() else "cpu"

    @cached_property
    def _gather_sess(self) -> ort.InferenceSession:
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self._device == "cuda"
            else ["CPUExecutionProvider"]
        )
        return make_gather_session(self._kv[2], providers)

    @cached_property
    def _present_idx(self) -> list[tuple[int, int]]:
        return [
            (self._out.index(f"present.{i}.key"), self._out.index(f"present.{i}.value"))
            for i in range(self.n_layers)
        ]

    @cached_property
    def _logits_idx(self) -> int:
        return self._out.index("logits") if "logits" in self._out else 0

    @cached_property
    def _hidden_idx(self) -> int | None:
        """Index of the optional `hidden_states` output -- present only on the
        tree+hidden re-export that feeds representation-aware drafters."""
        return self._out.index("hidden_states") if "hidden_states" in self._out else None

    @cached_property
    def _tree_mask_input(self) -> str | None:
        """A non-KV rank-4 float input is the additive attention bias tree mode needs."""
        for name, a in self._in.items():
            if a.type in _NP and len(a.shape) == 4 and not _PAST.match(name):
                return name
        return None

    @property
    def supports_tree(self) -> bool:
        return self._tree_mask_input is not None and "position_ids" in self._in

    def empty_kv(self) -> KVCache:
        h, d, dt = self._kv
        z = np.zeros((1, h, 0, d), dt)
        if self.device_kv:
            return [
                (
                    ort.OrtValue.ortvalue_from_numpy(z, self._device, 0),
                    ort.OrtValue.ortvalue_from_numpy(z, self._device, 0),
                )
                for _ in range(self.n_layers)
            ]
        return [(z.copy(), z.copy()) for _ in range(self.n_layers)]

    def rollback_kv(self, present: KVCache, committed: int) -> KVCache:
        """Chain accept: keep the first `committed` positions. numpy trim by default;
        on-device Gather when device_kv (delegated to by the Verifier)."""
        if not self.device_kv:
            return trim_kv(present, committed)
        return device_gather(
            self._gather_sess, present, np.arange(committed, dtype=np.int64), self._device
        )

    def gather_kv(self, present: KVCache, committed: int, path: list[int]) -> KVCache:
        """Tree accept: keep committed rows + the accepted path's scattered rows."""
        if not self.device_kv:
            return gather_kv(present, committed, path)
        rows = np.r_[:committed, committed + np.asarray(path)]
        return device_gather(self._gather_sess, present, rows, self._device)

    def run(
        self,
        primary: list[int] | np.ndarray,
        past: KVCache,
        past_len: int,
        position_ids: np.ndarray | None = None,
        attn_bias: np.ndarray | None = None,
    ) -> tuple[np.ndarray, KVCache, np.ndarray | None]:
        """One forward pass. `primary` is token ids (text) or [N, hidden] embeds (VLM),
        per `takes_embeds`. Returns logits[N, vocab], present KV (len past_len+N), and
        hidden states[N, H] if the graph emits them (else None)."""
        if self.device_kv:
            return self._run_bound(primary, past, past_len, position_ids, attn_bias)
        if self.takes_embeds:
            x = np.asarray(primary, np.float32)[None]
        else:
            x = np.asarray([primary], np.int64)
        n = x.shape[1]
        feeds: dict[str, np.ndarray] = {self.primary: x}
        if "attention_mask" in self._in:
            feeds["attention_mask"] = np.ones((1, past_len + n), np.int64)
        if "position_ids" in self._in:
            feeds["position_ids"] = (
                position_ids
                if position_ids is not None
                else np.arange(past_len, past_len + n, dtype=np.int64)[None]
            )
        if self._tree_mask_input is not None:
            # graph requires a 4D mask: feed the tree bias, or a causal one for chains
            bias = attn_bias if attn_bias is not None else _causal_bias(n, past_len)
            feeds[self._tree_mask_input] = bias.astype(np.float32)
        elif attn_bias is not None:
            raise NotImplementedError(
                "decoder has no 4D additive-mask input -- tree attention needs a "
                "re-export with position_ids + a 4D mask (model contract)."
            )
        for i, (k, v) in enumerate(past):
            feeds[f"past_key_values.{i}.key"] = k
            feeds[f"past_key_values.{i}.value"] = v
        outs = self.session.run(None, feeds)
        logits = outs[self._logits_idx][0]
        present = [(outs[ki], outs[vi]) for ki, vi in self._present_idx]
        hidden = outs[self._hidden_idx][0] if self._hidden_idx is not None else None
        return logits, present, hidden

    def _run_bound(
        self,
        primary: list[int] | np.ndarray,
        past: KVCache,
        past_len: int,
        position_ids: np.ndarray | None,
        attn_bias: np.ndarray | None,
    ) -> tuple[np.ndarray, KVCache, np.ndarray | None]:
        """io-bound forward: past KV are device OrtValues bound directly as inputs and
        present KV outputs stay on-device (no host round-trip -- the win at real seq
        lengths). The small inputs (ids/mask/positions/bias) are cheap to upload each step,
        and logits/hidden are pulled back to host because sampling and drafter callbacks
        run there. Numerically identical to `run`; it only moves where the KV lives."""
        if self.takes_embeds:
            x = np.asarray(primary, np.float32)[None]
        else:
            x = np.asarray([primary], np.int64)
        n = x.shape[1]
        io = self.session.io_binding()
        io.bind_cpu_input(self.primary, x)
        if "attention_mask" in self._in:
            io.bind_cpu_input("attention_mask", np.ones((1, past_len + n), np.int64))
        if "position_ids" in self._in:
            pos = (
                position_ids
                if position_ids is not None
                else np.arange(past_len, past_len + n, dtype=np.int64)[None]
            )
            io.bind_cpu_input("position_ids", np.ascontiguousarray(pos, np.int64))
        if self._tree_mask_input is not None:
            bias = attn_bias if attn_bias is not None else _causal_bias(n, past_len)
            io.bind_cpu_input(self._tree_mask_input, np.ascontiguousarray(bias, np.float32))
        elif attn_bias is not None:
            raise NotImplementedError(
                "decoder has no 4D additive-mask input -- tree attention needs a "
                "re-export with position_ids + a 4D mask (model contract)."
            )
        for i, (k, v) in enumerate(past):
            io.bind_ortvalue_input(f"past_key_values.{i}.key", k)
            io.bind_ortvalue_input(f"past_key_values.{i}.value", v)
        for name in self._out:
            # KV stays on device to feed the next step; logits/hidden go to host for
            # sampling and drafter observe().
            io.bind_output(name, self._device if name.startswith("present") else "cpu")
        self.session.run_with_iobinding(io)
        outs = io.get_outputs()
        logits = outs[self._logits_idx].numpy()[0]
        present = [(outs[ki], outs[vi]) for ki, vi in self._present_idx]
        hidden = outs[self._hidden_idx].numpy()[0] if self._hidden_idx is not None else None
        return logits, present, hidden
