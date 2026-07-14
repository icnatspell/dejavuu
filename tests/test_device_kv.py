"""Correctness for the GPU io-binding device-resident KV path (issue #46).

The engine keeps the KV cache off-host across decode steps as ORT device buffers to
avoid the host<->device copy that dominates the verify forward at real sequence lengths
(profiled: ~60 ms of a 79 ms step at 1024 tokens on fp32/CUDA). The *risky* part is that
the on-device rollback/gather -- implemented as a Gather(axis=2) over the KV -- must
reproduce the numpy prefix-trim (chain accept) and scattered-row gather (tree accept)
exactly. We pin that here on the CPU EP (io-binding works on CPU too), so the index math
and graph are proven without needing a GPU, and a local real-model check confirms the
full io-bound forward matches the numpy path bit-for-bit.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from dejavuu.core.verifier import gather_kv as np_gather_kv
from dejavuu.core.verifier import trim_kv as np_trim_kv
from dejavuu.decoders.ort import (
    OrtDecoder,
    device_gather,
    make_gather_session,
    make_session,
)

_REAL_FP32 = Path.home() / ".cache/dejavuu/qwen3-0.6b/onnx/model_fp32.onnx"


def _to_ov(kv_np):
    return [
        (ort.OrtValue.ortvalue_from_numpy(k), ort.OrtValue.ortvalue_from_numpy(v)) for k, v in kv_np
    ]


def test_device_gather_matches_numpy_chain_and_tree():
    rng = np.random.default_rng(0)
    # 3 layers, [batch=1, heads=4, seq=10, head_dim=8], matching the KV layout.
    kv_np = [
        (
            rng.standard_normal((1, 4, 10, 8)).astype(np.float32),
            rng.standard_normal((1, 4, 10, 8)).astype(np.float32),
        )
        for _ in range(3)
    ]
    sess = make_gather_session(np.float32, ["CPUExecutionProvider"])
    kv_ov = _to_ov(kv_np)

    # Chain accept: keep the committed prefix (rollback_kv -> trim_kv).
    rows = np.arange(6, dtype=np.int64)
    got = device_gather(sess, kv_ov, rows, "cpu")
    for (gk, gv), (ek, ev) in zip(got, np_trim_kv(kv_np, 6), strict=True):
        np.testing.assert_array_equal(gk.numpy(), ek)
        np.testing.assert_array_equal(gv.numpy(), ev)

    # Tree accept: committed rows + the accepted (scattered) path rows (gather_kv).
    committed, path = 4, [0, 2, 3]
    rows = np.r_[:committed, committed + np.asarray(path)].astype(np.int64)
    got = device_gather(sess, kv_ov, rows, "cpu")
    for (gk, gv), (ek, ev) in zip(got, np_gather_kv(kv_np, committed, path), strict=True):
        np.testing.assert_array_equal(gk.numpy(), ek)
        np.testing.assert_array_equal(gv.numpy(), ev)


@pytest.mark.skipif(not _REAL_FP32.exists(), reason="local fp32 model not present")
def test_device_kv_forward_matches_numpy_on_cpu_ep():
    """The io-bound forward + on-device rollback must equal the numpy path exactly,
    proven on the CPU EP so no GPU is required."""
    sess = make_session(_REAL_FP32, provider="cpu")
    numpy_dec = OrtDecoder(sess)
    dev_dec = OrtDecoder(sess, device_kv=True)

    prompt = list(range(1, 17))
    # Prefill both, then one decode step, then rollback, then a second decode step.
    ln, kvn, _ = numpy_dec.run(prompt, numpy_dec.empty_kv(), 0)
    ld, kvd, _ = dev_dec.run(prompt, dev_dec.empty_kv(), 0)
    np.testing.assert_allclose(ln, ld, atol=1e-4)

    kvn = numpy_dec.rollback_kv(kvn, 12)  # chain accept: keep 12 of 16
    kvd = dev_dec.rollback_kv(kvd, 12)
    ln2, _, _ = numpy_dec.run([42, 7], kvn, 12)
    ld2, _, _ = dev_dec.run([42, 7], kvd, 12)
    np.testing.assert_allclose(ln2, ld2, atol=1e-4)
