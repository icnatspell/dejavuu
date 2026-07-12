"""Retrieval-based speculative decoding on raw onnxruntime.

Layout: `core/` (Verifier contract + engine), `decoders/` (OrtDecoder + Model/VLM),
`drafters/` (the method zoo), `eval/` (Spec-Bench / MMSpec harnesses), `tools/`
(build + benchmark scripts). `api.DejaVu` is the drop-in entry point.
"""

from dejavuu.api import DejaVu

__all__ = ["DejaVu"]
