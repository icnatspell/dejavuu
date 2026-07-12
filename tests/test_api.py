"""Public-API validation wiring: `DejaVu.from_pretrained` must reject bad config before
touching the network or a model. Model-free -- these all raise during validation."""

import pytest
from pydantic import ValidationError

from dejavuu import DejaVu


def test_unknown_method_rejected_before_download():
    with pytest.raises(ValueError, match="unknown method"):
        DejaVu.from_pretrained("any/repo", method="nope")


def test_hf_backend_requires_device():
    with pytest.raises(ValidationError, match="device"):
        DejaVu.from_pretrained("any/repo", backend="hf")


def test_unknown_backend_rejected():
    with pytest.raises(ValidationError):
        DejaVu.from_pretrained("any/repo", backend="tensorrt")
