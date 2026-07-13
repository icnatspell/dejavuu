"""Public benchmark run-specification behavior."""

import pytest
from pydantic import ValidationError

from dejavuu.eval.config import MeasurementSpec, RunSpec


def test_run_spec_inserts_baseline_and_rejects_unknown_methods_early():
    spec = RunSpec(methods=("pld", "token_recycling"))

    assert spec.methods == ("baseline", "pld", "token_recycling")

    with pytest.raises(ValidationError, match="unknown method"):
        RunSpec(methods=("not-a-drafter",))


def test_cold_model_measurements_cannot_hide_warmups():
    with pytest.raises(ValidationError, match="cold model measurements require warmups=0"):
        MeasurementSpec(model_memory="cold", warmups=1)
