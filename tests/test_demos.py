"""Run the modules' `_demo()` self-checks under pytest. These assertions already
exist as `python -m dejavuu.drafters.<x>` sanity checks; this makes CI run them,
covering paths (adapld's semantic fallback, asam's verify-aware sizing) that the
model-free engine tests don't otherwise hit."""

import importlib

import pytest

MODULES = [
    "dejavuu.drafters.adapld",
    "dejavuu.drafters.asam",
    "dejavuu.drafters.pld_plus",
    "dejavuu.drafters.suffix_index",
    "dejavuu.drafters.token_recycling",
]


@pytest.mark.parametrize("mod", MODULES)
def test_module_demo(mod: str):
    importlib.import_module(mod)._demo()
