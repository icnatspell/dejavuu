"""Lossless gate for the transformers backend: every drafter must be bit-exact with the
plain autoregressive baseline under both chain AND tree verification, run through a real
HF model.

A tiny random Llama built from config (no download, no network) is a perfect lossless
oracle -- the weights are arbitrary, but baseline and speculative decode use the same
model greedily, so the accept rule guarantees identical output. Needs the `hf` extra;
skipped when torch/transformers aren't installed.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from dejavuu.core.engine import generate  # noqa: E402
from dejavuu.drafters import DRAFTERS, make_drafter  # noqa: E402

PROMPT = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2]  # repetitive so the drafters fire
NAMES = list(DRAFTERS)  # every registered drafter (baseline is not one)


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    """A tiny random Llama saved to disk once; backends load from it per attn impl."""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    torch.manual_seed(0)
    d = tmp_path_factory.mktemp("tiny_llama")
    LlamaForCausalLM(cfg).save_pretrained(d)
    return str(d)


@pytest.fixture(scope="module")
def hf_model(tiny_dir):
    from dejavuu.decoders.hf import HFBackend

    return HFBackend(tiny_dir, device="cpu")


@pytest.fixture(scope="module")
def baseline(hf_model) -> list[int]:
    return generate(hf_model, PROMPT, 24).tokens


@pytest.mark.parametrize("use_tree", [False, True], ids=["chain", "tree"])
@pytest.mark.parametrize("name", NAMES)
def test_hf_lossless(hf_model, name: str, use_tree: bool, baseline: list[int]):
    """Every drafter reproduces the baseline exactly under chain and tree verification
    through the HF backend. Datastore-backed drafters get the baseline continuation."""
    store = [PROMPT + baseline]
    out = generate(
        hf_model, PROMPT, 24, drafter=make_drafter(name, datastore=store), tree=use_tree
    ).tokens
    assert out == baseline


def test_hf_supports_tree(hf_model):
    assert hf_model.supports_tree is True


def test_hf_tree_sampled_stand_matches_baseline(hf_model):
    """Gumbel-ranked STAND branches remain lossless under real tree attention."""
    from dejavuu.core.sampling import Sampler

    sampler = Sampler(temperature=0.8, seed=7)
    base = generate(hf_model, PROMPT, 24, sampler=sampler).tokens
    out = generate(
        hf_model, PROMPT, 24, drafter=make_drafter("stand"), tree=True, width=4, sampler=sampler
    ).tokens
    assert out == base


@pytest.mark.parametrize("use_tree", [False, True], ids=["chain", "tree"])
def test_hf_sdpa_lossless(tiny_dir, use_tree: bool):
    """SDPA (the GPU-perf attention kernel) must honour the 4D tree mask and stay
    bit-exact -- so opting into it for speed never costs correctness. Compared against
    the SDPA backend's own greedy baseline."""
    from dejavuu.decoders.hf import HFBackend

    model = HFBackend(tiny_dir, device="cpu", attn_implementation="sdpa")
    base = generate(model, PROMPT, 24).tokens
    for name in ("pld", "suffix_decoding", "token_recycling"):
        out = generate(model, PROMPT, 24, drafter=make_drafter(name), tree=use_tree).tokens
        assert out == base, name


@pytest.mark.model
@pytest.mark.parametrize("use_tree", [False, True], ids=["chain", "tree"])
def test_hf_real_model_lossless(use_tree: bool):
    """Real weights (SmolLM2-135M: trained Llama + GQA + RoPE) stay bit-exact under
    chain and tree, including the hidden-state drafter. The random tiny model can't
    exercise realistic attention scales or GQA; this does. Opt-in (downloads ~270MB)."""
    from transformers import AutoTokenizer

    from dejavuu.decoders.hf import HFBackend

    mid = "HuggingFaceTB/SmolLM2-135M"
    prompt = AutoTokenizer.from_pretrained(mid)("The quick brown fox jumps over the lazy")[
        "input_ids"
    ]
    model = HFBackend(mid, device="cpu")
    base = generate(model, prompt, 32).tokens
    for name in ("pld", "suffix_decoding", "token_recycling", "adapld", "stand"):
        out = generate(model, prompt, 32, drafter=make_drafter(name), tree=use_tree).tokens
        assert out == base, name
