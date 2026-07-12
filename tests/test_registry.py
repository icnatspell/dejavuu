"""The drafter registry factory + a couple of stateful-drafter edge cases the
model-free suite doesn't otherwise hit."""

import pytest

from dejavuu.drafters import (
    BASELINE,
    DRAFTERS,
    METHODS,
    REST,
    SAMDecoding,
    SuffixDecoding,
    make_drafter,
    require_method,
)


def test_make_drafter_baseline_is_none():
    assert make_drafter("baseline") is None


def test_make_drafter_plain_ignores_datastore():
    # a non-datastore drafter is constructed bare even if a corpus is passed.
    d = make_drafter("suffix_decoding", datastore=[[1, 2, 3]])
    assert isinstance(d, SuffixDecoding)


def test_make_drafter_feeds_datastore_to_retrieval():
    d = make_drafter("rest", datastore=[[7, 8, 9, 10]])
    assert isinstance(d, REST)
    # the corpus seeded the static index: "8 9" -> continuation exists
    d.reset([7, 8, 9])
    tree = d.propose([0, 7, 8, 9], past_len=4, budget=3)
    assert tree.token_ids[:2] == [9, 10]


def test_every_method_constructs():
    for name in METHODS:
        made = make_drafter(name)
        assert (made is None) == (name == BASELINE)


def test_methods_is_baseline_plus_registry():
    assert METHODS[0] == BASELINE
    assert set(METHODS[1:]) == set(DRAFTERS)
    assert BASELINE not in DRAFTERS  # baseline is not a drafter spec


def test_unknown_method_lists_valid_names():
    with pytest.raises(ValueError, match="unknown method"):
        require_method("nope")
    with pytest.raises(ValueError, match="unknown method"):
        make_drafter("nope")


def test_datastore_capability_comes_from_the_spec():
    # asd is ASAM but deliberately never gets a datastore; rest always does.
    assert DRAFTERS["rest"].needs_datastore is True
    assert DRAFTERS["asd"].needs_datastore is False
    assert DRAFTERS["asam_verify"].kwargs == {"verify_aware": True}


def test_rest_reset_rolls_generation_into_store():
    # REST accumulates the live generation and folds it into the datastore on the next
    # reset (cross-request memory). Feed a generation, reset, then it can be retrieved.
    r = REST(min_match=2)
    r.reset([])
    r.update([4, 5, 6])  # pretend these were generated
    r.reset([])  # rolls [4,5,6] into the static index
    r.reset([0, 4, 5])
    tree = r.propose([0, 4, 5], past_len=3, budget=2)
    assert tree.token_ids[:2] == [5, 6]  # "4 5" -> 6 recovered from the folded history


def test_sam_no_match_falls_back_to_anchor_chain():
    sam = SAMDecoding(min_match=2)
    sam.reset([])
    ctx = [1, 2, 3, 4, 5]  # nothing repeats -> no suffix match >= min_match
    tree = sam.propose(ctx, len(ctx), budget=4)
    assert tree.token_ids == [5]  # just the anchor, no guesses
    assert tree.parent == [-1]
