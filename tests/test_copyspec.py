"""CopySpec optimization behavior, tested through the drafter's public interface.

Registry-wide validity and bit-exactness (chain + tree) are already enforced for
CopySpec by tests/test_conformance.py -- losslessness is the verifier's job, so this
test only asserts the behavior the optimization adds: matching a *shorter* k-gram when
the full gamma-gram misses, which recovers copy opportunities on repetitive text a fixed
length would skip. Refs #2.
"""

from dejavuu.drafters.copyspec import CopySpec


def test_matches_shorter_kgram_when_no_full_gamma_match():
    # Suffix [1,2,3] recurs (at index 0), but the full 5-gram [9,9,1,2,3] does not.
    # Fixed gamma=5 would give up and draft nothing; descending match must fire.
    cs = CopySpec()
    prompt = [1, 2, 3, 9, 9, 9, 9, 1, 2, 3]
    cs.reset(prompt)

    draft = cs.propose(prompt, 0, budget=4)

    assert len(draft.token_ids) > 1  # a continuation was proposed
    assert draft.token_ids[1] == 9  # the token that followed [1,2,3] earlier
