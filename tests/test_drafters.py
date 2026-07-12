"""Fast, model-free checks of the suffix-match core and retrieval drafters."""

import numpy as np

from dejavuu.core.engine import generate as _generate
from dejavuu.core.verifier import KVCache, Verifier
from dejavuu.drafters import (
    REST,
    DraftTree,
    SAMDecoding,
    SuffixDecoding,
    SuffixIndex,
)


class _Toy(Verifier):
    # next = (token + 1) % V -> repeating cycle, predictable AND repetitive
    V = 5

    def empty_kv(self) -> KVCache:
        return []

    def forward(self, token_ids, past, past_len, position_ids=None, attn_bias=None):
        logits = np.full((len(token_ids), self.V), -9.0, np.float32)
        for i, t in enumerate(token_ids):
            logits[i, (t + 1) % self.V] = 9.0
        return logits, [], None


def test_suffix_index_longest_match_and_boundary():
    idx = SuffixIndex(order=4)
    idx.extend([1, 2, 3, 4])
    idx.append(idx.SEP)
    idx.extend([9, 2, 3, 7])  # "2 3" recurs, with a different continuation
    # longest suffix "2 3" -> most recent follow (7); recency, capped at budget
    assert idx.continuation([5, 2, 3], budget=2) == ([7], 2)
    # frequency: "3" is followed by 4 and by 7 -> tie broken to most recent (7)
    cont, n = idx.continuation([3], budget=1, by_freq=True)
    assert n == 1
    assert cont in ([4], [7])
    # no continuation runs across a SEP boundary
    assert idx.continuation([1, 2, 3, 4], budget=3)[0] == []  # 4 is at end-of-doc


def test_suffix_decoding_drafts_within_prompt():
    sd = SuffixDecoding(min_match=2)
    ctx = [5, 1, 2, 3, 1, 2, 3, 1, 2]  # "1 2" -> "3" repeats
    sd.reset(ctx)
    tree = sd.propose(ctx, len(ctx), budget=4)
    assert tree.token_ids[:2] == [2, 3]  # root=last token, then predicted "3"


def test_cacheback_drafts_a_cached_follower_after_accepted_tokens():
    """Cacheback stores local leader/follower pairs from emitted output, then
    retrieves that follower when the leader recurs."""
    from dejavuu.drafters import Cacheback

    cacheback = Cacheback(leader_len=2, follower_len=2)
    cacheback.reset([])
    cacheback.update([1, 2, 3, 4])

    tree = cacheback.propose([9, 1, 2], past_len=3, budget=4)
    assert tree.token_ids == [2, 3, 4]


def test_cacheback_tree_branches_over_recent_cached_followers():
    from dejavuu.drafters import Cacheback

    cacheback = Cacheback(leader_len=2, follower_len=2, follower_capacity=2)
    cacheback.update([1, 2, 3, 4, 1, 2, 5, 6])

    tree = cacheback.propose_tree([9, 1, 2], past_len=3, budget=4, width=2)
    assert sorted(tree.token_ids[node] for node in tree.children(0)) == [3, 5]


def test_cacheback_evicts_the_least_recently_used_leader():
    from dejavuu.drafters import Cacheback

    cacheback = Cacheback(leader_len=2, follower_len=2, leader_capacity=1)
    cacheback.update([1, 2, 3, 4])
    cacheback.update([5, 6, 7, 8])

    assert cacheback.propose([0, 1, 2], past_len=3, budget=2).token_ids == [2]
    assert cacheback.propose([0, 5, 6], past_len=3, budget=2).token_ids == [6, 7, 8]


def test_cacheback_is_lossless_under_chain_and_tree_verification():
    from dejavuu.drafters import Cacheback

    baseline = _generate(_Toy(), [0], 30)
    for tree in (False, True):
        generated = _generate(_Toy(), [0], 30, Cacheback(leader_len=2, follower_len=2), tree=tree)
        assert generated.tokens == baseline.tokens
        assert generated.accepted > 0


def test_rest_ignores_live_ctx_uses_datastore():
    rest = REST(datastore=[[7, 8, 9, 10, 11]], min_match=2)
    rest.reset([7, 8, 9])
    tree = rest.propose([0, 7, 8, 9], past_len=4, budget=3)
    assert tree.token_ids == [9, 10, 11]  # continuation pulled from the datastore


def test_continuations_branch_top_width_next_tokens():
    idx = SuffixIndex(order=4)
    idx.extend([1, 2, 9])  # "1 2" -> 9
    idx.append(idx.SEP)
    idx.extend([1, 2, 9])  # "1 2" -> 9 again (freq 2)
    idx.append(idx.SEP)
    idx.extend([1, 2, 7])  # "1 2" -> 7 (freq 1)
    conts, n = idx.continuations([0, 1, 2], budget=3, width=2)
    assert n == 2
    firsts = [c[0] for c in conts]
    assert firsts[0] == 9  # most frequent first
    assert set(firsts) == {9, 7}  # both branches present


def test_branches_builds_valid_tree():
    t = DraftTree.branches(5, [[9, 10], [7]], budget=8)
    assert t.token_ids == [5, 9, 10, 7]
    assert t.parent == [-1, 0, 1, 0]  # two chains off the root
    assert all(p < i for i, p in enumerate(t.parent) if p >= 0)  # topological
    # budget caps total guesses across branches
    capped = DraftTree.branches(5, [[1, 2, 3], [4, 5, 6]], budget=2)
    assert len(capped.token_ids) - 1 == 2


def test_sam_picks_longer_match_source():
    # static store has a long match "1 2 3 4"->5; dynamic only a short "4"->9
    sam = SAMDecoding(datastore=[[1, 2, 3, 4, 5]], min_match=2)
    sam.reset([])
    ctx = [8, 4, 9, 1, 2, 3, 4]  # suffix "1 2 3 4" hits static; "4" alone hits dynamic
    sam.dynamic.extend([4, 9])  # seed a competing short dynamic match
    tree = sam.propose(ctx, len(ctx), budget=4)
    assert tree.token_ids[:2] == [4, 5]  # static's longer match wins -> continuation 5


if __name__ == "__main__":
    test_suffix_index_longest_match_and_boundary()
    test_suffix_decoding_drafts_within_prompt()
    test_rest_ignores_live_ctx_uses_datastore()
    test_continuations_branch_top_width_next_tokens()
    test_branches_builds_valid_tree()
    test_sam_picks_longer_match_source()
    print("ok")


def test_anpd_adapts_draft_length():
    from dejavuu.drafters import ANPD

    d = ANPD(init_len=2, max_len=8)
    ctx = [1, 2, 3, 1, 2, 3, 1, 2]
    d.reset(ctx)
    start = d.draft_len
    for _ in range(10):  # repeated full accepts ramp draft_len up to the cap
        d.propose(ctx, 0, budget=16)
        d.update([0] * (d._proposed + 1))
    assert start < d.draft_len <= 8

    d.draft_len = 6
    d.propose(ctx, 0, budget=16)
    d.update([0, 0])  # over-drafted, only 1 accepted -> shrink
    assert d.draft_len < 6


def test_anpd_lossless_matches_baseline():
    from dejavuu.drafters import ANPD

    m = _Toy()
    base = _generate(m, [0], 30)
    anpd = _generate(m, [0], 30, ANPD())
    assert anpd.tokens == base.tokens
    assert anpd.accepted > 0  # repetitive cycle -> n-gram drafts land


def test_lookahead_lossless_and_pools_branches():
    from dejavuu.drafters import Lookahead

    m = _Toy()
    base = _generate(m, [0], 30)
    look = _generate(m, [0], 30, Lookahead())
    assert look.tokens == base.tokens  # lossless
    assert look.accepted > 0

    # pool surfaces multiple next-token candidates as branches
    d = Lookahead(n_max=2)
    ctx = [9, 1, 2, 9, 1, 2, 9, 1, 3, 9, 1]
    tree = d.propose_tree(ctx, 0, budget=4, width=3)
    assert sorted(tree.token_ids[c] for c in tree.children(0)) == [2, 3]


def test_pld_tree_branches_on_forking_ngram():
    from dejavuu.drafters import PLD

    # "1 2" is followed by 9 (older) and by 7 (newer) -> two branches off the root
    ctx = [1, 2, 9, 1, 2, 7, 1, 2]
    tree = PLD(n_min=1, n_max=3).propose_tree(ctx, 0, budget=4, width=2)
    assert sorted(tree.token_ids[c] for c in tree.children(0)) == [7, 9]


def test_asam_tree_branches_from_longer_source():
    from dejavuu.drafters import ASAM

    # dynamic history forks "1 2" -> {9, 7}; tree mode surfaces both as branches
    asam = ASAM(min_match=2)
    asam.reset([])
    ctx = [1, 2, 9, 1, 2, 7, 1, 2]
    tree = asam.propose_tree(ctx, 0, budget=8, width=2)  # room for both branches
    assert sorted(tree.token_ids[c] for c in tree.children(0)) == [7, 9]


def test_logit_spec_uses_the_top_logit_to_retrieve_a_next_next_token_draft():
    """The current verifier logits choose first-token tree branches; each branch
    retrieves the continuation that historically followed that candidate."""
    from dejavuu.drafters import LogitSpec

    d = LogitSpec(k=2, order=2)
    logits = np.full((1, 12), -9.0, np.float32)
    logits[0, 2] = 9.0
    logits[0, 7] = 8.0
    d.observe([5], logits)

    # Earlier [2, 9, 10] and [7, 8] make both logit candidates useful branches.
    ctx = [2, 9, 10, 7, 8, 5]
    assert d.propose(ctx, 0, budget=3).token_ids == [5, 2, 9, 10]

    tree = d.propose_tree(ctx, 0, budget=8, width=2)
    children = {tree.token_ids[c]: c for c in tree.children(0)}
    assert set(children) == {2, 7}
    assert tree.token_ids[tree.children(children[2])[0]] == 9


def test_logit_spec_prefers_logits_cached_for_the_matching_context():
    """The same anchor token can have different likely successors in two contexts.
    A path-specific cache must recover the successor for the matching context rather
    than the most recently observed occurrence of that token."""
    from dejavuu.drafters import LogitSpec

    d = LogitSpec(k=1, order=2)
    first = [1, 5]
    d.propose(first, 0, budget=2)
    logits = np.full((1, 12), -9.0, np.float32)
    logits[0, 2] = 9.0
    d.observe([5], logits)

    second = [9, 5]
    d.propose(second, 0, budget=2)
    logits.fill(-9.0)
    logits[0, 7] = 9.0
    d.observe([5], logits)

    assert d.propose(first, 0, budget=2).token_ids[:2] == [5, 2]
