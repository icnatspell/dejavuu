"""AdaPLD -- adaptive PLD+ (arXiv 2606.05742): the current SOTA in this family.

Extends PLD+ with (a) a semantic fallback -- when no n-gram matches, retrieve by
hidden-state similarity over the run's own history, fixing PLD's "no-hit" failure --
and (b) a branched draft tree: the reranked main copy path plus top-K next-token
branches from the target's own logits (Token Recycling style), each extended by one
hidden-reranked successor token, verified in one pass with tree attention.

Chain mode returns just the reranked main copy path (~ PLD+). Needs a hidden-emitting
decoder; degrades to PLD without one. Tree branches run only when the decoder
supports tree verification. Lossless -- the verifier owns correctness.
"""

from __future__ import annotations

from dejavuu.drafters.base import DraftTree
from dejavuu.drafters.pld_plus import PLDPlus, _cos
from dejavuu.drafters.token_recycling import TokenRecycling


class AdaPLD(PLDPlus):
    SEM_FLOOR = 0.0  # min cosine to draft off a semantic match (else stay chain=[anchor])

    def __init__(self, n_min: int = 1, n_max: int = 3, k: int = 4):
        super().__init__(n_min, n_max)
        self._tr = TokenRecycling(k)  # token -> top-k (successor, prob) from verify logits

    @property
    def successors(self):
        return self._tr.successors

    def observe(self, input_tokens: list[int], logits) -> None:
        self._tr.observe(input_tokens, logits)

    def _semantic_cont(self, ctx: list[int], budget: int) -> list[int]:
        """No lexical hit: retrieve the past position whose hidden window best matches the
        current context, and copy what followed it. The query is the QWIN-token window
        ending just before the anchor (whose own hidden isn't computed yet); a candidate
        at p aligns p->end-1, so p+1 is the anchor slot and the post-anchor draft starts
        at p+2. A follower that actually equals the anchor token is a strong signal, so it
        gets a similarity bonus. ponytail: brute-force cosine over the memory (no ANN) --
        fine at benchmark sizes, swap in an index if it bites."""
        n = len(ctx)
        q = self._window(range(n - 1 - self.QWIN, n - 1))  # ends before the anchor
        if q is None:
            return []
        anchor = ctx[-1]
        best_p, best_s = None, self.SEM_FLOOR
        for p in self.hid:
            if p >= n - 2:  # skip the query span / anchor and leave room to copy
                continue
            cand = self._window(range(p - self.QWIN + 1, p + 1))
            if cand is None:
                continue
            s = _cos(q, cand) + (1.0 if ctx[p + 1] == anchor else 0.0)
            if s > best_s:
                best_s, best_p = s, p
        if best_p is None:
            return []
        return ctx[best_p + 2 : best_p + 2 + budget]

    def propose(self, ctx: list[int], past_len: int, budget: int) -> DraftTree:
        cont = self._lexical_cont(ctx, budget) or self._semantic_cont(ctx, budget)
        return DraftTree.chain([ctx[-1], *cont])

    def _successor_token(self, ctx: list[int], tok: int) -> int | None:
        """Most recent (hidden-reranked) occurrence of `tok` -> the token that followed."""
        occ = [i for i in range(len(ctx) - 2, -1, -1) if ctx[i] == tok]
        if not occ:
            return None
        return ctx[self._rerank(occ, 1, len(ctx) - 1) + 1]

    def propose_tree(self, ctx: list[int], past_len: int, budget: int, width: int) -> DraftTree:
        root = ctx[-1]
        main = self._lexical_cont(ctx, budget)
        conts = [main] if main else []
        first = main[0] if main else None
        for tok, _ in self.successors.get(root, [])[:width]:
            if tok == first:  # already the main path's first step
                continue
            branch = [tok]
            nxt = self._successor_token(ctx, tok)
            if nxt is not None:
                branch.append(nxt)
            conts.append(branch)
        if not conts:
            return DraftTree.chain([root])
        return DraftTree.branches(root, conts, budget)


def _demo() -> None:
    import numpy as np

    # semantic fallback: no n-gram match on the suffix, retrieve by hidden similarity.
    d = AdaPLD(n_min=1, n_max=1)
    ctx = [7, 1, 2, 3, 9]  # suffix token 9 never recurs -> lexical miss
    d.hid = {i: np.array([0.0, 0.0]) for i in range(5)}
    d.hid[1] = np.array([1.0, 0.0])  # position 1 looks like the query...
    d.hid[3] = np.array([1.0, 0.0])  # query = hid[len-2] = hid[3]
    # best match p=1 (skip p>=3); post-anchor draft starts at ctx[1+2]=ctx[3]=3
    assert d.propose(ctx, 0, 1).token_ids == [9, 3]

    # tree: main copy path + a logit branch token, assembled as branches.
    t = AdaPLD(n_min=1, n_max=2)
    rep = [5, 6, 5, 6]  # "5 6" recurs -> lexical main path fires
    logits = np.full((len(rep), 8), -9.0, np.float32)
    logits[-1, 7] = 9.0  # anchor's top successor is token 7
    t.observe(rep, logits)
    tree = t.propose_tree(rep, 0, 6, width=2)
    assert tree.token_ids[0] == 6
    assert len(tree.token_ids) > 1
    assert 7 in tree.token_ids  # the logit branch token made it in
    print("adapld demo ok")


if __name__ == "__main__":
    _demo()
