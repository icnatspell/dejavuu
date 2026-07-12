"""Lookahead / Ouroboros -- multi-candidate n-gram pool drafting.

Lookahead Decoding grows an n-gram pool from the model's own output and drafts by
*pooling several* candidate continuations of the matched n-gram (Ouroboros adds a
draft model + phrase refinement -- out of scope, that needs a second model). The
distinctive piece we keep is the multi-candidate retrieval: where PLD takes the single
most-recent continuation and SuffixDecoding scores one path, Lookahead verifies the
top-N distinct continuations of the longest matched n-gram as parallel tree branches,
catching the right one when a context has branched before.

We deliberately drop the Jacobi-iteration half of Lookahead: it spends extra forward
passes to manufacture n-grams, which is counterproductive on a small target where
decode is already cheap (§7). Pool is harvested for free from the running context.
Pure CPU, lossless (proposer only); chain mode degrades to PLD-style single match.
"""

from __future__ import annotations

from collections import Counter

from dejavuu.drafters.base import Drafter, DraftTree


class Lookahead(Drafter):
    """Longest trailing n-gram match over the context, then pool continuations:
    chain mode returns the most recent one (PLD); tree mode returns the `width` most
    frequent distinct continuations as branches (the Lookahead pool)."""

    def __init__(self, n_min: int = 1, n_max: int = 3):
        self.n_min, self.n_max = n_min, n_max

    def _match(self, ctx: list[int], budget: int) -> tuple[int, list[list[int]]]:
        """Longest n with >=1 earlier occurrence of the trailing n-gram; return n and
        the continuations (each up to `budget` tokens) at every occurrence, latest
        first. (0, []) if nothing matches."""
        for n in range(min(self.n_max, len(ctx) - 1), self.n_min - 1, -1):
            pat = ctx[-n:]
            conts = [
                ctx[j + n : j + n + budget]
                for j in range(len(ctx) - n - 1, -1, -1)
                if ctx[j : j + n] == pat and ctx[j + n : j + n + budget]
            ]
            if conts:
                return n, conts
        return 0, []

    def propose(self, ctx: list[int], past_len: int, budget: int) -> DraftTree:
        _, conts = self._match(ctx, budget)
        return DraftTree.chain([ctx[-1], *conts[0]]) if conts else DraftTree.chain([ctx[-1]])

    def propose_tree(self, ctx: list[int], past_len: int, budget: int, width: int) -> DraftTree:
        _, conts = self._match(ctx, budget)
        if not conts:
            return DraftTree.chain([ctx[-1]])
        # pool: branch on the most frequent *next* tokens (where the frontier accept
        # happens); for each, take its most recent full continuation (conts is latest
        # first). This is the Lookahead multi-candidate set.
        top = [t for t, _ in Counter(c[0] for c in conts).most_common(width)]
        per = max(1, budget // width)  # spread the budget so candidates actually branch
        branches = [next(c for c in conts if c[0] == t)[:per] for t in top]
        return DraftTree.branches(ctx[-1], branches, budget)
