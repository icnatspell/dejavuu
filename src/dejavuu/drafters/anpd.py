"""ANPD -- Adaptive N-gram Parallel Decoding.

PLD drafts a fixed `budget` of tokens every step; ANPD adapts the draft length to the
*recent* acceptance it actually got. On a small target where decode is cheap (§7),
over-drafting wastes verification on tokens that get rejected, and under-drafting
leaves speedup on the table. ANPD tracks one integer -- the current draft length --
and grows it when the last draft was fully accepted, eases it back toward the observed
accept length otherwise. Pure CPU, stateless retrieval (same longest-trailing-n-gram
match as PLD), lossless (the drafter only proposes; verification owns correctness).
"""

from __future__ import annotations

from dejavuu.drafters.base import Drafter, DraftTree


class ANPD(Drafter):
    """Longest trailing n-gram match (like PLD), but the continuation length adapts to
    recent acceptance instead of always drafting `budget`. `draft_len` rises by 1 when
    the previous draft was fully accepted (we could have used more) and eases toward
    the realized accept length on a rejection, clamped to [1, max_len]."""

    def __init__(self, n_min: int = 1, n_max: int = 3, init_len: int = 4, max_len: int = 8):
        self.n_min, self.n_max, self.max_len = n_min, n_max, max_len
        self._init = init_len
        self.draft_len = init_len
        self._proposed = 0  # guesses proposed last step, for the adaptive update

    def reset(self, prompt_ids: list[int]) -> None:
        self.draft_len = self._init
        self._proposed = 0

    def propose(self, ctx: list[int], past_len: int, budget: int) -> DraftTree:
        root = ctx[-1]
        k = min(self.draft_len, budget)
        for n in range(min(self.n_max, len(ctx) - 1), self.n_min - 1, -1):
            pat = ctx[-n:]
            for j in range(len(ctx) - n - 1, -1, -1):  # earlier occurrences, latest first
                if ctx[j : j + n] == pat:
                    cont = ctx[j + n : j + n + k]
                    if cont:
                        self._proposed = len(cont)
                        return DraftTree.chain([root, *cont])
        self._proposed = 0  # no match: a miss, don't let it shrink draft_len
        return DraftTree.chain([root])

    def update(self, accepted: list[int]) -> None:
        if self._proposed == 0:
            return
        acc_len = len(accepted) - 1  # emitted = accepted guesses + 1 bonus token
        if acc_len >= self._proposed:  # fully accepted -> probe longer
            self.draft_len = min(self.draft_len + 1, self.max_len)
        else:  # over-drafted -> ease toward what actually landed
            self.draft_len = max(1, (self.draft_len + acc_len + 1) // 2)
