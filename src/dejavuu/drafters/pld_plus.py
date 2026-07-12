"""PLD+ -- prompt lookup with hidden-state reranking (Somasundaram 2025).

Plain PLD copies the *most recent* n-gram match. PLD+ gathers every match, then
reranks them by cosine similarity of the target model's hidden states and copies the
best continuation. Needs a decoder that emits hidden states (the SmolVLM tree+hidden
export); on a token-only decoder the hidden memory stays empty and it degrades to
plain PLD (most-recent match). Chain-only. Lossless: the verifier owns correctness,
so the rerank only affects which guess we copy, never the output.
"""

from __future__ import annotations

import numpy as np

from dejavuu.drafters.base import Drafter, DraftTree


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else -1.0


class PLDPlus(Drafter):
    QWIN = 3  # trailing hidden window for the semantic query (AdaPLD)

    def __init__(self, n_min: int = 1, n_max: int = 3):
        self.n_min, self.n_max = n_min, n_max
        self.hid: dict[int, np.ndarray] = {}  # abs seq position -> hidden row

    def reset(self, prompt_ids: list[int]) -> None:
        self.hid = {}  # positions are per-request seq indices

    def observe_hidden(self, tokens: list[int], hidden, base_pos: int) -> None:
        for i in range(len(tokens)):
            self.hid[base_pos + i] = hidden[i]

    def _window(self, positions: range) -> np.ndarray | None:
        """Mean of the available hidden rows over `positions` -- the context signature
        of that span. None if none are indexed yet (cold start / prompt tokens)."""
        vecs = [self.hid[p] for p in positions if p in self.hid]
        return np.mean(vecs, axis=0) if vecs else None

    def _rerank(self, cands: list[int], n: int, end: int) -> int:
        """Pick the match at `j` whose matched-n-gram hidden window best matches the
        current one. We compare the n-gram *minus its last token* on both sides: the
        current anchor (position end) has no hidden yet (it's a freshly emitted bonus
        token), so excluding that one slot keeps the comparison honest instead of one
        position stale. `cands` is latest-first, so a missing window falls back to plain
        PLD's most-recent match."""
        q = self._window(range(end - n + 1, end))
        if q is None:  # n==1 (empty span) or cold: fall back to the token before
            q = self._window(range(end - 1, end))
        if q is None:
            return cands[0]
        best_j, best_s, scored = cands[0], -2.0, False
        for j in cands:
            h = self._window(range(j, j + n - 1))
            if h is None:
                h = self._window(range(j - 1, j))
            if h is None:
                continue
            s = _cos(q, h)
            if s > best_s:
                best_s, best_j, scored = s, j, True
        return best_j if scored else cands[0]

    def _lexical_cont(self, ctx: list[int], budget: int) -> list[int]:
        """Longest trailing n-gram match, hidden-reranked -> its following tokens."""
        end = len(ctx) - 1  # anchor position (its own hidden isn't computed yet)
        for n in range(min(self.n_max, len(ctx) - 1), self.n_min - 1, -1):
            pat = ctx[-n:]
            cands = [j for j in range(len(ctx) - n - 1, -1, -1) if ctx[j : j + n] == pat]
            if not cands:
                continue
            j = self._rerank(cands, n, end)
            cont = ctx[j + n : j + n + budget]
            if cont:
                return cont
        return []

    def propose(self, ctx: list[int], past_len: int, budget: int) -> DraftTree:
        return DraftTree.chain([ctx[-1], *self._lexical_cont(ctx, budget)])


def _demo() -> None:
    # [1,2] recurs at j=0 (->3) and j=3 (->4); current suffix is [1,2].
    ctx = [1, 2, 3, 1, 2, 4, 1, 2]
    p = PLDPlus(n_min=1, n_max=3)
    # cold (no hidden) -> most recent match j=3 -> continuation [4] (plain PLD)
    assert p.propose(ctx, 0, 1).token_ids == [2, 4]
    # hidden makes the *older* match (j=0) look like the current context -> [3]
    p.hid = {0: np.array([1.0, 0.0]), 3: np.array([0.0, 1.0]), 6: np.array([1.0, 0.0])}
    assert p.propose(ctx, 0, 1).token_ids == [2, 3]
    print("pld_plus demo ok")


if __name__ == "__main__":
    _demo()
