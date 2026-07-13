"""Token Recycling -- draft from the verify logits the engine already computes."""

from __future__ import annotations

import numpy as np

from dejavuu.drafters.base import Drafter, DraftTree


class TokenRecycling(Drafter):
    """Recycle the verify logits the engine already computes: remember each token's
    top-k successors *with their model probabilities*, then walk them to draft
    (plan 5.2). Sparse dict, not a [vocab, k] matrix -- same thing without 1M empty
    rows. Chain mode follows top-1; `propose_tree` grows a Sequoia tree from the
    stored probs (`DraftTree.grow`) instead of uniform-width branching, so budget
    goes to high-acceptance paths rather than low-probability siblings."""

    def __init__(self, k: int = 4):
        self.k = k
        # token -> top-k (next token, model prob), prob descending
        self.successors: dict[int, list[tuple[int, float]]] = {}

    def observe(self, input_tokens: list[int], logits) -> None:
        k = min(self.k, logits.shape[-1])
        rows = np.arange(len(input_tokens))[:, None]
        # top-k by logit, descending. argpartition is O(vocab) (vs a full O(vocab log vocab)
        # sort) -- it isolates the k best unordered, then we sort just those k. Partition
        # around the high side directly so we never negate the whole [N, vocab] matrix.
        part = np.argpartition(logits, -k, axis=-1)[:, -k:]  # [N, k] best, unordered
        topk = np.take_along_axis(part, np.argsort(-logits[rows, part], axis=-1), axis=-1)
        # softmax the k survivors per row -- no [N, vocab] exp/probs array. The dropped
        # denominator only rescales a row's own candidates, which leaves each token's
        # successor order (and the recycling tree it grows) unchanged.
        top_logits = np.take_along_axis(logits, topk, axis=-1)
        e = np.exp(top_logits - top_logits[:, :1])
        probs = e / e.sum(axis=-1, keepdims=True)
        for tok, row, prow in zip(input_tokens, topk, probs, strict=True):
            self.successors[tok] = [(int(t), float(p)) for t, p in zip(row, prow)]

    def propose(self, ctx: list[int], past_len: int, budget: int) -> DraftTree:
        chain = [ctx[-1]]
        cur = ctx[-1]
        seen = {cur}
        for _ in range(budget):
            succ = self.successors.get(cur)
            nxt = succ[0][0] if succ else None
            if nxt is None or nxt in seen:  # stop at unknowns / cycles
                break
            chain.append(nxt)
            seen.add(nxt)
            cur = nxt
        return DraftTree.chain(chain)

    def propose_tree(self, ctx: list[int], past_len: int, budget: int, width: int) -> DraftTree:
        """Sequoia tree from the stored successor probs (width ignored: the budget is
        allocated by path acceptance probability, not a fixed branch factor)."""
        return DraftTree.grow(ctx[-1], lambda t: self.successors.get(t, []), budget)


def _demo() -> None:
    # argpartition top-k must match a full-sort top-k, tokens ordered by logit desc.
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((3, 5000)).astype(np.float32)
    tr = TokenRecycling(k=4)
    tr.observe([10, 11, 12], logits)
    for r, tok in enumerate([10, 11, 12]):
        want = [int(t) for t in (-logits[r]).argsort()[:4]]  # reference: full sort
        got = [t for t, _ in tr.successors[tok]]
        assert got == want, (tok, got, want)
    print("token_recycling top-k ok")


if __name__ == "__main__":
    _demo()
