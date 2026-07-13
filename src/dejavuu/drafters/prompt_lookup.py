"""PLD -- Prompt Lookup Decoding."""

from __future__ import annotations

from dejavuu.drafters.base import Drafter, DraftTree


class PLD(Drafter):
    """Prompt-lookup: longest trailing n-gram match over the context so far,
    return the following tokens as a chain. Pure CPU, stateless."""

    def __init__(self, n_min: int = 1, n_max: int = 4):
        # n_max=4 (not 3) wins on int8: a 4-gram match is more precise where it exists,
        # and cheap multi-token verify rewards the extra reach. 5-grams recur too rarely
        # to pay off. n_min stays 1 -- dropping 1-gram matches tripled draft precision but
        # cut recall/acceptance, a bad trade when verify is cheap. See issue #43.
        self.n_min, self.n_max = n_min, n_max

    def propose(self, ctx: list[int], past_len: int, budget: int) -> DraftTree:
        root = ctx[-1]
        for n in range(min(self.n_max, len(ctx) - 1), self.n_min - 1, -1):
            pat = ctx[-n:]
            # search earlier occurrences, latest first
            for j in range(len(ctx) - n - 1, -1, -1):
                if ctx[j : j + n] == pat:
                    cont = ctx[j + n : j + n + budget]
                    if cont:
                        return DraftTree.chain([root, *cont])
        return DraftTree.chain([root])

    def propose_tree(self, ctx: list[int], past_len: int, budget: int, width: int) -> DraftTree:
        """Same longest n-gram match as `propose`, but branch into the top `width`
        distinct next-tokens (latest occurrence first -- PLD's own recency tiebreak),
        each followed by its continuation. Falls back to a chain when the match forks
        no more than once. Stateless; lossless (the verifier owns correctness)."""
        root = ctx[-1]
        for n in range(min(self.n_max, len(ctx) - 1), self.n_min - 1, -1):
            pat = ctx[-n:]
            conts: list[list[int]] = []
            seen_next: set[int] = set()
            # latest occurrence first, so the first continuation kept per next-token
            # is the most recent one (matches propose's recency preference)
            for j in range(len(ctx) - n - 1, -1, -1):
                if ctx[j : j + n] != pat:
                    continue
                cont = ctx[j + n : j + n + budget]
                if cont and cont[0] not in seen_next:
                    seen_next.add(cont[0])
                    conts.append(cont)
                    if len(conts) >= width:
                        break
            if conts:
                return DraftTree.branches(root, conts, budget)
        return DraftTree.chain([root])
