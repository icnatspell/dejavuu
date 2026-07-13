"""Retrieval drafter with a verifier-logit fallback.

Retrieval drafters (PLD, suffix index, ...) win where the future repeats the past and
collapse to accepted length ~1 where it doesn't (reasoning, math, novel text). The
verifier's logits -- handed to every drafter for free via `observe()` -- carry a usable
next-token signal exactly there (token_recycling reaches accepted length ~1.3-1.5 on
those categories where retrieval floors at ~1.1). This composes the two: use the base
drafter's proposal where it fires; fall back to a logit-table drafter where it finds
nothing. Lossless -- the verifier still owns acceptance, so a bad fallback can only cost
a wasted verify slot, never correctness.
"""

from __future__ import annotations

from dejavuu.drafters.base import Drafter, DraftTree
from dejavuu.drafters.prompt_lookup import PLD
from dejavuu.drafters.suffix_decoding import SuffixDecoding
from dejavuu.drafters.token_recycling import TokenRecycling


def _graft(base_tree: DraftTree, fb: DraftTree, budget: int) -> DraftTree:
    """Add the fallback's continuation as a new chain hanging off the shared root, using
    only budget the base left unused. Skips a first token the root already branches into
    (no duplicate sibling). Returns a valid tree (parents stay topological)."""
    tokens = list(base_tree.token_ids)
    parent = list(base_tree.parent)
    root_children = {tokens[c] for c in range(1, len(tokens)) if parent[c] == 0}
    prev = 0
    for i, tok in enumerate(fb.token_ids[1:]):
        if len(tokens) - 1 >= budget:
            break
        if i == 0 and tok in root_children:  # base already covers this branch
            break
        tokens.append(tok)
        parent.append(prev)
        prev = len(tokens) - 1
    return DraftTree(tokens, parent)


class Hybrid(Drafter):
    """Base retrieval drafter + logit-table fallback. Every engine hook feeds BOTH
    sub-drafters so the fallback's table and the base's index both stay warm."""

    def __init__(self, base: Drafter, fallback: Drafter, mode: str = "empty"):
        # mode: "empty" -> use the fallback only where the base finds nothing (V1).
        #       "graft" -> also add the fallback as a sibling branch on leftover budget.
        #       "tail"  -> extend the base's deepest path with fallback successors on
        #                  leftover budget (grows the winning path, not a rival branch).
        self.base = base
        self.fallback = fallback
        self.mode = mode

    def propose(self, ctx: list[int], past_len: int, budget: int) -> DraftTree:
        tree = self.base.propose(ctx, past_len, budget)
        if self.mode == "tail":
            return self._extend_tail(tree, budget)  # linear -> valid on the chain path
        # Sibling branches would get wrong logits with no tree attention, so on the chain
        # path we can only replace-on-empty, never merge.
        if len(tree.token_ids) > 1:
            return tree
        return self.fallback.propose(ctx, past_len, budget)

    def propose_tree(self, ctx: list[int], past_len: int, budget: int, width: int) -> DraftTree:
        tree = self.base.propose_tree(ctx, past_len, budget, width)
        if self.mode == "tail":
            return self._extend_tail(tree, budget)
        guesses = len(tree.token_ids) - 1
        if guesses == 0:  # base empty -> use the fallback outright
            return self.fallback.propose_tree(ctx, past_len, budget, width)
        if self.mode != "graft" or guesses >= budget:  # base fired and (V1, or no budget left)
            return tree
        # graft the fallback's chain as an extra root branch on leftover budget.
        return _graft(tree, self.fallback.propose_tree(ctx, past_len, budget, width), budget)

    def _extend_tail(self, tree: DraftTree, budget: int) -> DraftTree:
        """Continue the base's deepest path with the fallback's successors, using the
        budget the base left unused. When the base is empty this extends from the root
        (== the plain fallback); when it fired short, it lengthens the winning path
        rather than adding a rival branch."""
        guesses = len(tree.token_ids) - 1
        if guesses >= budget:
            return tree
        depth = [0] * len(tree.parent)
        for i, p in enumerate(tree.parent):
            if p >= 0:
                depth[i] = depth[p] + 1
        tail = max(range(len(depth)), key=lambda i: depth[i])  # deepest node (first if tie)
        # Fallback successors of the tail token (token_recycling keys on ctx[-1]).
        cont = self.fallback.propose([tree.token_ids[tail]], 0, budget - guesses).token_ids[1:]
        tokens, parent = list(tree.token_ids), list(tree.parent)
        seen = set(tokens)
        prev = tail
        for tok in cont:
            if len(tokens) - 1 >= budget or tok in seen:  # respect budget; no path cycle
                break
            tokens.append(tok)
            parent.append(prev)
            seen.add(tok)
            prev = len(tokens) - 1
        return DraftTree(tokens, parent)

    def reset(self, prompt_ids: list[int]) -> None:
        self.base.reset(prompt_ids)
        self.fallback.reset(prompt_ids)

    def update(self, accepted: list[int]) -> None:
        self.base.update(accepted)
        self.fallback.update(accepted)

    def observe(self, input_tokens: list[int], logits) -> None:
        self.base.observe(input_tokens, logits)
        self.fallback.observe(input_tokens, logits)

    def observe_hidden(self, tokens: list[int], hidden, base_pos: int) -> None:
        self.base.observe_hidden(tokens, hidden, base_pos)
        self.fallback.observe_hidden(tokens, hidden, base_pos)

    def note_cost(self, verify_s: float, submitted: int) -> None:
        self.base.note_cost(verify_s, submitted)
        self.fallback.note_cost(verify_s, submitted)

    def set_sampling(self, sampler, position: int, tree: bool = False) -> None:
        self.base.set_sampling(sampler, position, tree)
        self.fallback.set_sampling(sampler, position, tree)


class SuffixRecycle(Hybrid):
    """Online suffix index with a verifier-logit fallback."""

    def __init__(self):
        super().__init__(SuffixDecoding(), TokenRecycling())


class SuffixRecycleMerge(Hybrid):
    """suffix_recycle, but grafts the logit fallback as a sibling branch whenever suffix
    leaves budget unused (not only when suffix is empty)."""

    def __init__(self):
        super().__init__(SuffixDecoding(), TokenRecycling(), mode="graft")


class SuffixRecycleTail(Hybrid):
    """suffix_recycle, but *extends* suffix's deepest path with logit-table successors on
    leftover budget -- lengthening the winning path instead of adding a rival branch."""

    def __init__(self):
        super().__init__(SuffixDecoding(), TokenRecycling(), mode="tail")


class PldRecycle(Hybrid):
    """PLD with a verifier-logit fallback. Note: PLD (n_min=1) almost always finds
    *some* n-gram match, so it is rarely empty -- the fallback seldom fires, which is
    itself the point (fallback-on-empty is the wrong trigger for a greedy retriever)."""

    def __init__(self):
        super().__init__(PLD(), TokenRecycling())
