"""Cacheback -- bounded online n-gram cache drafting.

The cache maps a fixed-length *leader* n-gram to recently observed fixed-length
*followers*.  It is updated from emitted target tokens only, so a cache hit is
always a speculative proposal: the shared verifier remains the correctness gate.
"""

from __future__ import annotations

from collections import OrderedDict

from dejavuu.drafters.base import Drafter, DraftTree


class Cacheback(Drafter):
    """Draft from a bounded LRU cache of token n-gram continuations.

    `leader_capacity` bounds distinct lookup keys and `follower_capacity` bounds
    alternatives retained per key. Both limits apply in the online hot path.
    """

    def __init__(
        self,
        leader_len: int = 4,
        follower_len: int = 4,
        leader_capacity: int = 16_384,
        follower_capacity: int = 4,
    ):
        if min(leader_len, follower_len, leader_capacity, follower_capacity) < 1:
            raise ValueError("Cacheback lengths and capacities must be positive")
        self.leader_len = leader_len
        self.follower_len = follower_len
        self.leader_capacity = leader_capacity
        self.follower_capacity = follower_capacity
        self.cache: OrderedDict[tuple[int, ...], OrderedDict[tuple[int, ...], None]] = OrderedDict()
        self._tail: list[int] = []

    def reset(self, prompt_ids: list[int]) -> None:
        # The cache persists across requests; only the rolling update window is local.
        self._tail = []

    def update(self, accepted: list[int]) -> None:
        """Insert each newly complete leader/follower pair into the LRU cache."""
        for token in accepted:
            self._tail.append(token)
            needed = self.leader_len + self.follower_len
            if len(self._tail) < needed:
                continue
            leader = tuple(self._tail[-needed : -self.follower_len])
            follower = tuple(self._tail[-self.follower_len :])
            followers = self.cache.get(leader)
            if followers is None:
                followers = OrderedDict()
                self.cache[leader] = followers
            else:
                self.cache.move_to_end(leader)
            followers.pop(follower, None)
            followers[follower] = None
            while len(followers) > self.follower_capacity:
                followers.popitem(last=False)
            while len(self.cache) > self.leader_capacity:
                self.cache.popitem(last=False)
            del self._tail[:-needed]

    def _follower(self, leader: tuple[int, ...]) -> tuple[int, ...] | None:
        followers = self.cache.get(leader)
        if followers is None:
            return None
        self.cache.move_to_end(leader)
        follower = next(reversed(followers))  # most recently useful local continuation
        followers.move_to_end(follower)
        return follower

    def _followers(self, leader: tuple[int, ...], width: int) -> list[tuple[int, ...]]:
        """Return up to `width` most-recent alternatives and mark the lookup hot."""
        followers = self.cache.get(leader)
        if followers is None:
            return []
        self.cache.move_to_end(leader)
        out = list(reversed(followers))[:width]
        for follower in reversed(out):
            followers.move_to_end(follower)
        return out

    def propose(self, ctx: list[int], past_len: int, budget: int) -> DraftTree:
        if len(ctx) < self.leader_len or budget < 1:
            return DraftTree.chain([ctx[-1]])
        window = list(ctx)
        drafted: list[int] = []
        while len(drafted) < budget:
            follower = self._follower(tuple(window[-self.leader_len :]))
            if follower is None:
                break
            take = follower[: budget - len(drafted)]
            drafted.extend(take)
            window.extend(take)
        return DraftTree.chain([ctx[-1], *drafted])

    def propose_tree(self, ctx: list[int], past_len: int, budget: int, width: int) -> DraftTree:
        """Recursively expand recent cached followers into verifier tree branches."""
        if len(ctx) < self.leader_len or budget < 1 or width < 1:
            return DraftTree.chain([ctx[-1]])
        tokens, parent = [ctx[-1]], [-1]
        frontier: list[tuple[int, list[int]]] = [(0, list(ctx))]
        while frontier and len(tokens) - 1 < budget:
            node, window = frontier.pop(0)
            for follower in self._followers(tuple(window[-self.leader_len :]), width):
                branch_window = list(window)
                previous = node
                for token in follower:
                    if len(tokens) - 1 >= budget:
                        return DraftTree(tokens, parent)
                    tokens.append(token)
                    parent.append(previous)
                    previous = len(tokens) - 1
                    branch_window.append(token)
                frontier.append((previous, branch_window))
        return DraftTree(tokens, parent)
