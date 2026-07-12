"""N-Gram Trie speculative drafting from in-context continuations.

The trie groups every continuation that followed the same prompt n-gram. Unlike a
first-token candidate pool, shared prefixes stay merged and later tokens may branch
too, so tree verification can test several complete in-context futures in one pass.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from dejavuu.drafters.base import Drafter, DraftTree


@dataclass
class _Node:
    count: int = 0
    children: dict[int, _Node] = field(default_factory=dict)

    def insert(self, tokens: list[int]) -> None:
        node = self
        for token in tokens:
            node = node.children.setdefault(token, _Node())
            node.count += 1


class NGramTrie(Drafter):
    """Build a prompt n-gram -> continuation trie at request setup.

    ``prefix`` is the largest lookup n-gram. ``continuation`` bounds the offline
    prompt-derived continuation stored below each matched prefix; the runtime budget
    remains the final authority on submitted tree nodes.
    """

    def __init__(self, prefix: int = 3, continuation: int = 10, min_prefix: int = 1):
        self.prefix = prefix
        self.continuation = continuation
        self.min_prefix = min_prefix
        self.tries: dict[tuple[int, ...], _Node] = {}

    def reset(self, prompt_ids: list[int]) -> None:
        self.tries = {}
        for n in range(self.min_prefix, self.prefix + 1):
            roots: defaultdict[tuple[int, ...], _Node] = defaultdict(_Node)
            for start in range(len(prompt_ids) - n):
                key = tuple(prompt_ids[start : start + n])
                cont = prompt_ids[start + n : start + n + self.continuation]
                if cont:
                    roots[key].insert(cont)
            self.tries.update(roots)

    def _root(self, ctx: list[int]) -> _Node | None:
        for n in range(min(self.prefix, len(ctx)), self.min_prefix - 1, -1):
            root = self.tries.get(tuple(ctx[-n:]))
            if root is not None:
                return root
        return None

    @staticmethod
    def _ranked(node: _Node, width: int) -> list[tuple[int, _Node]]:
        return sorted(node.children.items(), key=lambda item: item[1].count, reverse=True)[:width]

    def propose(self, ctx: list[int], past_len: int, budget: int) -> DraftTree:
        root = self._root(ctx)
        chain = [ctx[-1]]
        node = root
        while node is not None and node.children and len(chain) - 1 < budget:
            token, node = self._ranked(node, 1)[0]
            chain.append(token)
        return DraftTree.chain(chain)

    def propose_tree(self, ctx: list[int], past_len: int, budget: int, width: int) -> DraftTree:
        root = self._root(ctx)
        if root is None or budget <= 0:
            return DraftTree.chain([ctx[-1]])
        tokens, parent = [ctx[-1]], [-1]

        def visit(node: _Node, parent_idx: int) -> None:
            for token, child in self._ranked(node, width):
                if len(tokens) - 1 >= budget:
                    return
                idx = len(tokens)
                tokens.append(token)
                parent.append(parent_idx)
                visit(child, idx)

        visit(root, 0)
        return DraftTree(tokens, parent)
