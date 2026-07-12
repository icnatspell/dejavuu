"""Shared longest-suffix index used by the retrieval drafters (SuffixDecoding, REST,
SAM-Decoding). Pure token ids, so the same index backs both LLM and VLM drafters."""

from __future__ import annotations

import math
from collections import Counter, defaultdict


class SuffixIndex:
    """Reusable longest-suffix -> continuation over a token corpus. Documents are
    separated by a SEP sentinel so matches/continuations never span a boundary.

    ponytail: fixed-order n-gram hash index -- O(order) lookup, ~order x N memory.
    Not a suffix automaton; swap one in only if corpora get huge or you need
    unbounded-length matches (matters for tree branching, not capped chains)."""

    SEP = -1  # never a real token id

    def __init__(self, order: int = 8):
        self.order = order
        self.buf: list[int] = []
        self.pos: dict[tuple[int, ...], list[int]] = defaultdict(list)  # ngram->follow
        self._run = 0  # tokens since last SEP (caps match length at a boundary)

    def append(self, token: int) -> None:
        self.buf.append(token)
        if token == self.SEP:
            self._run = 0
            return
        p = len(self.buf) - 1  # this token is the follow-position of the ngrams before it
        for n in range(1, min(self.order, self._run) + 1):
            self.pos[tuple(self.buf[p - n : p])].append(p)
        self._run += 1

    def extend(self, tokens: list[int]) -> None:
        for t in tokens:
            self.append(t)

    def continuation(
        self, suffix: list[int], budget: int, by_freq: bool = False
    ) -> tuple[list[int], int]:
        """Longest k<=order with suffix[-k:] seen before -> up to `budget` following
        tokens. `by_freq` picks the most common next token across matches (else most
        recent). Returns (continuation, matched length k); ([], 0) if no match."""
        for n in range(min(self.order, len(suffix)), 0, -1):
            ends = self.pos.get(tuple(suffix[-n:]))
            if not ends:
                continue
            if by_freq:
                # follow-positions are never SEP (append registers only on non-SEP
                # tokens), so no boundary filtering is needed here -- the doc boundary
                # is enforced by the forward-walk below.
                top = Counter(self.buf[p] for p in ends).most_common(1)[0][0]
                ends = [p for p in ends if self.buf[p] == top]
            pos = ends[-1]  # most recent occurrence (appended in position order)
            out: list[int] = []
            while len(out) < budget and pos < len(self.buf) and self.buf[pos] != self.SEP:
                out.append(self.buf[pos])
                pos += 1
            if out:
                return out, n
        return [], 0

    def weighted_entropy(self) -> float:
        """Occurrence-weighted mean next-token entropy (bits) over every indexed
        context -- the SuffixDecoding 'structuredness' diagnostic. Each n-gram is a
        suffix-tree node; its output distribution is the tokens seen to follow it.
        Low = predictable outputs, so retrieval drafting (PLD/SuffixDecoding/SAM)
        pays off; high = open-ended, near-baseline. As a rough guide from the paper,
        ~0.1 bits -> ~10x, ~2.5 -> ~2x, >3 -> modest. 0.0 for an empty index."""
        tot_w = tot_h = 0.0
        for ends in self.pos.values():
            counts = Counter(self.buf[p] for p in ends)
            n = len(ends)
            h = -sum((c / n) * math.log2(c / n) for c in counts.values())
            tot_h += n * h
            tot_w += n
        return tot_h / tot_w if tot_w else 0.0

    def continuations(
        self, suffix: list[int], budget: int, width: int
    ) -> tuple[list[list[int]], int]:
        """Tree variant of `continuation`: at the longest match, branch into the top
        `width` next-tokens (by frequency), each followed to its continuation (most
        recent occurrence of that next-token). Returns (continuations, matched len k);
        ([], 0) if no match. Each continuation starts with its branching token."""
        for n in range(min(self.order, len(suffix)), 0, -1):
            ends = self.pos.get(tuple(suffix[-n:]))
            if not ends:
                continue
            freq = Counter(self.buf[p] for p in ends)  # follow-positions are never SEP
            outs = []
            for tok, _ in freq.most_common(width):
                pos = next(p for p in reversed(ends) if self.buf[p] == tok)
                out: list[int] = []
                while len(out) < budget and pos < len(self.buf) and self.buf[pos] != self.SEP:
                    out.append(self.buf[pos])
                    pos += 1
                if out:
                    outs.append(out)
            if outs:
                return outs, n
        return [], 0


def _demo() -> None:
    det = SuffixIndex(order=2)
    det.extend([5, 5, 5, 5])  # every context predicts 5 -> perfectly predictable
    assert det.weighted_entropy() == 0.0
    mix = SuffixIndex(order=1)
    mix.extend([0, 1, 0, 2])  # (0,)->{1,2} H=1 w2 ; (1,)->{0} H=0 w1 -> 2/3
    assert abs(mix.weighted_entropy() - 2 / 3) < 1e-9
    print("suffix_index demo ok")


if __name__ == "__main__":
    _demo()
