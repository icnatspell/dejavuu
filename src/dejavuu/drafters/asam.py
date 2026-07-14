"""ASAM (Adaptive SAM) retrieval drafter -- SAM's two-source longest match, with an acceptance-
calibrated cap on draft length. `verify_aware=True` opts into the cost-aware variant that
also sizes drafts to the measured verify cost. Contains the rest of the zoo as special
cases (see class docstring)."""

from __future__ import annotations

from dejavuu.drafters.base import Drafter, DraftTree
from dejavuu.drafters.suffix_index import SuffixIndex


class ASAM(Drafter):
    """One drafter that subsumes the others. Retrieval is SAM-Decoding: a static
    datastore + the live generation, drafting from whichever yields the longer suffix
    match (match length n = retrieval confidence, and the ceiling on draft length).

    Two ways to size the draft under that ceiling:

    * `verify_aware=False` (default): the actual draft length is min(n, cap), where `cap`
      is an ANPD-style running estimate of what the model accepts -- grown on a fully
      accepted draft, eased toward the realized accept length on a rejection.

    * `verify_aware=True`: each step pick the k in 1..n that maximizes expected throughput
      `E_accept(k) / (1 + g*k)`, where E_accept uses a geometric acceptance model with
      EMA per-token accept rate `alpha`, and `g = c1/c0` is the marginal-vs-fixed verify
      cost (learned online from `note_cost`). When verify is cheap (g->0) the optimum
      pushes to the ceiling (aggressive); when expensive it pulls in (conservative). Only
      drafted (k>=1) steps feed the cost fit -- a launch-bound backend's verify jumps
      sharply from M=1 to M>=2, so mixing the cheap no-draft steps in would smear that jump
      into `c1` and wrongly collapse drafts to length 1.

    Special cases: datastore off + length at ceiling -> SuffixDecoding; short matches +
    cap binds -> ANPD; datastore on -> SAM-Decoding. Lossless under greedy (pure
    retrieval; sizing only shortens a draft, never changes which tokens are proposed)."""

    def __init__(
        self,
        datastore: list[list[int]] | None = None,
        order: int = 8,
        min_match: int = 2,
        max_len: int = 8,
        verify_aware: bool = False,
        beta: float = 0.9,  # EMA decay for the online alpha / cost estimates
    ):
        self.static = SuffixIndex(order)
        for doc in datastore or []:
            self.static.extend(doc)
            self.static.append(self.static.SEP)
        self.dynamic = SuffixIndex(order)
        self.min_match = min_match
        self.max_len = max_len
        self.verify_aware = verify_aware
        self.beta = beta
        self.seen = 0
        self.cap = max_len  # (non-verify-aware) starts aggressive, eases to realized accept
        self._proposed = 0
        # verify-aware state: EMA accept rate + EMA regression moments for verify cost
        self._alpha = 0.5
        self._n = self._x = self._y = self._xx = self._xy = 0.0

    def reset(self, prompt_ids: list[int]) -> None:
        self.dynamic.append(self.dynamic.SEP)
        self.seen = 0
        self.cap = self.max_len
        self._proposed = 0

    def _ingest(self, ctx: list[int]) -> None:
        self.dynamic.extend(ctx[self.seen :])
        self.seen = len(ctx)

    def propose(self, ctx: list[int], past_len: int, budget: int) -> DraftTree:
        self._ingest(ctx)
        cs, ns = self.static.continuation(ctx, budget, by_freq=True)
        cd, nd = self.dynamic.continuation(ctx, budget, by_freq=True)
        cont, n = (cs, ns) if ns >= nd else (cd, nd)  # higher-confidence source
        if n < self.min_match:
            self._proposed = 0
            return DraftTree.chain([ctx[-1]])
        ceiling = min(n, budget)  # retrieval evidence caps the draft
        k = self._best_len(ceiling) if self.verify_aware else min(ceiling, self.cap)
        self._proposed = k
        return DraftTree.chain([ctx[-1], *cont[:k]])

    def propose_tree(self, ctx: list[int], past_len: int, budget: int, width: int) -> DraftTree:
        """Branch off the longer-match source (SAM-Decoding style). The adaptive
        draft-length cap is a chain-mode sizing heuristic; under tree verification the
        node budget bounds total guesses instead, so we leave `_proposed = 0` here to
        keep the chain-specific accept/cost adaptation dormant. Still lossless -- pure
        retrieval, the verifier owns correctness."""
        self._ingest(ctx)
        cs, ns = self.static.continuations(ctx, budget, width)
        cd, nd = self.dynamic.continuations(ctx, budget, width)
        conts, n = (cs, ns) if ns >= nd else (cd, nd)  # higher-confidence source
        self._proposed = 0
        if n < self.min_match:
            return DraftTree.chain([ctx[-1]])
        return DraftTree.branches(ctx[-1], conts, budget)

    def _best_len(self, ceiling: int) -> int:
        """argmax_k speed(k) over k in 1..ceiling, speed(k) = E_accept(k)/(1 + g*k)."""
        a, g = self._alpha, self._cost_ratio()
        best_k, best = 1, -1.0
        for k in range(1, ceiling + 1):
            eacc = (1 - a ** (k + 1)) / (1 - a) if a < 1 else float(k + 1)
            speed = eacc / (1 + g * k)  # c0 cancels; only the c1/c0 ratio matters
            if speed > best:
                best, best_k = speed, k
        return best_k

    def _cost_ratio(self) -> float:
        """c1/c0 from the EMA regression verify_s ~= c0 + c1*submitted; 0 before warmup
        (no spread in submitted yet) -> treat verify as free -> draft to the ceiling."""
        if self._n < 1e-6:
            return 0.0
        x, y = self._x / self._n, self._y / self._n
        var = self._xx / self._n - x * x
        if var < 1e-9:
            return 0.0
        c1 = (self._xy / self._n - x * y) / var
        c0 = y - c1 * x
        return max(0.0, c1) / max(c0, 1e-9)

    def note_cost(self, verify_s: float, submitted: int) -> None:
        # Only drafted (k>=1) steps inform the cost of *drafting*. A no-draft (k=0, M=1)
        # step is a different, cheaper population: on a launch-bound backend verify jumps
        # sharply from M=1 to M>=2, so folding k=0 samples into this linear fit smears that
        # fixed jump into the per-token slope (c1), which then over-penalises length and
        # collapses drafts to 1. Excluding them keeps the slope the true marginal cost.
        if not self.verify_aware or submitted <= 0:
            return
        b, x, y = self.beta, float(submitted), verify_s
        self._n = b * self._n + (1 - b)
        self._x = b * self._x + (1 - b) * x
        self._y = b * self._y + (1 - b) * y
        self._xx = b * self._xx + (1 - b) * x * x
        self._xy = b * self._xy + (1 - b) * x * y

    def update(self, accepted: list[int]) -> None:
        if self._proposed == 0:
            return
        acc_len = len(accepted) - 1  # emitted = accepted guesses + 1 bonus token
        if self.verify_aware:  # EMA the per-token accept rate for the throughput model
            r = acc_len / self._proposed
            self._alpha = self.beta * self._alpha + (1 - self.beta) * r
            return
        if acc_len >= self._proposed:  # fully accepted -> probe longer
            self.cap = min(self.cap + 1, self.max_len)
        else:  # over-drafted -> ease toward what actually landed
            self.cap = max(1, (self.cap + acc_len + 1) // 2)


def _demo() -> None:
    # non-verify-aware cap calibrates down under persistent over-drafting
    d = ASAM(max_len=8)
    d.reset([])
    d.cap, d._proposed = 8, 6
    d.update([0, 1])  # acc_len 1 << 6 proposed
    assert d.cap < 8, d.cap

    # verify-aware: cheap no-draft (k=0) samples must be ignored so the M=1->M>=2 jump is
    # not smeared into the slope. With the jump excluded and a cheap marginal token, the
    # sizer drafts long; a steep marginal cost pulls the length in.
    la = ASAM(verify_aware=True)
    la.reset([])
    la._alpha = 0.6
    for _ in range(50):
        la.note_cost(0.007, 0)  # no-draft steps: must not fool the drafted-cost fit
        for k in (1, 2, 3, 4):  # drafted: cheap per-token slope -> go long
            la.note_cost(0.013 + 0.0005 * k, k)
    cheap_k = la._best_len(4)
    la2 = ASAM(verify_aware=True)
    la2.reset([])
    la2._alpha = 0.6
    for k in range(1, 5):  # steep per-token slope -> pull the length in
        la2.note_cost(0.010 + 0.02 * k, k)
    pricey_k = la2._best_len(4)
    assert cheap_k >= 3, cheap_k
    assert cheap_k > pricey_k, (cheap_k, pricey_k)
    print(f"asam ok; cap adapts, verify-aware k: cheap={cheap_k} pricey={pricey_k}")


if __name__ == "__main__":
    _demo()
