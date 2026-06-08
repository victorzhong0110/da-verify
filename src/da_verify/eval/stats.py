"""Statistics for comparing verification conditions — the project's moat.

Two conditions are run on the SAME tasks, so the per-task correct/wrong outcomes
are PAIRED. The right tools for that:

- Wilson score interval for each condition's accuracy (a proportion). Better than
  the normal approximation at small n / extreme p, which is exactly our regime
  (40 tasks, accuracy near 0.8).

- McNemar's test for whether the conditions DIFFER. It looks only at the
  DISCORDANT pairs — tasks one condition got right and the other wrong — because
  tasks both get right (or both wrong) carry no information about a difference.
  We use the EXACT binomial version (no scipy dep): under H0 each discordant pair
  is a coin flip, so the two-sided p-value is 2·P(X ≤ min(b,c)), X~Binom(b+c, ½).

Why not a t-test / unpaired chi-square? The data is paired binary on the same
items; a t-test assumes continuous independent samples and would be wrong here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def wilson_interval(c: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval (default z=1.96) for c successes in n trials."""
    if n == 0:
        return (0.0, 0.0)
    p = c / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts.

    b = #(condition A correct, B wrong); c = #(A wrong, B correct).
    Concordant pairs are intentionally ignored.
    """
    n = b + c
    if n == 0:
        return 1.0  # no discordant pairs -> no evidence of a difference
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


@dataclass(frozen=True)
class PairedComparison:
    n: int
    acc_a: float
    acc_b: float
    ci_a: tuple[float, float]
    ci_b: tuple[float, float]
    both: int          # both correct
    a_only: int        # A correct, B wrong  -> "B broke it"
    b_only: int        # A wrong, B correct  -> "B fixed it"
    neither: int       # both wrong
    net: int           # b_only - a_only (positive => B helped on net)
    risk_diff: float   # acc_b - acc_a
    mcnemar_p: float

    def summary(self, a_name: str = "A", b_name: str = "B") -> str:
        sig = "significant" if self.mcnemar_p < 0.05 else "n.s."
        return (
            f"{a_name}: {self.acc_a:.1%} (95% CI {self.ci_a[0]:.1%}–{self.ci_a[1]:.1%})\n"
            f"{b_name}: {self.acc_b:.1%} (95% CI {self.ci_b[0]:.1%}–{self.ci_b[1]:.1%})\n"
            f"Δ = {self.risk_diff:+.1%}   "
            f"{b_name} fixed {self.b_only}, broke {self.a_only} (net {self.net:+d})\n"
            f"McNemar exact p = {self.mcnemar_p:.4f} ({sig} at α=0.05; "
            f"discordant pairs = {self.a_only + self.b_only})"
        )


def compare_paired(a_correct: list[bool], b_correct: list[bool]) -> PairedComparison:
    """Compare two conditions' per-task correctness (aligned, same task order)."""
    if len(a_correct) != len(b_correct):
        raise ValueError("conditions must cover the same tasks (equal length, aligned)")
    n = len(a_correct)
    both = sum(1 for x, y in zip(a_correct, b_correct) if x and y)
    a_only = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    b_only = sum(1 for x, y in zip(a_correct, b_correct) if not x and y)
    neither = n - both - a_only - b_only
    ca, cb = sum(a_correct), sum(b_correct)
    return PairedComparison(
        n=n,
        acc_a=ca / n if n else 0.0,
        acc_b=cb / n if n else 0.0,
        ci_a=wilson_interval(ca, n),
        ci_b=wilson_interval(cb, n),
        both=both, a_only=a_only, b_only=b_only, neither=neither,
        net=b_only - a_only,
        risk_diff=(cb - ca) / n if n else 0.0,
        mcnemar_p=mcnemar_exact_p(a_only, b_only),
    )
