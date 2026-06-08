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
import random
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


@dataclass(frozen=True)
class RateComparison:
    n_tasks: int
    mean_a: float
    mean_b: float
    mean_diff: float        # mean_b - mean_a
    ci_lo: float
    ci_hi: float
    improved: int           # tasks where B's per-task rate > A's
    worsened: int           # tasks where B's per-task rate < A's

    @property
    def significant(self) -> bool:
        return self.ci_lo > 0 or self.ci_hi < 0  # 95% CI excludes 0

    def summary(self, a_name: str = "A", b_name: str = "B") -> str:
        sig = "SIGNIFICANT (CI excludes 0)" if self.significant else "n.s. (CI includes 0)"
        return (
            f"{a_name} mean per-task rate: {self.mean_a:.1%}\n"
            f"{b_name} mean per-task rate: {self.mean_b:.1%}\n"
            f"Δ = {self.mean_diff:+.1%}  (95% bootstrap CI {self.ci_lo:+.1%}..{self.ci_hi:+.1%}) — {sig}\n"
            f"{b_name} improved {self.improved} tasks, worsened {self.worsened} (of {self.n_tasks})"
        )


def bootstrap_paired_diff(
    a_rates: list[float], b_rates: list[float], n_boot: int = 10000, seed: int = 0
) -> RateComparison:
    """Paired bootstrap 95% CI on mean(b)-mean(a) over matched items.

    For k>1 the per-task unit is a RATE (n_correct/k), not a single binary, so
    McNemar no longer applies. We resample tasks (the independent unit — avoids
    pseudo-replication across the correlated samples within a task) and take the
    2.5/97.5 percentiles of the mean paired difference. Assumption-light.
    """
    if len(a_rates) != len(b_rates):
        raise ValueError("rate lists must be aligned (same tasks, same order)")
    n = len(a_rates)
    diffs = [b - a for a, b in zip(a_rates, b_rates)]
    mean_diff = sum(diffs) / n if n else 0.0
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        boots.append(s)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot)]
    return RateComparison(
        n_tasks=n,
        mean_a=sum(a_rates) / n if n else 0.0,
        mean_b=sum(b_rates) / n if n else 0.0,
        mean_diff=mean_diff, ci_lo=lo, ci_hi=hi,
        improved=sum(1 for d in diffs if d > 0),
        worsened=sum(1 for d in diffs if d < 0),
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
