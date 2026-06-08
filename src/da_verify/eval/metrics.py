"""Capability vs reliability metrics over repeated samples.

WHY we run each task k times (not once):
  A single run conflates "can the model do this at all?" with "does it do it
  *reliably*?" Verification is expected to help reliability most, so we must be
  able to see that axis. With k samples per task we get two distinct numbers:

  - pass@k  : prob that AT LEAST ONE of k samples is correct  -> CAPABILITY
              (can it get there if it tries k times?)
  - pass^k  : were ALL k samples correct                       -> RELIABILITY
              (does it get there every time?)

  pass@1 (= mean per-sample correctness) is the plain accuracy.

pass@k uses the unbiased estimator from the HumanEval paper (Chen et al. 2021):
  pass@k = 1 - C(n-c, k) / C(n, k)
where n = samples drawn, c = correct samples. This is less biased than the naive
"1 if any correct" when n is small — it estimates the true pass@k from n>=k draws.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples of which c are correct (Chen et al. 2021)."""
    if k > n:
        raise ValueError(f"pass@{k} needs n>={k} samples, got n={n}")
    if c > n:
        raise ValueError(f"c={c} correct cannot exceed n={n} samples")
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0  # impossible to draw k all-wrong -> guaranteed a correct one
    return 1.0 - comb(n - c, k) / comb(n, k)


def reliability(n: int, c: int) -> float:
    """pass^n: 1.0 iff every sample was correct (strict consistency)."""
    return 1.0 if (n > 0 and c == n) else 0.0


@dataclass(frozen=True)
class TaskScore:
    id: int
    level: str
    n_samples: int
    n_correct: int
    n_format_ok: int          # samples that emitted all required @name fields
    n_candidate: int          # samples that emitted ANY answer field

    @property
    def pass_at_1(self) -> float:
        return self.n_correct / self.n_samples if self.n_samples else 0.0

    def pass_at(self, k: int) -> float:
        return pass_at_k(self.n_samples, self.n_correct, k)

    @property
    def is_reliable(self) -> float:
        return reliability(self.n_samples, self.n_correct)


def aggregate(scores: list[TaskScore], k: int) -> dict:
    """Roll task scores up into headline numbers + a by-level breakdown."""
    n = len(scores)
    if n == 0:
        return {}

    def _mean(f):
        return sum(f(s) for s in scores) / n

    ns = {s.n_samples for s in scores}
    out = {
        "n_tasks": n,
        # report the set (not a lie) if tasks ended up with different sample counts
        "samples_per_task": scores[0].n_samples if len(ns) == 1 else sorted(ns),
        "pass@1": _mean(lambda s: s.pass_at_1),
        f"pass@{k}": _mean(lambda s: s.pass_at(k)),
        "reliability(pass^k)": _mean(lambda s: s.is_reliable),
        "format_ok_rate": _mean(lambda s: s.n_format_ok / s.n_samples),
        "candidate_rate": _mean(lambda s: s.n_candidate / s.n_samples),
    }
    by_level: dict[str, dict] = {}
    for level in sorted({s.level for s in scores}):
        sub = [s for s in scores if s.level == level]
        by_level[level] = {
            "n": len(sub),
            "pass@1": sum(s.pass_at_1 for s in sub) / len(sub),
            f"pass@{k}": sum(s.pass_at(k) for s in sub) / len(sub),
        }
    out["by_level"] = by_level
    return out
