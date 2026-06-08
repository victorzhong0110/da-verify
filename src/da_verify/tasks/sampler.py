"""Stratified sampler: pick a balanced ~40-question subset for the headline study.

WHY STRATIFY (plain language):
  If we grabbed 40 random questions we might land 30 easy + 2 hard, and our
  "C0 vs C2" comparison would say more about luck than about verification. We
  want the subset to span the same difficulty/complexity as the full set, so
  the effect we measure generalises.

STRATA WE BALANCE:
  - level         : easy / medium / hard  (the benchmark's own difficulty tag)
  - complexity    : single-answer vs multi-answer (more sub-answers = more ways
                    to be partially wrong = where verification should matter)

VERIFIABILITY-FIRST SCOPING (an earned decision, documented):
  The whole project's thesis is *trustworthy* verification. Categorical gold in
  DAEval is semantically inconsistent ('no'/'False'/'not normal'), so auto-
  grading it is noisy. For the HEADLINE 40 we therefore prefer robustly
  verifiable questions (numeric / clean list), and we record the excluded
  categorical-heavy ones rather than hiding the choice. `prefer_verifiable` can
  be turned off to draw a fully representative sample for a separate analysis.

Determinism: a fixed seed makes the subset reproducible. No Date/random global
state — we use a seeded random.Random instance.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from .loader import Task
from .verifier import infer_type


def complexity_bucket(task: Task) -> str:
    return "single" if task.n_subanswers == 1 else "multi"


def has_noisy_categorical(task: Task) -> bool:
    """True if any gold field is free-text categorical (the auto-eval hazard)."""
    return any(infer_type(g.value) == "categorical" for g in task.gold)


def stratified_sample(
    tasks: list[Task],
    n: int = 40,
    seed: int = 20260607,
    prefer_verifiable: bool = True,
) -> list[Task]:
    """Return ~n tasks balanced over level x complexity.

    Allocation is proportional to the full-set stratum sizes, so the subset
    mirrors the population. Within a stratum we (optionally) prefer robustly
    verifiable tasks, then fall back to the rest if a stratum is short.
    """
    rng = random.Random(seed)

    strata: dict[tuple[str, str], list[Task]] = defaultdict(list)
    for t in tasks:
        strata[(t.level, complexity_bucket(t))].append(t)

    total = len(tasks)
    # Proportional, then fix rounding drift so the parts sum to exactly n.
    raw = {k: len(v) / total * n for k, v in strata.items()}
    alloc = {k: int(v) for k, v in raw.items()}
    while sum(alloc.values()) < n:
        # give the leftover slots to the strata with the largest fractional part
        k = max(raw, key=lambda k: (raw[k] - alloc[k], len(strata[k])))
        alloc[k] += 1

    chosen: list[Task] = []
    for k, want in alloc.items():
        pool = list(strata[k])
        rng.shuffle(pool)
        if prefer_verifiable:
            clean = [t for t in pool if not has_noisy_categorical(t)]
            noisy = [t for t in pool if has_noisy_categorical(t)]
            pool = clean + noisy  # clean first, noisy as fallback
        chosen.extend(pool[:want])

    return sorted(chosen, key=lambda t: t.id)


def save_subset(tasks: list[Task], path: Path, meta: dict | None = None) -> None:
    """Persist only the chosen ids (+ provenance) so the subset is reproducible."""
    payload = {
        "meta": meta or {},
        "ids": [t.id for t in tasks],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_subset_ids(path: Path) -> list[int]:
    return json.loads(path.read_text(encoding="utf-8"))["ids"]
